# M0RGOUTHGBL (shared-PMA, e\_layers=1) 구조 상세 문서

대상 실험 파일: `conf/plan/M0_quick_combo_rgpv_sa_outfuse_sharedpma_hgatebias_linear_k321_e5_vt_el1_predlen.yaml`

이 문서는 위 plan에서 실제로 활성화된 `M0` 구조를 코드 기준으로 정리한다.  
핵심 구현 파일은 `src/itransformer/models/m0.py`이고, 기본 모델 설정은 `conf/model/M0.yaml`를 따른다.

---

## 1) 모델 철학 (왜 이런 구조인가)

이 설정의 철학은 다음 3가지다.

1. **다중 스케일 슬롯으로 표현 분해**
   - 시계열을 하나의 길이에서만 보지 않고, 여러 patch scale에서 슬롯을 뽑아 표현한다.
   - 설정상 `k_per_scale=[3,2,1]`이므로 작은/중간/전역 스케일의 정보를 서로 다른 슬롯 집합으로 유지한다.

2. **표현 공간이 아니라 출력 공간에서 보정**
   - `slot_fuse.mode=residual_gated_output`을 사용한다.
   - 즉, 슬롯 표현을 먼저 각각 예측값으로 변환한 뒤, baseline 예측을 보정하는 형태로 합성한다.
   - 설계 의도는 baseline 안정성을 유지하면서 extra slot은 correction 역할만 하게 하는 것이다.

3. **긴 horizon 보정 강도를 사전 제어**
   - `horizon_gate`를 bias 기반으로 두고 `-2.0 -> -8.0` 선형 초기화한다.
   - 긴 예측 구간으로 갈수록 gate가 더 작게 시작되어 correction 폭주를 완화하도록 유도한다.

---

## 2) 이 plan에서 실제 활성화된 핵심 오버라이드

- `model.e_layers=1`
- `model.multislot.k_per_scale=[3,2,1]`
- `model.multislot.slotizer.mode=pma`
- `model.multislot.slotizer.share_across_scales=true`
- `model.multislot.pma.kv_include_seeds=true`
- `model.multislot.slot_attn.enabled=true`
- `model.multislot.slot_attn.mode=post`
- `model.multislot.slot_fuse.mode=residual_gated_output`
- `model.multislot.slot_fuse.horizon_gate.enabled=true`
- `model.multislot.slot_fuse.horizon_gate.mode=bias`
- `model.multislot.slot_fuse.horizon_gate.init_schedule=linear`
- `model.multislot.slot_fuse.horizon_gate.init_start=-2.0`
- `model.multislot.slot_fuse.horizon_gate.init_end=-8.0`
- `model.d_model=512`
- `model.multislot.d_model=256`

학습 관련:

- `train.epochs=20`, `batch_size=128`, `patience=0`
- `train.ssl_ckpt_path=null` (scratch fine-tune)
- `train.val_flag=test` (검증에 test split 사용)
- `optim.name=adam`, `optim.scheduler=onecycle`

---

## 3) 전체 데이터 흐름 (end-to-end)

표기:

- `B`: batch size
- `L`: lookback length (`data.seq_len`)
- `N`: 입력 변수 수
- `C`: covariate 수 (`x_mark`)
- `T`: 토큰 수 (`N` 또는 `N+C`)
- `K`: 전체 슬롯 수 (`sum(k_per_scale)=6`)
- `d_in`: multislot 내부 차원 (`model.multislot.d_model=256`)
- `d`: encoder 차원 (`model.d_model=512`)
- `H`: 예측 길이 (`data.pred_len`, sweep 대상)

흐름:

1. 입력 정규화 (`use_norm=true`)
   - `x_enc: [B, L, N]`를 변수별 mean/stdev로 정규화.

2. Multi-scale slot embedding (`enc_embedding`)
   - 입력을 토큰축으로 전치/결합해서 `x: [B, T, L]`.
   - scale별 patchify + patch projection + scale별 positional embedding.
   - scale별 temporal conv 적용.
   - PMA slotizer로 `K_p`개 슬롯 생성.
   - scale embedding 추가 후 모든 scale의 슬롯 concat.
   - 내부 차원 `256`을 encoder 차원 `512`로 투영.
   - 결과 `z: [B, T, K, d]`.

3. Encoder (`e_layers=1`)
   - `z`를 `B*K`로 펼쳐 `Encoder` 통과 후 다시 복원.
   - 변수 토큰만 취득: `enc_x: [B, N, K, d]`.

4. Post slot attention
   - `slot_attn.mode=post`이므로 encoder 후 슬롯축 self-attention 1회.
   - 각 변수별로 슬롯들만 섞는다.

5. Output-space residual-gated fuse
   - 슬롯마다 `Linear(d -> H)` head를 독립 적용해 `pred_slots: [B, N, K, H]`.
   - baseline 슬롯(`p=L` scale의 첫 슬롯)을 기준으로 extras의 delta를 계산.
   - alpha(슬롯별 mixing) + horizon gate(시점별 보정 강도)로 correction 합성.
   - `fused_pred: [B, N, H]`.
   - 축 변환 후 `dec_out: [B, H, N]`.

6. 역정규화
   - 1번에서 쓴 stdev/mean으로 출력을 원 스케일로 복원.

---

## 4) 모듈별 상세 설명

## 4.1 MultiScaleSlotEmbedding

구현 위치: `src/itransformer/models/m0.py`의 `MultiScaleSlotEmbedding`.

### (a) patchify + pad 정책

- 각 scale `p`마다 `L_pad = ceil(L/p)*p`.
- 부족한 tail은 `pad_value`(기본 0.0)로 right pad.
- patch tensor는 `[B, T, m_p, p]` (`m_p=L_pad/p`).

### (b) patch projection + pos emb

- `Linear(p -> d_in)`으로 patch를 임베딩.
- scale별 학습형 positional embedding `pos_emb[p]: [m_p, d_in]` 추가.

### (c) temporal conv

- `CausalConvBlock1D`를 patch index 축(`m_p`)에 적용.
- 현재 plan은 기본값 사용:
  - `type=standard`
  - `kernel_size=3`, `layers=2`, `padding_mode=causal`
- 출력은 residual로 입력에 더해진다.

### (d) PMA slotizer (shared across scales)

- `slotizer.mode=pma`.
- `slotizer.share_across_scales=true`라서 PMA 모듈 가중치는 scale 간 공유.
- 단, **seed는 scale별/슬롯별로 별도 파라미터**다.
- `pma.kv_include_seeds=true`:
  - Query = seed
  - Key/Value = `[patch_tokens ; seeds]`
  - 즉 seed-to-seed 상호작용을 K/V에 포함.

### (e) scale embedding

- slotizer 출력 슬롯에 scale별 `scale_emb[p]`를 더해 scale identity를 주입.

### (f) out projection

- 내부 `d_in=256`과 encoder `d=512`가 다르므로 `Linear(256 -> 512)` 적용.

---

## 4.2 Encoder + Slot Attention

### (a) Encoder

- `e_layers=1`.
- 기본 iTransformer encoder block(FullAttention 기반).
- 이 설정에서는 `cond_ln` 비활성이라 조건부 LN 경로는 사용 안 함.

### (b) Post slot attention

- `slot_attn.enabled=true`, `mode=post`.
- encoder가 끝난 뒤 `enc_x: [B, N, K, d]`에서 슬롯축 `K`에 self-attention 적용.
- 의미:
  - 동일 변수 내에서 서로 다른 슬롯 표현 간 정보 교환.

---

## 4.3 Residual-Gated Output Fuse

구현 위치: `ITransformerM0.forecast()`의 `fuse_mode == residual_gated_output`.

### (a) per-slot output head

- `K`개 슬롯 각각에 독립 projector `Linear(d -> H)` 적용.
- 결과 `pred_slots: [B, N, K, H]`.

### (b) baseline / extras

- baseline index는 global scale(`p=L`)의 첫 슬롯.
- `baseline: [B, N, H]`.
- `extras: [B, N, K-1, H]`.
- `delta = extras - baseline.unsqueeze(2)`.

### (c) alpha (slot mixing 가중치)

- `horizon_alpha`가 꺼져 있으므로 `ResidualGatedFuse.compute_alpha_g(enc_x)` 사용.
- alpha shape은 `[B, N, K-1, 1]` (horizon 공통).
- 즉 슬롯별 혼합비는 시점별로 달라지지 않고, correction 크기만 horizon gate로 조절된다.

### (d) horizon gate (bias mode)

- `horizon_gate.mode=bias`이므로 입력 의존이 아니라 파라미터 벡터 기반.
- 초기화는 `linear` 스케줄: `-2.0 -> -8.0` (`H` 길이 선형).
- 실제 게이트:
  - `g_h = sigmoid(bias_h)`, shape `[1,1,H]`.
  - 짧은 horizon은 상대적으로 큰 gate, 긴 horizon은 매우 작은 gate로 시작.

### (e) 최종 수식

- `corr = sum_i alpha_i * delta_i`  (i는 extra 슬롯)
- `y_hat = baseline + g_h * corr`

이 구조는 "baseline + horizon-wise correction" 형태다.

---

## 5) 현재 설정에서 비활성인 주요 옵션

다음은 `M0`에 구현은 있지만 이 plan에서는 꺼져 있다.

- `multislot.diversity.enabled=false` (attention diversity 정규화 미사용)
- `multislot.cond_ln.enabled=false`
- `slot_fuse.horizon_alpha.enabled=false`
- `slot_fuse.global_ma.enabled=false`
- `slot_fuse.extra_slot_dropout=0.0`
- `slotizer.post_slot_attn.enabled=false` (slotizer 직후 per-scale slot-attn 미사용)

---

## 6) 학습/평가 동작상의 해석 포인트

1. `train.val_flag=test`
   - validation이 test split에서 계산되므로, 일반적인 train/val 분리 실험과 해석이 다르다.

2. `patience=0`
   - early stopping 없이 끝까지 학습.

3. `onecycle` 스케줄
   - step 단위 LR 곡선이므로 epoch 내 변동이 크다.

4. output fuse의 안정성
   - 표현을 직접 합치는 방식보다 보정 해석이 직관적.
   - 다만 alpha가 horizon 공통이므로, horizon별 slot 조합까지 분화하려면 `horizon_alpha` 옵션이 필요하다.

---

## 7) 디버깅 체크리스트 (이 설정 전용)

1. baseline 슬롯 인덱스 확인
   - `scales`에 `p=L`이 반드시 포함되어야 의도한 baseline이 잡힌다.

2. horizon gate 값 확인
   - 초기/학습 후 `sigmoid(output_hgate_bias)` 분포를 확인해 long horizon gate가 충분히 작아졌는지 점검.

3. alpha 편향 확인
   - `compute_alpha_g` 결과가 특정 extra 슬롯에 과도하게 쏠리는지 확인.

4. shared PMA 영향 확인
   - PMA 가중치는 공유되지만 seeds/pos/scale emb는 scale별이므로, 실제로 scale 분화가 되는지 attention/log로 확인.

5. slot-attn 효과 확인
   - `slot_attn.mode=post`가 off 대비 어떤 horizon에서 이득/손해인지 분리 검증.

---

## 8) 한 줄 요약

이 실험 설정은 **shared PMA + 1-layer encoder + post slot-attn + output-space residual correction + horizon-wise bias gate**를 결합한 구조이며, 목적은 multi-slot 정보를 유지하되 긴 horizon에서 correction 폭주를 gate 초기화로 제어하는 것이다.
