# M0 구현 기록 (Multi-scale Latent Slots)

이 문서는 `documents/analysis/new_model_struc.md` / `documents/analysis/m0_integration_plan.md` 를 기반으로,
현재 코드베이스에 **M0(iTransformerM0)** 를 실제로 통합/구현한 결과를 기록한다.

목표:
- 기존 훈련/평가 프레임워크(Hydra + orchestrator plan) 흐름을 그대로 유지
- `model=M0` override만으로 실험 가능
- 1차 구현에서는 meta 등 불필요한 failpoint를 제거하고 “모델 구조” 검증에 집중

---

## 1) 변경/추가된 파일

- 모델 구현
  - `src/itransformer/models/m0.py`
- 모델 설정
  - `conf/model/M0.yaml`
- 모델 팩토리/등록
  - `src/itransformer/models/factory.py`
  - `src/itransformer/models/__init__.py`

---

## 2) Hydra에서 M0 선택 방법

기본은 `conf/config.yaml`의 defaults를 따르며, 실험에서는 plan/run overrides로 아래처럼 선택한다.

- `model=M0`
- (권장) `metadata.enabled=false` (M0는 meta를 무시하도록 구현됨)

---

## 3) M0 입력/출력 계약 (기존 downstream.py와 호환)

- 입력
  - `x_enc`: `[B, L, N]`
  - `x_mark`: `[B, L, C]` (optional)
- 출력
  - `y_hat`: `[B, pred_len, N]`

주의:
- `meta_emb` 인자는 **호환성 유지를 위해 시그니처에 남겨두었지만**, M0 내부에서는 사용하지 않는다.

---

## 4) 모델 구조 개요(구현 기준)

### 4.1 Token 정의(순정 iTransformer와 동일)

1) `x_enc`를 token 축으로 뒤집기:
- `x_enc: [B, L, N] -> x: [B, N, L]`

2) `x_mark`가 있으면 token으로 concat:
- `x_mark: [B, L, C] -> mark: [B, C, L]`
- `x = concat([x, mark], dim=token) -> [B, T, L]` where `T=N+C`

이후 멀티스케일 임베딩/슬롯화는 **token별 독립 연산**으로 수행된다(변수 간 mixing 금지).

---

### 4.2 Multi-scale Slot Embedding (`MultiScaleSlotEmbedding`)

설정:
- `model.multislot.scales`: `[8, 32, L]` (L은 `${data.seq_len}`)
- `model.multislot.k_per_scale`: `[2, 1, 1]`
- `K_total = 4`

각 scale `p`에 대해:

1) right-pad(0) + patchify (non-overlap)
- `L_pad = ceil(L/p)*p`
- `x_p = pad_right(x, L_pad-L, value=0)`
- `m_p = L_pad/p`
- `patches: [B, T, m_p, p]` (`reshape`)

2) patch projection
- `patch_proj[p]: Linear(p -> d)`
- `emb: [B, T, m_p, d]`

3) patch positional embedding (learned)
- `pos_emb[p]: [ceil(L/p), d]`
- `emb += pos_emb[p][:m_p]` (broadcast)

4) temporal encoder (causal conv over m_p)
- token 간 mixing을 피하기 위해 `B*T`를 batch로 합친 뒤 conv 수행
- `CausalConvBlock1D`: `m_p==1`이면 bypass
- 출력 shape 유지: `[B, T, m_p, d]`

5) PMA slotizer (PMA + FFN/residual)
- `seeds[p]: [K_p, d]`
- `slots_p: [B, T, K_p, d]`

6) scale-id embedding
- `scale_emb[p]: [d]`
- `slots_p += scale_emb[p]`

마지막으로 scale들을 concat:
- `Z = cat(slots_p, dim=K) -> [B, T, K_total, d]`

---

### 4.3 Encoder 적용(가중치 공유, K회 효과)

원 설계는 “k 고정 후 encoder를 K번 반복 호출”이었고,
구현에서는 속도/단순성을 위해 아래와 같이 **batch flatten**으로 동일 효과를 만든다.

- `Z: [B, T, K, d]`
- `Z_flat = Z.permute(0,2,1,3).reshape(B*K, T, d)`
- `encoder(Z_flat)` 1회 호출
- reshape back: `U: [B, T, K, d]`

이 방식은:
- encoder 파라미터는 1세트만 사용(공유)
- slot 간 mixing은 발생하지 않음(서로 다른 k는 batch 차원으로 분리)

---

### 4.4 Slot Fuse + Projector

1) slot fuse (token별 MLP)
- `U: [B, T, K, d]`
- concat: `[B, T, K*d]`
- `SlotFuseMLP: Linear(K*d -> hidden -> d)`
- `fused: [B, T, d]`

2) projector
- `projector: Linear(d -> pred_len)`
- `dec: [B, T, pred_len]`
- `dec_out = dec.permute(0,2,1)[:, :, :N] -> [B, pred_len, N]`

---

## 5) use_norm 정책

순정 iTransformer(P0)와 동일:
- `x_enc`만 normalize/denormalize
- `x_mark`는 그대로 사용

---

## 6) meta 처리 정책(1차 구현)

요청에 따라 M0 코드에서 meta 관련 로직을 전부 제거했다.

- `meta_emb`는 인자로 받아도 무시됨(훈련 루프 호환성 목적)
- 향후 meta를 다시 넣을 경우, “slot/scale 구조”와 meta의 정렬 정책을 먼저 설계한 뒤 추가하는 것을 권장

---

## 7) 알려진 제약/리스크(1차 실험 해석 시 주의)

- padding mask를 1차에서 생략:
  - pad된 0 구간이 실제 입력처럼 처리됨(특히 p가 작을수록 마지막 patch에 영향을 줌)
  - 2차에서 padding mask 도입(또는 trim/left-pad 비교)을 고려
- 계산량:
  - encoder 비용이 `K_total` 배 증가하는 것이 정상(현재 K_total=4)

---

## 8) 스모크 테스트

간단히 end-to-end로 “학습이 돌아가는지” 확인하기 위한 plan:
- `conf/plan/M0_smoke.yaml`

이 plan은 ETTh1에서 `P0`와 `M0`를 각각 1 epoch만 supervised(scratch)로 실행한다.

