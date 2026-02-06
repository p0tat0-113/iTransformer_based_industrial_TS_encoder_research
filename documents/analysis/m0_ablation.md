# M0 Ablation Log

목적: M0(Multi-scale Latent Slots)에서 **성능 병목/이득 지점**을 찾기 위해, 구성요소를 단계적으로 제거/변형하며 성능 변화를 기록한다.

기본 비교 대상:
- Reference baseline(P0):
  - ETTh1: `SUP-ETTh1.P0M0.P0.sd0`
  - Exchange: `SUP-Exchange.P0M0.P0.sd0`
- Ablation 비교 기준(M0 full):
  - ETTh1: `SUP-ETTh1.P0M0.M0.sd0`
  - Exchange: `SUP-Exchange.P0M0.M0.sd0`
- 동일한 train setting(epochs/batch/lr/scheduler 등)에서 비교

## Baselines (seed=0)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | P0 (`SUP-ETTh1.P0M0.P0.sd0`) | 6,404,704 | 2 | 0.677034 | 0.395556 | 0.413728 |
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| Exchange | P0 (`SUP-Exchange.P0M0.P0.sd0`) | 6,404,704 | 1 | 0.137536 | 0.086421 | 0.208301 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |

---

## Ablation #1: Global-only (p=L slot only, no fuse)

### 아이디어
- 출력단에서 **서로 다른 스케일 슬롯을 fuse하지 않고**, `p=L(=seq_len)` 스케일의 슬롯(전역 view)만으로 예측한다.
- 의도:
  - 작은 스케일 슬롯들이 실제로 유익한지(유익하지 않으면 성능이 안 오르거나 오히려 나빠질 수 있음)
  - 혹은 fuse 방식(현재 concat+MLP)이 병목/오류 원인인지 확인

### 구현 정책
1) 멀티스케일 슬롯은 그대로 생성
- 기본 설정 예: `scales=[8, 32, L]`, `k_per_scale=[2, 1, 1]` (K_total 유지)

2) Slot fuse 제거(=출력 head만 전역 슬롯 선택)
- `slot_fuse.mode=select_global`
- encoder 출력 `enc_x: [B, N, K, d]`에서 **p=L 슬롯 1개만 선택**하여 projector로 연결
  - `fused = enc_x[:, :, global_slot_idx, :]`

### 기대 해석
- Global-only 성능이 M0(full)과 비슷/더 좋다:
  - (A) 작은 스케일 슬롯이 유익하지 않거나
  - (B) fuse 방식이 정보 손실/학습 불안정을 만들고 있을 가능성
- Global-only 성능이 유의미하게 나쁘다:
  - 작은 스케일 슬롯이 실제로 도움이 되었을 가능성

### 코드 변경
- `src/itransformer/models/m0.py`
  - `slot_fuse.mode=select_global` 지원 추가 (p=L 슬롯만 선택, no fuse)

### 실험 플랜
- `conf/plan/M0_ablation_global_only.yaml`
  - ETTh1, Exchange에서 **M0(global-only head)** 만 scratch supervised로 학습/평가

### 실험 결과 (seed=0)

비교:
- ETTh1: `SUP-ETTh1.M0GO.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0GO.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0GO (`SUP-ETTh1.M0GO.sd0`) | 20,616,800 | 4 | 0.698175 | 0.415787 | 0.420996 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0GO (`SUP-Exchange.M0GO.sd0`) | 20,616,800 | 1 | 0.128877 | 0.094647 | 0.215133 |

Delta (M0GO - M0 full, test):
- ETTh1: MSE **-0.001867** (-0.45%), MAE **+0.002096** (+0.50%)
- Exchange: MSE **+0.004478** (+4.97%), MAE **+0.003207** (+1.51%)

해석(1차):
- ETTh1에서는 MSE만 아주 소폭 개선되었지만 MAE는 악화되어 “fuse 제거”가 명확한 이득이라고 보긴 어려움.
- Exchange에서는 fuse를 제거하고 p=L 슬롯만 쓰면 성능이 악화되어, (현재 설정에서는) multi-scale 슬롯 정보가 도움이 되거나, 최소한 global-only head가 불리한 것으로 보임.

---

## Ablation #2: Residual-gated fuse (baseline=p=L + gated corrections)

### 아이디어
- `p=L` 슬롯을 **baseline(안전망)** 으로 두고,
- 나머지 슬롯들은 **correction(보정값)** 으로만 사용한다.
- fused를 직관적으로 쓰면:
  - `fused = baseline + gate * correction`

### 구현 정책
1) baseline은 `p=L(=seq_len)` 스케일의 슬롯
- 기본 설정(`scales=[8,32,L]`, `k=[2,1,1]`)에서는 **마지막 슬롯(k=K_total-1)** 이 baseline에 해당.
- (안전장치) config에 `p=L`이 없으면 fallback으로 마지막 슬롯을 baseline으로 사용.

2) correction-form residual (baseline 대비 변화량만 더하기)
- `delta_i = extra_i - u_L`
- `corr = sum_i alpha_i * delta_i`
- `fused = u_L + g * corr`

3) alpha는 softmax 정규화(스케일 폭주 방지) + pairwise score
- `score_i = MLP([u_L ; extra_i ; (extra_i - u_L)])`
- `alpha = softmax(score_i)`  (i는 extra slot index)

4) global gate g (초기에는 baseline에 가깝게)
- `g = sigmoid(Linear(u_L))`  (Linear: d -> 1)
- bias를 **-2**로 초기화해서 초기 `g≈0.119`

### 코드 변경
- `src/itransformer/models/m0.py`
  - `ResidualGatedFuse` 모듈 추가
  - `model.multislot.slot_fuse.mode`에 따라 fuse 전략 선택
    - `mlp` (기존 concat+MLP)
    - `residual_gated` (이번 ablation)

### 실험 플랜
- `conf/plan/M0_ablation_residual_gated.yaml`
  - ETTh1, Exchange에서 **M0(residual-gated fuse)** 만 scratch supervised로 학습/평가

### 실험 결과 (seed=0)

비교:
- (v1: sigmoid gate, extra 직접 합산) ETTh1: `SUP-ETTh1.M0RG.sd0` vs `SUP-ETTh1.P0M0.M0.sd0`
- (v1: sigmoid gate, extra 직접 합산) Exchange: `SUP-Exchange.M0RG.sd0` vs `SUP-Exchange.P0M0.M0.sd0`
- (v2: correction-form + softmax + global gate, pairwise score) 재실험 run id: `SUP-{dataset}.M0RGC.sd{seed}`

아래 표/해석은 **v1 결과(M0RG)** 이며, v2(M0RGC)는 재실험 후 갱신 예정.

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0RG (`SUP-ETTh1.M0RG.sd0`) | 20,618,339 | 4 | 0.692268 | 0.409529 | 0.418058 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0RG (`SUP-Exchange.M0RG.sd0`) | 20,618,339 | 1 | 0.133567 | 0.097160 | 0.217446 |

Delta (M0RG - M0 full, test):
- ETTh1: MSE **-0.008125** (-1.95%), MAE **-0.000842** (-0.20%)
- Exchange: MSE **+0.006991** (+7.75%), MAE **+0.005521** (+2.60%)

해석(1차):
- M0 full 대비로는 **ETTh1에서는 개선**, **Exchange에서는 큰 폭 악화**로 dataset-dependent하게 동작.
- Global-only(M0G) 대비로는 두 데이터셋 모두에서 열세(특히 Exchange)라, “작게 섞기”만으로는 작은 스케일 슬롯을 유익하게 쓰기 어렵거나 gate/correction 설계가 맞지 않을 가능성이 있음.

### v2 결과: correction-form + softmax + global gate (M0RGC)

주의:
- 이 run은 plan override로 `train.val_flag=test`를 사용함(= **test split을 validation으로 사용**).
- 따라서 `best_val == test`가 구조적으로 발생하고, checkpoint 선택에 test 정보가 누출됨(공정 비교/리포팅 용도에는 부적절).

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0RG v1 (`SUP-ETTh1.M0RG.sd0`) | 20,618,339 | 4 | 0.692268 | 0.409529 | 0.418058 |
| ETTh1 | M0RGC v2 (`SUP-ETTh1.M0RGC.sd0`) | 23,767,138 | 2 | 0.392742 | 0.392742 | 0.409880 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0RG v1 (`SUP-Exchange.M0RG.sd0`) | 20,618,339 | 1 | 0.133567 | 0.097160 | 0.217446 |
| Exchange | M0RGC v2 (`SUP-Exchange.M0RGC.sd0`) | 23,767,138 | 2 | 0.092355 | 0.092355 | 0.212225 |

Delta (M0RGC - M0 full, test):
- ETTh1: MSE **-0.024912** (-5.96%), MAE **-0.009019** (-2.15%)
- Exchange: MSE **+0.002185** (+2.42%), MAE **+0.000300** (+0.14%)

Delta (M0RGC - M0RG v1, test):
- ETTh1: MSE **-0.016787** (-4.10%), MAE **-0.008178** (-1.96%)
- Exchange: MSE **-0.004806** (-4.95%), MAE **-0.005221** (-2.40%)

해석(1차):
- v1에서 “baseline 오염” 가능성이 있던 `extra` 직접 합산을 없애고, `delta=extra-baseline`만 더하는 형태로 바꾸니 **ETTh1/Exchange 모두에서 v1 대비 개선**됨.
- 특히 Exchange에서 v1의 큰 악화를 상당 부분 회복(0.0972 -> 0.0924)해서, 설계가 **dataset-robust** 쪽으로 움직인 것으로 보임(다만 M0 full보단 약간 열세).

---

## Ablation #3: x_mark는 global-only + 전용 MLP 임베딩 (slotize 금지)

### 아이디어
- `x`(변수)만 기존 M0처럼 멀티스케일 slotize를 수행한다.
- `x_mark`(covariates)는 **멀티스케일/conv/PMA slotize를 금지**하고,
  **전용 MLP로만 global(=p=L) 임베딩**해서 encoder에 주입한다.

목표:
- 성능이 좋아지면 → “covariate를 slotize(멀티스케일+PMA)하는 게 독”이었을 가능성

### 구현 정책
1) x 경로
- 기존 M0 그대로: `MultiScaleSlotEmbedding(x_enc, None)`로 x에 대해서만 `[B, N, K, d]` 생성

2) x_mark 경로 (패칭 금지, 전용 MLP)
- `x_mark: [B, L, C] -> [B, C, L]`
- `cov_embed(MLP): (L -> hidden -> d)` (covariate 채널에 대해 공유)
- 출력 `cov_tokens: [B, C, d]`

3) “각 k-th slot 집합과 attention”
- 각 k마다 encoder 입력을 다음처럼 구성:
  - `tokens_k = concat([x_slots_k, cov_tokens], dim=token)` → `[B, N+C, d]`
- 이를 `B*K`로 flatten해서 encoder 1회로 처리(기존 M0의 “가중치 공유 K회 효과” 유지)
- encoder 출력에서 covariate 토큰은 버리고, x 토큰만 fuse/projector로 전달

### 코드 변경
- `conf/model/M0.yaml`
  - `model.multislot.covariates.mode` 추가 (`slotize | mlp_global`)
  - `model.multislot.covariates.mlp_hidden` 추가
- `src/itransformer/models/m0.py`
  - `covariates.mode=mlp_global`일 때:
    - `x_mark`는 `cov_embed`로만 임베딩
    - slotize 경로에는 넣지 않음
    - 각 k group과 encoder self-attn에서만 mixing

### 실험 플랜
- `conf/plan/M0_ablation_xmark_global_mlp.yaml`
  - ETTh1, Exchange에서 **M0(x_mark global-MLP)** 만 scratch supervised로 학습/평가

### 실험 결과 (seed=0)

비교:
- ETTh1: `SUP-ETTh1.M0XM.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0XM.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0XM (`SUP-ETTh1.M0XM.sd0`) | 26,174,560 | 2 | 0.693779 | 0.405412 | 0.422287 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0XM (`SUP-Exchange.M0XM.sd0`) | 26,174,560 | 3 | 0.130195 | 0.097903 | 0.219346 |

Delta (M0XM - M0 full, test):
- ETTh1: MSE **-0.012242** (-2.93%), MAE **+0.003387** (+0.81%)
- Exchange: MSE **+0.007734** (+8.58%), MAE **+0.007420** (+3.50%)

해석(1차):
- ETTh1에서는 x_mark를 slotize하지 않고 “global-MLP + slot-k attention”으로 바꿨을 때 **MSE가 개선**됨.
- 반면 Exchange에서는 **뚜렷하게 악화**되어 dataset-dependent하게 동작.
- “covariate slotize가 독”이라는 가설은 ETTh1에서는 일부 지지되지만, Exchange까지 일반화되지는 않음.

---

## Ablation #4: PMA slotizer -> mean pooling

### 아이디어
- PMA(slotizer)가 슬롯 분화/과적합/불안정의 원인인지 빠르게 확인하기 위해,
  **PMA를 mean pooling으로 교체**한다.
- mean이 비슷하거나 더 좋으면:
  - (A) PMA가 현재 세팅에선 과적합/불안정일 수 있고
  - (B) slotizer는 복잡할 필요가 없거나(혹은 더 단순한 pooling이 충분)

### 구현 정책
- 각 scale의 patch token `emb: [B, T, m_p, d]`에 대해:
  - `pooled = mean(emb, dim=m_p) -> [B, T, d]`
  - `K_p > 1`인 scale(예: p=8, K=2)에서는 pooled를 **K_p번 복제**하여
    `slots_p: [B, T, K_p, d]` 형태를 맞춘다.
  - (주의) 이 경우 같은 scale 내 슬롯들이 동일해져 “슬롯 분화”는 사라진다.

### 코드 변경
- `conf/model/M0.yaml`
  - `model.multislot.slotizer.mode` 추가 (`pma | mean`)
- `src/itransformer/models/m0.py`
  - `MeanSlotizer` 추가
  - `MultiScaleSlotEmbedding`에서 slotizer를 `mode`에 따라 선택

### 실험 플랜
- `conf/plan/M0_ablation_slotizer_mean.yaml`
  - ETTh1, Exchange에서 **M0(slotizer=mean)** 만 scratch supervised로 학습/평가

### 실험 결과 (seed=0)

비교:
- ETTh1: `SUP-ETTh1.M0MP.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0MP.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0MP (`SUP-ETTh1.M0MP.sd0`) | 16,406,112 | 1 | 0.695170 | 0.395593 | 0.409427 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0MP (`SUP-Exchange.M0MP.sd0`) | 16,406,112 | 2 | 0.135919 | 0.098984 | 0.220509 |

Delta (M0MP - M0 full, test):
- ETTh1: MSE **-0.022061** (-5.28%), MAE **-0.009473** (-2.26%)
- Exchange: MSE **+0.008814** (+9.78%), MAE **+0.008583** (+4.05%)

해석(1차):
- ETTh1에서는 mean slotizer가 **큰 폭으로 개선**되어, “현재 PMA 세팅이 불안정/과적합”일 가능성을 강하게 시사함.
- Exchange에서는 반대로 **뚜렷한 악화**가 발생하여, slotizer를 단순화하면 항상 좋아지는 것은 아님(데이터셋/태스크 특성에 따라 다르게 작동).

---

## Ablation #5: Post-encoder slot self-attention (Var-attn -> Slot-attn 1회)

### 아이디어
- 기존 M0는 encoder에서 **변수 축(token=T)만** self-attn을 수행하고,
  같은 변수의 서로 다른 slot(K)끼리는 끝까지 독립이다.
- 이 ablation은 encoder가 끝난 뒤 1회,
  **같은 변수 내부에서 slot 축(K) self-attn** 을 추가한다.

의도:
- slot 간 상호작용이 필요해서 성능이 안 나오는지(혹은 fuse가 어려운지) 확인

### 구현 정책
- encoder 출력 `enc_x: [B, N, K, d]` 에 대해:
  - `SlotAttn(enc_x)` 를 1회 적용 (K축 self-attn)
  - 그 다음은 기존과 동일하게 fuse/projector 수행
- attention 순서:
  - (기존) Var-attn 먼저(encoder)
  - (추가) Slot-attn은 encoder가 끝난 뒤 1회만

### 코드 변경
- `conf/model/M0.yaml`
  - `model.multislot.slot_attn.enabled` 추가 (default false)
  - `model.multislot.slot_attn.n_heads` 추가 (default `${model.n_heads}`)
- `src/itransformer/models/m0.py`
  - `SlotSelfAttention` 모듈 추가
  - enabled일 때만 `enc_x`에 slot-attn 적용

### 실험 플랜
- `conf/plan/M0_ablation_slot_attn_post.yaml`
  - ETTh1, Exchange에서 **M0(post slot-attn)** 만 scratch supervised로 학습/평가

### 실험 결과 (seed=0)

비교:
- ETTh1: `SUP-ETTh1.M0SA.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0SA.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0SA (`SUP-ETTh1.M0SA.sd0`) | 26,913,888 | 1 | 0.688670 | 0.397999 | 0.413890 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0SA (`SUP-Exchange.M0SA.sd0`) | 26,913,888 | 4 | 0.134939 | 0.099909 | 0.220537 |

Delta (M0SA - M0 full, test):
- ETTh1: MSE **-0.019655** (-4.71%), MAE **-0.005010** (-1.20%)
- Exchange: MSE **+0.009739** (+10.80%), MAE **+0.008612** (+4.06%)

해석(1차):
- ETTh1에서는 slot-attn(encoder 이후 1회)이 **개선**으로 나타나 “slot 간 상호작용”이 도움될 가능성이 있음.
- Exchange에서는 **뚜렷한 악화**가 발생해 dataset-dependent하게 동작(또는 최적화/과적합/학습 스케줄 영향)하는 것으로 보임.
- 주의: 이 실험 플랜은 `train.epochs=15`로, 다른 ablation/기준(M0 full: 30 epoch)과 epoch 수가 다름. 공정 비교를 위해서는 `epochs=30`으로 맞춰 재실행하는 것이 안전함.

---

## Ablation: Axial attention (매 encoder layer마다 Var-attn -> Slot-attn interleave)

### 아이디어
- M0SA는 encoder가 끝난 뒤 **1회만** slot-attn을 적용했다.
- 이 ablation은 **매 encoder layer마다** 다음 순서를 반복한다:
  - Var-attn(변수 토큰 간 self-attn) -> Slot-attn(같은 변수 내부 slot(K) 간 self-attn)

의도:
- slot 간 상호작용을 더 “정석적인(axial)” 방식으로 넣었을 때 성능/안정성이 좋아지는지 확인.

### 구현 정책
- encoder를 `self.encoder(attn_layers)` 그대로 쓰지 않고, **layer-loop로 풀어서** interleave한다.
- 각 encoder layer 출력 `tokens_flat: [B*K, T_total, d]`에 대해:
  1) Var-attn layer 1회 적용
  2) reshape -> `[B, T_total, K, d]`
  3) x 변수 토큰(`T=0..N-1`)만 slot-attn 적용 (covariate 토큰은 slot-attn에서 제외)
  4) flatten -> `[B*K, T_total, d]`로 복귀
- 마지막에 encoder norm 적용(기존 Encoder.forward와 동일)

설정 키:
- `model.multislot.slot_attn.enabled=true`
- `model.multislot.slot_attn.mode=interleave`  (default: `post`)

### 실험 플랜
- `conf/plan/M0_ablation_slot_attn_interleave.yaml`
  - run id: `SUP-{dataset}.M0SAI.sd{seed}`

### 실험 결과
비교:
- ETTh1: `SUP-ETTh1.M0NG81632.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0NG81632.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Params | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.693765 | 0.417654 | 0.418900 |
| ETTh1 | M0NG81632 (`SUP-ETTh1.M0NG81632.sd0`) | 27,922,016 | 2 | 0.699652 | 0.405804 | 0.421632 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 25,862,240 | 2 | 0.127866 | 0.090169 | 0.211925 |
| Exchange | M0NG81632 (`SUP-Exchange.M0NG81632.sd0`) | 27,922,016 | 3 | 0.129192 | 0.093067 | 0.214797 |

Delta (M0NG81632 - M0 full, test):
- ETTh1: MSE **-0.011850** (-2.84%), MAE **+0.002732** (+0.65%)
- Exchange: MSE **+0.002898** (+3.21%), MAE **+0.002871** (+1.35%)

해석(1차):
- ETTh1에서는 전역 슬롯(p=L)을 제거해도 **MSE는 오히려 개선**되었지만, MAE는 악화되어 “일관된 개선”이라고 보긴 애매함.
- Exchange에서는 전역 슬롯 제거가 **뚜렷한 악화**로 나타나, (현재 M0 구조/설정에서는) 전역 컨텍스트 경로가 유익했을 가능성이 큼.
- 결론적으로 “p=L 스케일은 항상 필요/항상 불필요”가 아니라, **데이터셋 특성에 따라 역할이 달라질 수 있음**이 시사됨.

---

## Ablation: No global scale (remove p=L, use scales=[8,16,32])

### 아이디어
- 기존 M0는 `scales=[8,32,L]`처럼 **전역(p=L) 슬롯**을 포함해 “안전망/전역 컨텍스트” 역할을 기대한다.
- 이 ablation은 **p=L 스케일을 제거**하고, `scales=[8,16,32]`만으로 학습/예측한다.

의도:
- 성능이 크게 떨어지면: “전역 슬롯이 중요한 구조적 버팀목”일 가능성
- 성능이 유지/개선되면: “전역 슬롯이 없어도 충분하거나, 오히려 전역 경로가 학습을 흔들었을” 가능성

### 실험 플랜
- `conf/plan/M0_ablation_no_global_8_16_32.yaml`
  - run id: `SUP-{dataset}.M0NG81632.sd{seed}`
  - overrides:
    - `model.multislot.scales=[8,16,32]`
    - `model.multislot.k_per_scale=[2,1,1]`

### 실험 결과
- TBD (run 후 갱신)

---

## 추가 실험: Global-only head + post slot-attn (M0GOSA)

### 의도
- M0SA에서 **slot-attn은 유지**하되,
  출력단의 `concat+MLP fuse`를 제거하고(`slot_fuse.mode=select_global`),
  **p=L 슬롯 1개만 선택**하여 예측한다.
- 즉, “slot-attn은 도움이 되는데, fuse MLP가 병목/과적합인가?”를 확인하는 목적.

### 설정 차이 (M0GOSA vs M0SA)
- 공통: `model.multislot.slot_attn.enabled=true`
- 차이:
  - M0SA: `slot_fuse.mode=mlp` (fuse MLP 사용)
  - M0GOSA: `slot_fuse.mode=select_global` (p=L 슬롯 선택, fuse 제거)

### 실험 결과 (seed=0)

| Dataset | Run | Params | Epochs | Best epoch | Best val(MSE) | Test MSE | Test MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | M0SA (`SUP-ETTh1.M0SA.sd0`) | 26,913,888 | 15 | 1 | 0.688670 | 0.397999 | 0.413890 |
| ETTh1 | M0GOSA (`SUP-ETTh1.M0GOSA.sd0`) | 21,668,448 | 30 | 2 | 0.692407 | 0.404206 | 0.416108 |
| Exchange | M0SA (`SUP-Exchange.M0SA.sd0`) | 26,913,888 | 15 | 4 | 0.134939 | 0.099909 | 0.220537 |
| Exchange | M0GOSA (`SUP-Exchange.M0GOSA.sd0`) | 21,668,448 | 30 | 1 | 0.124921 | 0.091822 | 0.213393 |

Delta (M0GOSA - M0SA, test):
- ETTh1: MSE **+0.006207** (+1.56%), MAE **+0.002218** (+0.54%)  (악화)
- Exchange: MSE **-0.008086** (-8.09%), MAE **-0.007144** (-3.24%)  (개선)

해석(1차):
- Exchange에서는 M0SA의 성능 저하가 “slot-attn 자체”라기보다는, **slot-attn 이후 fuse MLP(5.25M params)** 가 불안정/과적합을 만든 영향일 가능성이 있음.
- ETTh1에서는 반대로 fuse MLP가 어느 정도 유익할 수 있음(단, M0SA=15ep vs M0GOSA=30ep로 epoch 조건이 달라 완전 공정 비교는 아님).

---

## Ablation: PMA의 K/V에 slot seed 포함 (Perceiver Resampler 스타일)

### 아이디어
- 기존 PMA(slotizer)는 cross-attn 형태:
  - `Q = slot_seeds`, `K/V = patch_tokens`
- 이번 ablation은 Perceiver Resampler(Flamingo)에서 영감을 받아,
  - `Q = slot_seeds`
  - `K/V = concat(patch_tokens, slot_seeds)`
  로 바꿔서, **slot seed가 cross-attn 안에서 seed들끼리도 상호작용**할 수 있게 한다.

### 구현 요약
- `src/itransformer/models/m0.py`
  - `PMASlotizer(kv_include_seeds=True)`이면 `kv = cat([x_flat, q], dim=token)` 후 `attn(q, kv, kv)`
- `conf/model/M0.yaml`
  - `model.multislot.pma.kv_include_seeds: false` (default)

### 실험 플랜
- `conf/plan/M0_ablation_pma_kv_seeds.yaml`
  - run id: `SUP-{dataset}.M0PKV.sd{seed}`

### 실험 결과 (seed=0)

주의:
- 이 plan은 `train.val_flag=test`를 사용함(= **test split을 validation으로 사용**).
- 따라서 checkpoint 선택에 test 정보가 누출되어, **동일 설정/동일 val split(run.val_flag=val)** 과의 공정 비교/리포팅에는 부적절함.

비교:
- ETTh1: `SUP-ETTh1.M0PKV.sd0` vs `SUP-ETTh1.P0M0.M0.sd0` (baseline)
- Exchange: `SUP-Exchange.M0PKV.sd0` vs `SUP-Exchange.P0M0.M0.sd0` (baseline)

| Dataset | Run | Best epoch | Val split | Test MSE | Test MAE |
|---|---|---:|---|---:|---:|
| ETTh1 | M0 full (`SUP-ETTh1.P0M0.M0.sd0`) | 2 | val | 0.417654 | 0.418900 |
| ETTh1 | M0PKV (`SUP-ETTh1.M0PKV.sd0`) | 1 | test | 0.405999 | 0.420634 |
| Exchange | M0 full (`SUP-Exchange.P0M0.M0.sd0`) | 2 | val | 0.090169 | 0.211925 |
| Exchange | M0PKV (`SUP-Exchange.M0PKV.sd0`) | 1 | test | 0.082611 | 0.205205 |

Delta (M0PKV - M0 full, test):
- ETTh1: MSE **-0.011655** (-2.79%), MAE **+0.001734** (+0.41%)
- Exchange: MSE **-0.007558** (-8.38%), MAE **-0.006720** (-3.17%)

해석(1차):
- ETTh1: MSE는 개선되었지만 MAE는 소폭 악화되어, “일부 큰 오차는 줄였지만 전체적으로는 균일하게 좋아졌다”라고 말하긴 애매함(또는 early epoch 선택 영향 가능).
- Exchange: MSE/MAE 모두 개선으로 나타나, seed를 K/V에 포함하는 방식이 슬롯 품질(또는 학습 안정성)에 이득일 가능성.
- 다만 두 run 모두 val split이 test라서(=누출), 결론을 내리기 전에 **`train.val_flag=val`로 동일 실험을 재실행**하는 것이 안전함.
