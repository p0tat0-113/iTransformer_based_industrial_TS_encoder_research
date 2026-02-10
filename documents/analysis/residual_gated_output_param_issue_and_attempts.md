# Residual Gated Output 파라미터 이슈 정리

## 1) 문제 정의

현재 이슈는 `slot_fuse.mode=residual_gated_output`에서 파라미터가 크게 증가한다는 점이다.  
특히 `M0_quick_combo_rgpv_sa_outfuse_sharedpma_hgatebias_linear_k321_e20_vt_el1_predlen` 계열에서,
`slot_fuse`와 `slot_projectors`가 전체 모델 크기를 크게 끌어올린다.

---

## 2) 왜 파라미터가 커지는가

`residual_gated_output`의 핵심 구조:

1. `baseline` 슬롯(`p=L`)의 예측을 생성
2. 나머지 `extras` 슬롯들의 예측도 각각 생성
3. `delta = extras - baseline`을 만들고
4. `alpha`(score MLP 기반)로 가중합한 correction을 만들고
5. `horizon gate`로 correction 크기를 제어해 최종 예측에 더함

여기서 파라미터를 크게 만드는 항목은:

- `score_mlp`의 첫 선형층: `Linear(3*d_model -> hidden)`
- 슬롯별 예측 헤드: `slot_projectors` (`K`개)

즉, **alpha 계산 MLP + 슬롯별 projector**가 핵심 비용이다.

---

## 3) 실제 파라미터 분해 (ETTh1, pred_len=96 기준)

기준 run: `SUP-ETTh1.M0RGOUTHGBL.sharedpma.el1.k321.e20.vt.pr96.sd0`

- Total: `10,371,074`
- `enc_embedding`: `2,670,848`
- `encoder`: `3,153,408`
- `slot_attn` (post): `1,051,648`
- `slot_fuse` (ResidualGatedFuse): `3,150,338`
- `slot_projectors`: `295,488`
- `projector`: `49,248`
- `output_hgate_bias`: `96`

`slot_fuse` 내부:

- `slot_fuse.score_mlp`: `3,149,825`
- `slot_fuse.gate`: `513`

즉, `slot_fuse` 대부분은 사실상 `score_mlp`가 차지한다.

### 3.1 수식으로 보는 크기

`d=512`, `hidden=2048`일 때:

- `Linear(3d -> hidden)`  
  = `1536 * 2048 + 2048`  
  = `3,147,776`
- `Linear(hidden -> 1)`  
  = `2048 + 1`  
  = `2,049`
- 합계(`score_mlp`) = `3,149,825`

슬롯별 projector(`K=6`)는:

- 한 개 헤드 = `d*pred_len + pred_len`
- 전체 = `K * (d*pred_len + pred_len)`
- `pred_len=96`일 때: `295,488`
- `pred_len=720`일 때: `2,216,160`

즉, **`score_mlp`는 pred_len과 무관하게 고정 비용**,  
**`slot_projectors`는 pred_len이 커질수록 선형 증가**한다.

---

## 4) 지금까지 시도한 완화 방법

## 4.1 `slot_fuse.hidden` 축소 (2048 -> 512)

실험군: `...hidden_sweep` (`h512`)

- 파라미터(ETTh1, pr96 기준):  
  `10,371,074 -> 8,008,706` (`-22.8%`)
- 정확도 경향(ETTh1/ETTh2, 4 horizons, 총 8케이스):
  - MSE win: `3/8`
  - MAE win: `2/8`

요약: **파라미터는 크게 줄지만 성능 저하 리스크가 큼**.

## 4.2 Low-rank factorization (`score_mode=lowrank`)

실험군: `...lowrank_sweep` (`rank=128/192/256/384`, `hidden=2048`)

- 총 비교: `2 datasets x 4 horizons x 4 ranks = 32`
- 전체 집계:
  - MSE win: `12/32`
  - MAE win: `14/32`
  - Both(MSE+MAE 동시 개선): `10/32`
- 효율:
  - 속도: 대략 `1.77x ~ 1.95x` 빠름
  - 파라미터: 대략 `14.03% ~ 25.91%` 감소

랭크별 전반 경향:

- `rank=192`: 중간 horizon(특히 192/336)에서 강한 편
- `rank=384`: 종합적으로 가장 안정적(전체 균형 최상)

요약: **정확도 희생을 최소화하면서 파라미터/시간을 줄이는 가장 유효한 접근**.

## 4.3 Horizon gate 초기화 스케줄 (linear/sigmix)

실험군: `...hgatebias_linear...`, `...hgatebias_sigmix...`

- 목적: long horizon correction 폭주 완화
- 결과: 일부 케이스에서 수치 변화는 있지만,  
  **파라미터 절감에는 직접 기여하지 않음** (구조는 동일)

요약: **안정화/학습 동역학 튜닝용**이며, 파라미터 이슈의 직접 해법은 아님.

---

## 5) 현재 결론

`residual_gated_output` 파라미터 병목은 구조적으로:

1. `score_mlp` 첫층(`3d -> hidden`)  
2. `slot_projectors`의 pred_len 의존 증가

에 의해 발생한다.

따라서 현재까지의 결과를 기준으로:

- 가장 현실적인 해법: `score_mode=lowrank` (특히 `rank=192` 또는 `384`)
- `hidden` 축소 단독 사용은 성능 손실 위험이 큼
- gate init 스케줄은 성능 안정화 보조 수단이지 파라미터 해결책은 아님

---

## 6) 실무용 권장안 (현재 시점)

1. 기본 후보: `lowrank rank=384`  
   (전체 조합에서 균형적)
2. 정확도 민감 구간 보완: `rank=192`와 병행 비교
3. long-horizon 안정성은 `hgate bias schedule`로 별도 튜닝

이 조합이 현재까지 실험 결과상, **성능/효율/안정성의 균형점**에 가장 가깝다.
