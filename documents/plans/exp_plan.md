# 실험 계획 표

## 반복 규칙 요약(전 실험 공통)

| 항목          | 규칙                                                                                                | 비고                                 |
| ----------- | ------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 반복 단위       | **모든 “학습이 포함된 실험 조건”은 seed 3개(0/1/2)로 반복**                                                        | 평균±표준편차 산출                         |
| 평가만 수행하는 항목 | Shuffle/Missing sweep/Robustness curve/CKA/Trade-off 집계 등 **추론·분석만**은 학습 반복 없이 **학습된 체크포인트마다 1회** | 단, seed별 체크포인트가 있으므로 결과는 seed별로 존재 |
| 고정 항목       | optimizer/lr/scheduler/epochs/batch size, 데이터셋 전처리/공식 split                                       | iTransformer baseline 설정 고정        |

---

## 실험 ID 규칙(최종안)

### 1) Run ID (훈련)

**형식**

```
{I}.{D}.{V}.{H}.sd{seed}
```

| 구성요소    | 표기  | 예시                                                                                          |
| ------- | --- | ------------------------------------------------------------------------------------------- |
| 식별코드    | I   | A-TR-1, ...                                                                                 |
| 데이터셋    | D   | ETT / Traffic / Weather / ECL …                                                             |
| 모델 변형   | V   | A0 / A1Add / A1Cat / A1Fus / A2Add / … / P0 / P1 / P2 / … / SSL1(Var-MAE) / SSL2(Patch-MAE) |
| 하이퍼파라미터 | H   | patch_len / local_win_size / mask_ratio / SL / FT / LP 등(해당 Run에 영향 주는 핵심만)                 |
| seed    | sd  | sd0 / sd1 / sd2                                                                             |

---

### 2) Op ID (평가/진단/분석 결과)

**형식**

```
{I}__OP={code}[.{op_hparams}]__ON={RunID}
```

| 구성요소 | 표기  | 예시          |
| ---- | --- | ----------- |
| 식별코드 | I   | A-EV-1, ... |

| OP code 범주 | 코드       | 의미                                                                      |
| ---------- | -------- | ----------------------------------------------------------------------- |
| A 시나리오 평가  | S1/S2/S3 | Noise shift / Sampling 변형 / Bias·scale drift                            |
| A 진단       | T1/T2/T3 | Shuffle / Missing-metadata sweep/Semantic content ablation(A1 vs A2 비교) |
| B 분석       | F1~F5    | 비용 / trade-off / ~~lookback~~ / CKA / attention map                     |
| C 강건성      | R1/R2    | missing rate curve / sensors subset curve                               |

---

### 3) CMP/AGG (비교 또는 집계)

- **CMP**: 2개 Run 비교(예: A-DIAG-3)
- **AGG**: 여러 Run 집계(예: trade-off 그래프)

**형식**

```
CMP.{D}.{I}__L={RunID_left}__R={RunID_right}
AGG.{D}.{I}__ON={I1,I2,I3,...}
```

| 구성요소 | 표기  | 예시                              |
| ---- | --- | ------------------------------- |
| 식별코드 | I   | A-TR-1, ...                     |
| 데이터셋 | D   | ETT / Traffic / Weather / ECL … |

---

# 실험군 A 반복 표(메타 임베딩)

## A-1. 학습이 필요한 조건(Training Runs)

| 식별코드   | 데이터셋(D)    | 모델 변형(V)                                   | 반복(seed) | 산출물        |
| ------ | ---------- | ------------------------------------------ | -------- | ---------- |
| A-TR-1 | 모든 선택 데이터셋 | **A0** (메타 미사용 iTransformer)               | 0/1/2    | 예측 MSE/MAE |
| A-TR-2 | 모든 선택 데이터셋 | **A1(Add)**                                | 0/1/2    | 예측 MSE/MAE |
| A-TR-3 | 모든 선택 데이터셋 | **A1(Concat)**                             | 0/1/2    | 예측 MSE/MAE |
| A-TR-4 | 모든 선택 데이터셋 | **A1(Fusion MLP)**                         | 0/1/2    | 예측 MSE/MAE |
| A-TR-5 | 모든 선택 데이터셋 | **A2(Constant/UNK baseline - Add)**        | 0/1/2    | 예측 MSE/MAE |
| A-TR-6 | 모든 선택 데이터셋 | **A2(Constant/UNK baseline - Concat)**     | 0/1/2    | 예측 MSE/MAE |
| A-TR-7 | 모든 선택 데이터셋 | **A2(Constant/UNK baseline - Fusion MLP)** | 0/1/2    | 예측 MSE/MAE |

---

## A-2. 시나리오 기반 평가(Scenario Evaluation Runs: 학습 없이 평가만)

> 아래 항목은 **A-TR-1~7에서 학습된 체크포인트** 각각에 대해 평가만 수행.

| 식별코드   | 적용 대상 체크포인트                  | 시나리오(Op)            | 시나리오 상세 파라미터(Op_hparams)    | 반복(seed) | 산출물            |
| ------ | ---------------------------- | ------------------- | --------------------------- | -------- | -------------- |
| A-EV-1 | A0, A1 모든 체크포인트, A2 모든 체크포인트 | S1 Noise shift      | SNR 단계(예: L1~Lk)            | 0/1/2    | MSE/MAE(시나리오별) |
| A-EV-2 | A0, A1 모든 체크포인트, A2 모든 체크포인트 | S2 Sampling 변형      | downsample/보간 태그(sr_tag) 조합 | 0/1/2    | MSE/MAE        |
| A-EV-3 | A0, A1 모든 체크포인트, A2 모든 체크포인트 | S3 Bias/scale drift | offset/scale 단계(예: L1~Lk)   | 0/1/2    | MSE/MAE        |

---

## A-3. “메타 활용” 진단 테스트(평가 시점, 학습 없음)

> 아래 항목은 **해당 적용 대상 체크포인트**에 대해 평가만 수행.

| 식별코드     | 적용 대상 체크포인트                | 조작 방식(Op)                                                                                        | 반복(seed) | 기대 관찰                          |
| -------- | -------------------------- | ------------------------------------------------------------------------------------------------ | -------- | ------------------------------ |
| A-DIAG-1 | A1 모든 체크포인트                | T1(Shuffle test: 센서-메타 매칭 랜덤 셔플)                                                                 | 0/1/2    | 성능 하락(메타 사용 증거)                |
| A-DIAG-2 | A1 모든 체크포인트                | T2(Missing-metadata sweep: 0%/50%/100% 가림)                                                       | 0/1/2    | 메타 결측 민감도 곡선                   |
| A-DIAG-3 | A1 모든 체크포인트 vs A2 모든 체크포인트 | T3(Semantic content ablation: 동일 결합 방식(Add/Concat/Fusion MLP)에서 A1(원래 메타 텍스트)과 A2(상수/UNK 메타) 비교) | 0/1/2    | A1이 A2보다 우수하면 “메타 의미 정보” 효과 근거 |

> A-DIAG-3 실행 단위(권장): 결합 방식별 1:1 비교(CMP)로 생성  
> 예: `CMP.D.A-DIAG-3__L=A-TR-2.D.A1Add..sd0__R=A-TR-5.D.A2Add..sd0`

---

# 실험군 B 반복 표(Patching)

## B-1. 학습이 필요한 조건(Training Runs: SSL pretrain)

| 식별코드   | 데이터셋(D)    |                                                모델 변형(V) | patch_len(H) | 반복(seed) | 산출물                        |
| ------ | ---------- | ------------------------------------------------------: | -----------: | -------- | -------------------------- |
| B-TR-1 | 모든 선택 데이터셋 |                                       **P0** (no patch) |            - | 0/1/2    | pretrain ckpt + loss curve |
| B-TR-2 | 모든 선택 데이터셋 |                         **P1** (patch→mean pooling→1토큰) |   8/16/32/64 | 0/1/2    | pretrain ckpt + loss curve |
| B-TR-3 | 모든 선택 데이터셋 | **P2** (all patch tokens, global all-to-all attentnoin) |   8/16/32/64 | 0/1/2    | pretrain ckpt + loss curve |
| B-TR-4 | 모든 선택 데이터셋 | **P3** (all patch tokens, same timestep only attention) |   8/16/32/64 | 0/1/2    | pretrain ckpt + loss curve |
| B-TR-5 | 모든 선택 데이터셋 |   **P4** (all patch tokens, local window(±w) attention) |   8/16/32/64 | 0/1/2    | pretrain ckpt + loss curve |

> 주의: B 실험에서 사용하는 SSL 하이퍼파라미터(Var-MAE 방식 사용, mask ratio, epochs 등)는 **고정**이며, 위 Run들이 해당 설정으로 **각 조건별 SSL 체크포인트를 생성**한다.

---

## B-2. 평가/분석(학습 없이 체크포인트 기반으로 산출)

> 아래 항목은 **B-TR에서 생성된 pretrain ckpt**에 대해 **Linear Projection layer(=LP)만 해서 학습/평가**하거나, 체크포인트를 입력으로 분석만 수행.

| 식별코드       | 적용 대상 체크포인트        | 설정/변수(Op)                            | 반복(seed)  | 산출물                     |
| ---------- | ------------------ | ------------------------------------ | --------- | ----------------------- |
| B-EV-1     | P0~P4 모든 체크포인트     | F1(비용: 학습 시간/메모리/파라미터 수)             | 0/1/2     | 비용 표                    |
| B-EV-2     | P0~P4 모든 체크포인트     | F2(patch_len별 성능-비용 trade-off)       | 0/1/2     | 그래프 1장                  |
| ~~B-EV-3~~ | ~~P0~P4 모든 체크포인트~~ | ~~F3(lookback 변화(seq_len 후보 집합))~~   | ~~0/1/2~~ | ~~lookback vs 성능 표/곡선~~ |
| B-EV-4     | P0~P4 모든 체크포인트     | F4(첫 블록 출력과 마지막 블록 출력의 CKA, CKA 유사도) | 0/1/2     | CKA 값, 유사도(조건별)         |
| ~~B-EV-5~~ | ~~P0~P4 모든 체크포인트~~ | ~~F5(첫 블록과 마지막 블록의 attention map)~~  | ~~0/1/2~~ | ~~attention map~~       |

---

# 실험군 C 반복 표(SSL)

## C-1. SSL pretrain 조건(Training Runs: pretrain)

> B-TR-3 /  B-TR-4 / B-TR-5 중 하나의 모델 구조를 선택해서 실험에 사용

| 식별코드   | 데이터셋(D)    | 모델 변경(V)            |     mask_ratio(H) | 반복(seed) | 산출물                              |
| ------ | ---------- | ------------------- | ----------------: | -------- | -------------------------------- |
| C-PR-1 | 모든 선택 데이터셋 | **SSL1(Var-MAE)**   | 0.25 / 0.5 / 0.75 | 0/1/2    | pretrain ckpt, epoch별 loss curve |
| C-PR-2 | 모든 선택 데이터셋 | **SSL2(Patch-MAE)** | 0.25 / 0.5 / 0.75 | 0/1/2    | 〃                                |


---

## C-2. 다운스트림 평가(Training Runs: downstream)

| 식별코드   | 입력 체크포인트                 | 평가 방식(H) | 반복(seed) | 산출물                 |
| ------ | ------------------------ | -------- | -------- | ------------------- |
| C-DS-1 | no-SSL baseline(랜덤 init) | SL       | 0/1/2    | forecasting MSE/MAE |
| C-DS-2 | SSL1(Var-MAE) 모든 체크포인트   | FT       | 0/1/2    | forecasting MSE/MAE |
| C-DS-3 | SSL2(Patch-MAE) 모든 체크포인트 | LP       | 0/1/2    | forecasting MSE/MAE |

---

## C-3. 강건성 평가(평가 시점, 학습 없음)

> 아래 항목은 **C-DS에서 학습된 체크포인트**(SL 및 SSL의 FT/LP 결과)에 대해 평가만 수행.

| 식별코드   | 적용 대상               | x축 조건(Op)                         | 반복(seed) | 산출물      |
| ------ | ------------------- | --------------------------------- | -------- | -------- |
| C-RB-1 | SL vs SSL(FT/LP 각각) | R1(센서 측정값 부분적으로 결측: 0/10/30/50/70%)  | 0/1/2    | curve 1장 |
| C-RB-2 | SL vs SSL(FT/LP 각각) | R2(센서 채널 결측: 100/75/50/25%) | 0/1/2    | curve 1장 |

---

## 전체 반복 구조 한눈에 보기(요약 매트릭스)

| 실험군    | 학습이 반복되는 축                                               | 평가/진단/분석이 반복되는 축                                                 | 최소 반복 단위                                     |
| ------ | -------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------- |
| A(메타)  | 모델(A0, A1(3종), A2(3종)) × seed                            | 시나리오(S1~S3) + 진단(T1,T2) + 비교(CMP:T3)                             | (모델×seed) 1회 학습 → 여러 평가/진단 + 결합방식별 1:1 비교    |
| B(패칭)  | SSL pretrain 조건(P0/P1/P2옵션) × patch_len × seed           | F1~F5(비용/트레이드오프/~~lookback~~/CKA/attention map) + LP 성능(MSE/MAE) | (조건×seed) 1회 SSL pretrain → LP 평가/분석         |
| C(SSL) | pretext(2) × mask_ratio(3) × seed + downstream(SL/FT/LP) | R1/R2 강건성 곡선                                                     | (pretrain×seed) + (downstream×seed) → 강건성 평가 |
