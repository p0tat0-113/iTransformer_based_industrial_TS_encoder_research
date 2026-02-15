네. “프리셋 A/B × LR 3개 = 6-candidate”를 **대표 튜닝 세트(ETTh1/Weather × pred_len 96/720)**에서만 고르고, 그 뒤 전체 벤치마크(6 datasets × 4 horizons)로 고정 평가하는 프로토콜을 **네 코드베이스의 plan YAML**로 바로 돌릴 수 있게 **구체 파일 단위로** 정리해줄게.

아래는 **PatchTST / Informer / DLinear / TiDE** 각각에 대해:

* 대표 튜닝 세트용 plan (A/B 두 run + lr sweep 3개)
* 전체 벤치마크용 plan 템플릿 (A/B 중 최종 선택 1개 + lr 1개로 고정)

을 제공한다.

> 중요한 수정: 너희 repo의 `conf/train/downstream.yaml`는 `val_flag`만 있고 `save_best` 같은 키가 없어. 그래서 plan에서 `train.save_best=...` 같은 건 넣으면 안 된다. (방금 실제 파일 확인함)

---

## 0) 공통 스윕 정의 (대표 튜닝 세트)

대표 튜닝 세트는 아래처럼 고정:

* dataset: `ETTh1`, `Weather`
* pred_len: `96`, `720`
* seq_len: `96` (lookback 고정)
* seed: 0 (튜닝은 seed 1개로 예산 절약)
* epochs: 10
* batch_size: 128
* val_flag: val (기본이 val이라 굳이 안 써도 되지만, 안전하게 명시)

LR 후보 3개는 **모델별로 다르게** 잡는다(아래 plan에 포함).

---

## 1) PatchTST — 튜닝 plan (프리셋 A/B + OneCycle max_lr 3개)

### 프리셋 근거(소스 기반)

* **Preset A(작게)**: `PatchTST_supervised/scripts/PatchTST/etth1.sh`의 설정(예: `d_model=16, n_heads=4, d_ff=128, dropout=0.3, fc_dropout=0.3, e_layers=3, patch_len=16, stride=8`)
* **Preset B(크게)**: `PatchTST_supervised/scripts/PatchTST/weather.sh`의 설정(예: `d_model=128, n_heads=16, d_ff=256, dropout=0.2, fc_dropout=0.2, e_layers=3, patch_len=16, stride=8`)
* **스케줄러**: PatchTST 코드베이스는 `OneCycleLR` + `pct_start`를 사용하고(`pct_start` arg 존재), `lradj='TST'`일 때 scheduler의 lr을 그대로 쓰는 형태(너가 말한 해석이 맞는 방향). 그래서 여기선 **repo 파이프라인과 맞춰 `optim.scheduler=onecycle`**로 고정하고 **`onecycle_max_lr`만 튜닝**한다.

### 파일 생성

`conf/plan/TUNE_PatchTST_AB_maxlr3_rep4_e10.yaml`

```yaml
plan_id: TUNE_PatchTST_AB_maxlr3_rep4_e10

runs:
  # Preset A (ETTh1-script-like small)
  - id_template: "TUNE_PatchTST_A_{dataset}_pl{pred_len}_mlr{max_lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=PatchTST"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # PatchTST preset A
      - "model.patch_len=16"
      - "model.stride=8"
      - "model.padding_patch=end"
      - "model.d_model=16"
      - "model.n_heads=4"
      - "model.e_layers=3"
      - "model.d_ff=128"
      - "model.dropout=0.3"
      - "model.fc_dropout=0.3"
      - "model.head_dropout=0.0"

      # keep PatchTST's common options explicit (defaults in PatchTST repo)
      - "model.revin=1"
      - "model.affine=0"
      - "model.subtract_last=0"
      - "model.decomposition=0"
      - "model.kernel_size=25"
      - "model.individual=0"

      # optimizer / scheduler
      - "optim=adam"
      - "optim.scheduler=onecycle"
      - "optim.onecycle_max_lr={max_lr}"
      - "optim.onecycle_pct_start=0.4"
      - "optim.onecycle_total_steps=0"

  # Preset B (Weather-script-like larger)
  - id_template: "TUNE_PatchTST_B_{dataset}_pl{pred_len}_mlr{max_lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=PatchTST"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # PatchTST preset B
      - "model.patch_len=16"
      - "model.stride=8"
      - "model.padding_patch=end"
      - "model.d_model=128"
      - "model.n_heads=16"
      - "model.e_layers=3"
      - "model.d_ff=256"
      - "model.dropout=0.2"
      - "model.fc_dropout=0.2"
      - "model.head_dropout=0.0"

      - "model.revin=1"
      - "model.affine=0"
      - "model.subtract_last=0"
      - "model.decomposition=0"
      - "model.kernel_size=25"
      - "model.individual=0"

      - "optim=adam"
      - "optim.scheduler=onecycle"
      - "optim.onecycle_max_lr={max_lr}"
      - "optim.onecycle_pct_start=0.4"
      - "optim.onecycle_total_steps=0"

sweep:
  dataset: [ETTh1, Weather]
  pred_len: [96, 720]
  seed: [0]
  max_lr: ["1e-4", "3e-4", "1e-3"]
```

---

## 2) Informer — 튜닝 plan (프리셋 A/B + StepLR(type1) LR 3개)

Informer은 공식 repo 스크립트들이 **구조 파라미터(d_model 등)를 거의 고정**하고, 주로 seq_len/label_len/pred_len만 바꾸는 형태라서(ETTh1.sh, ETTm1.sh), “공식 값 2개”로 A/B를 만들기 어렵다. 그래서 여기선:

* **Preset B = Informer2020 기본값**(d_model=512, n_heads=8, d_ff=2048, e_layers=2, d_layers=1, factor=5, attn=prob, embed=timeF, distil=True, mix=True, dropout=0.05)
* **Preset A = B를 1/2 스케일로 다운스케일**(d_model=256, n_heads=4, d_ff=1024)
  → “공식 구조를 그대로 유지하면서 capacity만 줄인 후보”라는 점에서 리뷰 대응이 가장 깔끔하다.

스케줄러는 네 해석대로 **type1 = epoch마다 0.5배**라서, 너희 repo의 `steplr(step_size=1, gamma=0.5)`로 매핑한다.

### 파일 생성

`conf/plan/TUNE_Informer_AB_lr3_rep4_e10.yaml`

```yaml
plan_id: TUNE_Informer_AB_lr3_rep4_e10

runs:
  # Preset A (scaled-down)
  - id_template: "TUNE_Informer_A_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=Informer"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Informer preset A
      - "model.attn=prob"
      - "model.factor=5"
      - "model.d_model=256"
      - "model.n_heads=4"
      - "model.e_layers=2"
      - "model.d_layers=1"
      - "model.d_ff=1024"
      - "model.dropout=0.05"
      - "model.embed=timeF"
      - "model.activation=gelu"
      - "model.distil=true"
      - "model.mix=true"
      - "model.output_attention=false"

      # optimizer / scheduler (type1 ~= steplr step_size=1 gamma=0.5)
      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=steplr"
      - "optim.step_size=1"
      - "optim.gamma=0.5"

  # Preset B (official-default-like)
  - id_template: "TUNE_Informer_B_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=Informer"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Informer preset B
      - "model.attn=prob"
      - "model.factor=5"
      - "model.d_model=512"
      - "model.n_heads=8"
      - "model.e_layers=2"
      - "model.d_layers=1"
      - "model.d_ff=2048"
      - "model.dropout=0.05"
      - "model.embed=timeF"
      - "model.activation=gelu"
      - "model.distil=true"
      - "model.mix=true"
      - "model.output_attention=false"

      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=steplr"
      - "optim.step_size=1"
      - "optim.gamma=0.5"

sweep:
  dataset: [ETTh1, Weather]
  pred_len: [96, 720]
  seed: [0]
  lr: ["1e-4", "3e-4", "1e-3"]
```

---

## 3) DLinear — 튜닝 plan (프리셋 A/B + StepLR(type1) LR 3개)

DLinear은 구조가 단순해서 A/B를 capacity 축으로 잡는 게 가장 합리적이다:

* Preset A: `individual=false` (모든 변수 공유 head)
* Preset B: `individual=true` (변수별 head)

LR은 공식 스크립트들에서 `1e-4`부터 `1e-2`까지 넓게 튀어서(ETTh1은 5e-3도 등장), 대표 튜닝 세트(ETTh1/Weather)에서 둘 다 커버하도록 **{1e-4, 5e-4, 5e-3}**로 잡는다.

### 파일 생성

`conf/plan/TUNE_DLinear_AB_lr3_rep4_e10.yaml`

```yaml
plan_id: TUNE_DLinear_AB_lr3_rep4_e10

runs:
  - id_template: "TUNE_DLinear_A_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=DLinear"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Preset A
      - "model.individual=false"
      - "model.moving_avg=25"

      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=steplr"
      - "optim.step_size=1"
      - "optim.gamma=0.5"

  - id_template: "TUNE_DLinear_B_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=DLinear"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Preset B
      - "model.individual=true"
      - "model.moving_avg=25"

      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=steplr"
      - "optim.step_size=1"
      - "optim.gamma=0.5"

sweep:
  dataset: [ETTh1, Weather]
  pred_len: [96, 720]
  seed: [0]
  lr: ["1e-4", "5e-4", "5e-3"]
```

---

## 4) TiDE — 튜닝 plan (프리셋 A/B + Cosine LR 3개)

TiDE는 “공식 코드 repo 스크립트” 대신 **논문 Appendix의 튜닝 결과 테이블(Table 8)**가 사실상의 공식 설정값이다(너가 원한 “실제 설정값 근거”에 가장 잘 부합).

* Preset A = Table 8의 ETTh1 row (hiddenSize=256, encoder=2, decoder=2, temporalDecoderHidden=128, dropout=0.3, revIn=True, layerNorm=True 등)
* Preset B = Table 8의 Weather row (hiddenSize=512, encoder=1, decoder=1, temporalDecoderHidden=16, dropout=0.0, revIn=False, layerNorm=True 등)

LR은 테이블 값이 대략 `~3e-5`대라서 **{1e-5, 3e-5, 1e-4}**로 커버.

스케줄러는 너가 말한 “cosine decay”를 **repo의 `optim.scheduler=cosine`**로 매핑.

### 파일 생성

`conf/plan/TUNE_TiDE_AB_lr3_rep4_e10.yaml`

```yaml
plan_id: TUNE_TiDE_AB_lr3_rep4_e10

runs:
  - id_template: "TUNE_TiDE_A_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=TiDE"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Preset A (Table-8 ETTh1-like)
      - "model.hidden_size=256"
      - "model.encoder_layers=2"
      - "model.decoder_layers=2"
      - "model.decoder_output_dim=8"
      - "model.temporal_decoder_hidden=128"
      - "model.dropout=0.3"
      - "model.layer_norm=true"
      - "model.revin=true"

      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=cosine"
      - "optim.min_lr=0.0"
      - "optim.t_max=0"

  - id_template: "TUNE_TiDE_B_{dataset}_pl{pred_len}_lr{lr}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=TiDE"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # Preset B (Table-8 Weather-like)
      - "model.hidden_size=512"
      - "model.encoder_layers=1"
      - "model.decoder_layers=1"
      - "model.decoder_output_dim=8"
      - "model.temporal_decoder_hidden=16"
      - "model.dropout=0.0"
      - "model.layer_norm=true"
      - "model.revin=false"

      - "optim=adam"
      - "optim.lr={lr}"
      - "optim.scheduler=cosine"
      - "optim.min_lr=0.0"
      - "optim.t_max=0"

sweep:
  dataset: [ETTh1, Weather]
  pred_len: [96, 720]
  seed: [0]
  lr: ["1e-5", "3e-5", "1e-4"]
```

---

## 5) 전체 벤치마크 plan 템플릿 (각 모델 공통 형태)

튜닝 끝나면 각 모델별로 “(A or B) + 최적 lr”을 하나 고른 뒤, 아래 템플릿으로 고정 평가.

### 공통 스윕

* dataset: `[ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather]`
* pred_len: `[96, 192, 336, 720]`
* seed: `[0, 1, 2]` (최소 3 seeds 추천)
* seq_len=96 고정

아래는 **PatchTST** 예시 템플릿(다른 모델도 동일하게 복사해서 model/optim 부분만 바꿔 쓰면 됨).

`conf/plan/BENCH_PatchTST_FIXED.yaml`

```yaml
plan_id: BENCH_PatchTST_FIXED

runs:
  - id_template: "BENCH_PatchTST_FIXED_{dataset}_pl{pred_len}_sd{seed}"
    overrides:
      - "task=multivariate_forecasting"
      - "train=downstream"
      - "data={dataset}"
      - "model=PatchTST"

      - "data.seq_len=96"
      - "data.label_len=48"
      - "data.pred_len={pred_len}"

      - "train.epochs=10"
      - "train.batch_size=128"
      - "train.patience=0"
      - "train.val_flag=val"

      - "runtime.seed={seed}"

      # TODO: 여기에 튜닝에서 선택된 프리셋(A or B) 값을 그대로 박아넣기
      # e.g. (Preset B + max_lr=3e-4 가 best였다면)
      - "model.patch_len=16"
      - "model.stride=8"
      - "model.padding_patch=end"
      - "model.d_model=128"
      - "model.n_heads=16"
      - "model.e_layers=3"
      - "model.d_ff=256"
      - "model.dropout=0.2"
      - "model.fc_dropout=0.2"
      - "model.head_dropout=0.0"
      - "model.revin=1"
      - "model.affine=0"
      - "model.subtract_last=0"
      - "model.decomposition=0"
      - "model.kernel_size=25"
      - "model.individual=0"

      - "optim=adam"
      - "optim.scheduler=onecycle"
      - "optim.onecycle_max_lr=3e-4"   # TODO: best max_lr로 교체
      - "optim.onecycle_pct_start=0.4"
      - "optim.onecycle_total_steps=0"

sweep:
  dataset: [ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather]
  pred_len: [96, 192, 336, 720]
  seed: [0, 1, 2]
```

---

## 6) 튜닝 결과에서 “best 후보”를 고르는 규칙(리뷰어 방어용)

대표 튜닝 세트 4개 태스크 각각의 **val MSE**를 얻으면, best 선택은 아래 중 하나로 고정해:

* **(권장) 평균 MSE 최소**: 4개 태스크 val MSE를 단순 평균
* 또는 **평균 rank 최소**: 태스크별로 후보 6개를 rank 매기고 평균 rank 최소

이 선택 규칙은 본문/부록에 **정확히 한 문장**으로 써두면 된다.

---

원하는 다음 단계(바로 이어서 가능한 작업):

1. 위 plan을 실제로 돌렸을 때 산출되는 artifacts 구조 기준으로, **best 후보 자동 선택(aggregate)** 스크립트를 `analysis/` 쪽에 붙이는 것
2. (중요) Informer/TiDE가 “미래 time feature(batch_y_mark)”를 쓰게 하려면 training loop 인터페이스를 살짝 확장해야 하는데, 너희 repo는 현재 `model(batch_x, batch_x_mark)`만 호출하는 형태라면 성능이 불리해질 수 있음. 이 부분까지 포함해서 **downstream loop를 backward-compatible하게 확장하는 패치**도 바로 줄 수 있음.

Newly introduced terms: none
