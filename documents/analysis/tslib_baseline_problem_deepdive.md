# TSLib Baseline 저성능 이슈 분석 (Deep Dive)

작성일: 2026-02-10  
대상: `ETTh1`, `pred_len=96`, `TSLIB` 통합 실험 (`conf/plan/TSLIB_baseline_compare_main.yaml`)

---

## 1) 문제 요약

현재 베이스라인 저성능은 단일 원인이 아니라, 아래 3개가 겹친 영향으로 보는 것이 타당하다.

1. 디코더 시간마크(`x_mark_dec`) 입력 경로 불일치 (가장 치명적)
2. 시간 인코딩 설정 불일치 (`embed=timeF` vs `data.timeenc=0`)
3. 모델별 레시피 미반영(일괄 하이퍼파라미터)

---

## 2) 실제 관측된 성능 (ETTh1, pr96)

아래는 `artifacts/runs/SUP-ETTh1.TSLIB.*.e20.vt.pr96.sd0/downstream_metrics.json` 기준.

| model | test mse | test mae | params | sec/epoch |
|---|---:|---:|---:|---:|
| PatchTST | 0.3810 | 0.3967 | 6,903,904 | 12.89 |
| iTransformer | 0.3949 | 0.4094 | 6,404,704 | 4.60 |
| DLinear | 0.4716 | 0.4566 | 18,624 | 1.18 |
| TiDE | 0.6290 | 0.5314 | 2,534,141 | 15.96 |
| Autoformer | 0.8034 | 0.6651 | 10,535,943 | 32.74 |
| FEDformer | 1.0392 | 0.7683 | 16,827,399 | 104.31 |
| TimesNet | 진행 중 / 미집계 |

관찰 포인트:
- 디코더 시간정보를 많이 쓰는 계열(Autoformer/FEDformer/TiDE)에서 특히 성능 저하가 큼.
- FEDformer는 성능뿐 아니라 학습시간도 매우 큼(설정 영향 포함).

---

## 3) 원인 A: 디코더 시간마크 입력이 사실상 깨져 있음 (High)

### 증거

- 데이터셋은 `seq_y_mark`를 생성/반환함  
  `src/itransformer/data/datasets.py:82`  
  `src/itransformer/data/datasets.py:86`

- 그러나 학습/평가 루프에서 `batch_y_mark`를 버림 (`_`)  
  `src/itransformer/downstream.py:186`  
  `src/itransformer/downstream.py:530`

- 어댑터는 미래 구간 마크를 마지막 시점 반복으로 생성  
  `src/itransformer/models/tslib_adapter.py:179`

### 영향

- Autoformer/FEDformer/Transformer/TiDE는 `x_mark_dec`를 디코더 임베딩에 직접 사용.
- 실제 미래 시간정보 대신 "마지막 시점 복붙"이 들어가므로, horizon별 시간 패턴 학습이 크게 손상됨.

---

## 4) 원인 B: `timeF` 임베딩과 `timeenc` 설정 불일치 (High)

### 증거

- 현재 TSLIB 설정은 `embed: timeF`  
  `conf/model/TSLIB.yaml:30`

- ETTh1 데이터 설정은 `timeenc: 0`  
  `conf/data/ETTh1.yaml:13`

- TSLib 원본은 `embed == timeF`면 `timeenc = 1`을 사용  
  `external/Time-Series-Library/data_provider/data_factory.py:24`

### 영향

- `timeF` 경로는 연속형 time feature 전제를 갖는데, 현재는 정수 month/day/hour 입력이 사용됨.
- 시간특징 분포/스케일이 원본 레시피와 달라져 모델별 성능 저하 가능성이 큼.

---

## 5) 원인 C: 모델별 레시피 미반영 (Medium-High)

### 현재 플랜(일괄 적용)

- `train.epochs=10`, `batch_size=128`, `optim=adam`, `lr=1e-4`, `steplr(step=1,gamma=0.5)`  
  `conf/plan/TSLIB_baseline_compare_main.yaml:14`  
  `conf/plan/TSLIB_baseline_compare_main.yaml:25`

### 모델별 레시피 차이 예시 (TSLib scripts)

- TiDE ETTh1: `batch_size=512`, `learning_rate=0.1`, `d_model=256`, `d_ff=256`  
  `external/Time-Series-Library/scripts/long_term_forecast/ETT_script/TiDE_ETTh1.sh`

- TimesNet ETTh1: `d_model=16`, `d_ff=32`  
  `external/Time-Series-Library/scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh`

- PatchTST ETTh1: `e_layers=1`, horizon별 `n_heads` 별도  
  `external/Time-Series-Library/scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh`

### 영향

- 특정 모델(TiDE/TimesNet 등)은 현재 공통 설정에서 과소/과대학습 또는 구조 미스매치가 발생.
- 모델 간 상대 비교 자체가 왜곡될 수 있음.

---

## 6) 부가 관찰 (성능 외)

1. FEDformer가 유독 느린 것은 자연스러운 편.
   - 파라미터/연산량이 크고 현재 설정(`d_model=512`)이 무거움.

2. 로그가 초반에 거의 안 보이는 현상은 버퍼링 영향도 있음.
   - 실행 중인데 로그 파일 라인이 늦게 flush될 수 있음.

3. `runtime.deterministic=true` + CUDA 경고가 지속적으로 출력됨.
   - 재현성/속도 트레이드오프 관점에서 별도 정리 필요.

---

## 7) 수정 우선순위 (제안)

### P0: 입력 인터페이스 정합성 복구

- [x] `downstream.py`에서 `batch_y_mark`를 버리지 않도록 변경
- [x] `TSLibForecastAdapter.forward`에 `y_mark` 입력 경로 추가
- [x] `x_mark_dec`를 "복붙 생성" 대신 실제 `seq_y_mark` 기반으로 구성

완료 기준:
- `x_mark_dec`의 미래 구간이 실제 미래 시간마크와 1:1 대응됨

### P1: 시간 인코딩 정합성

- [x] `embed=timeF` 사용 시 `data.timeenc=1` 강제 또는 검증 에러 처리
- [x] 실험 config에 `embed/timeenc` 조합 체크 로깅 추가

완료 기준:
- `timeF` 실험에서 `timeenc=1`이 자동/명시적으로 보장됨

### P2: 모델별 레시피 분리

- [ ] 최소 1차 대상(예: iTransformer, DLinear, PatchTST, TimesNet, TiDE, Autoformer, FEDformer)별 override 프로파일 작성
- [ ] 공통 플랜 + 모델별 오버라이드 체계로 전환
- [ ] 재실험 후 delta 보고

완료 기준:
- "공통 설정 성능"과 "레시피 반영 성능"을 분리 비교 가능

---

## 8) 다음 논의 포인트

1. 먼저 P0/P1(입력/시간인코딩)만 고치고 공통 하이퍼파라미터로 재측정할지
2. 아니면 동시에 P2(모델별 레시피)까지 적용할지

권장 순서:
1) P0  
2) P1  
3) P2

이 순서가 원인 분리를 가장 명확하게 해준다.

---

## 9) P0 상세 패치 계획 (무영향/저위험 우선)

목표:
- TSLib 경로에서만 디코더 시간마크(`x_mark_dec`)를 실제 `seq_y_mark`로 전달한다.
- P0/M0/패치모델 기존 동작은 바이트 단위로 동일하게 유지한다.
- 데이터 split/스케일링/학습 레시피/옵티마이저 로직은 건드리지 않는다.

### 9.1 변경 범위 (코드 파일)

1. `src/itransformer/models/tslib_adapter.py`
- `forward`에 keyword-only 인자 추가: `y_mark_dec=None`
- `_build_decoder_inputs`에 `y_mark_dec` 입력 경로 추가
- `y_mark_dec`가 있으면 이를 우선 사용, 없으면 기존(마지막 마크 반복) fallback 유지

2. `src/itransformer/downstream.py`
- train/eval 루프에서 `batch_y_mark`를 버리지 않고 텐서로 전달
- `cfg.model.variant == "TSLIB"`일 때만 `y_mark_dec`를 adapter로 전달
- 그 외 모델은 기존 호출 시그니처를 유지

3. `src/itransformer/eval.py`
- `_predict` 시그니처에 `y_mark` 추가
- `_run_eval`에서 `batch_y_mark`를 전달
- 역시 `TSLIB` 분기에서만 `y_mark_dec` 사용

비범위(명시적으로 미변경):
- `src/itransformer/data/*` (Dataset/DataLoader)
- `src/itransformer/models/itransformer.py`, `src/itransformer/models/m0.py`, `src/itransformer/models/patch_transformer.py`
- 학습/평가 metric 계산식

### 9.2 인터페이스 설계 (호환성 핵심)

`TSLibForecastAdapter.forward` 목표 시그니처:
- 기존: `forward(x_enc, x_mark_enc=None, meta_emb=None)`
- 변경: `forward(x_enc, x_mark_enc=None, meta_emb=None, *, y_mark_dec=None)`

설계 이유:
- 기존 positional 호출(`model(x, x_mark, meta_emb)`)을 깨지 않음
- 신규 입력은 keyword-only라 오용 가능성 감소
- `TSLIB` 외 모델은 호출 경로 자체를 변경하지 않음

### 9.3 `x_mark_dec` 구성 규칙 (정합성)

1. `y_mark_dec`가 존재하면:
- 기대 shape: `[B, label_len + pred_len, T]` (dataset `seq_y_mark`)
- 길이가 충분하면 뒤에서 `label_len + pred_len`을 잘라 사용
- 길이가 부족하면 앞쪽 zero-pad 후 사용(크래시 방지)

2. `y_mark_dec`가 없으면:
- 기존 로직 유지(마지막 encoder mark 반복)
- 즉, 신규 패치 적용 후에도 과거 run/config는 동작 보장

### 9.4 안전장치

1. TSLIB 강제 분기:
- `cfg.model.variant == "TSLIB"`일 때만 `y_mark_dec`를 forward keyword로 넘김
- 그 외 variant는 기존 경로 유지

2. 타입/디바이스 정합:
- `batch_y_mark`는 `torch.float32`, model device로 변환
- `x_mark_enc`와 dtype/device 불일치 시 adapter 내부에서 정렬

3. 예외/로그:
- shape가 명백히 비정상일 때만 명시적 에러
- 정상 케이스에서는 로그 노이즈 추가하지 않음

### 9.5 검증 계획 (패치 직후)

1. 정적/기능 스모크:
- `P0` 1-step forward: 기존과 동일하게 통과
- `M0` 1-step forward: 기존과 동일하게 통과
- `TSLIB-Autoformer` 1-step forward: 통과 + 출력 shape 확인

2. 입력 정합 검증:
- 단일 배치에서 `x_mark_dec[:, -pred_len:, :]`가 실제 `batch_y_mark[:, -pred_len:, :]`와 동일한지 assert

3. 회귀 방지:
- 동일 seed에서 `P0/M0`의 train step loss가 패치 전후 동일(허용 오차 1e-7 수준)

### 9.6 롤백 계획

문제 발생 시 즉시 복귀:
1. `downstream.py`/`eval.py`의 `TSLIB` 분기 호출만 되돌리면 기존 동작으로 복원 가능
2. adapter의 `y_mark_dec` 인자는 optional이므로, 코드 잔존 시에도 비활성 상태로 운영 가능

### 9.7 완료 기준 (Definition of Done)

1. TSLIB 실험에서 디코더 미래 마크가 실제 데이터셋 `seq_y_mark`를 사용
2. P0/M0/패치 모델의 학습/평가 결과가 패치 전과 동일
3. 기존 실험 스크립트/플랜 파일 수정 없이 실행 가능

---

## 10) P1 상세 패치 계획 (시간 인코딩 정합성)

목표:
- TSLib 원본과 동일하게 `embed`와 `timeenc` 조합을 강제한다.
- 잘못된 조합은 조용히 흘려보내지 않고 즉시 감지한다.
- TSLIB 외 경로(P0/M0/Px/기존 데이터 파이프라인)는 동작 불변으로 유지한다.

핵심 기준(원본 TSLib):
- `embed == timeF` 이면 `timeenc = 1`
- `embed != timeF` 이면 `timeenc = 0`

근거:
- `external/Time-Series-Library/data_provider/data_factory.py`에서  
  `timeenc = 0 if args.embed != 'timeF' else 1` 로 고정

### 10.1 변경 범위 (최소/안전)

1. `src/itransformer/data/factory.py`
- `data_provider` 진입 시점에 TSLIB 전용 정합성 검증 함수 호출
- 필요한 경우(정책이 auto일 때만) 로컬 변수 `timeenc`만 조정해서 dataset 생성에 사용
- 기본 정책은 `strict`로 두어 무의식적 설정 변경을 방지

2. `src/itransformer/data/mix.py`
- mix 경로도 동일한 검증 함수 사용
- 현재 TSLIB에서 mix를 주로 쓰지는 않더라도, 경로 불일치를 사전 차단

3. `conf/model/TSLIB.yaml`
- 정책 필드 추가:
  - `timeenc_policy: strict` (기본)
  - 허용값: `strict | auto | off`

4. `conf/plan/TSLIB_baseline_compare_main.yaml`
- 명시적으로 `data.timeenc=1` 추가(현재 `embed=timeF` 기준)
- 실험 설정이 파일만 봐도 self-contained 하게 보이도록 개선

비범위:
- `src/itransformer/data/datasets.py` (feature 계산 구현 자체는 유지)
- P0/M0/Patch 모델 forward/학습로직
- optimizer/scheduler/metric

### 10.2 정책 설계 (무피해 관점)

`strict` (기본, 권장):
- 조합 불일치 시 즉시 `ValueError`
- 자동 수정하지 않음
- 재현성/추적성 최우선

`auto` (옵션):
- 불일치 시 TSLib 규칙대로 내부 `timeenc`를 보정하여 사용
- 경고 로그를 명확히 출력
- 빠른 실험용, 단 기본값으로는 쓰지 않음

`off` (예외용):
- 검증/보정 모두 비활성화
- 디버깅/레거시 호환 목적 외 사용 비권장

안전장치:
- 위 정책은 `cfg.model.variant == "TSLIB"`에서만 활성
- 그 외 variant는 기존 경로 그대로 통과

### 10.3 구현 상세 (어거지 방지)

1. “하나의 진실 소스” 함수화
- 예: `_resolve_tslib_timeenc(cfg, requested_timeenc) -> int`
- 책임:
  - embed 값 정규화(`timeF`/기타)
  - 기대 timeenc 계산
  - policy에 따라 error/보정/패스 수행
  - 진단 메시지 생성

2. 데이터로더 생성 직전 적용
- `factory.py`의 `data_set = data_cls(...)` 바로 앞에서 `timeenc` 결정
- `mix.py`의 `_build_loader`에서도 동일 함수 호출

3. 로그 전략
- 정상 조합: 최초 1회 간단히 출력(variant=TSLIB일 때)
  - 예: `embed=timeF, timeenc=1 (validated)`
- auto 보정 발생: 강한 경고 1회 출력
- strict 실패: 에러 메시지에 “현재값/기대값/해결 override”를 모두 포함

### 10.4 호환성/리스크 관리

리스크 1: 기존 TSLIB 플랜이 `timeenc=0`로 되어 있어 strict에서 즉시 실패
- 대응: baseline plan에 `data.timeenc=1` 명시 추가
- 효과: 실패를 “초기 구성오류 발견”으로 전환

리스크 2: non-TSLIB 경로 영향
- 대응: variant gate로 코드 경로 완전 분리
- 검증: P0/M0 단일 배치 forward 회귀 테스트 유지

리스크 3: mix dataset 경로 불일치
- 대응: 동일 검증 함수를 mix에도 적용해 일관성 확보

### 10.5 검증 계획

1. 단위 검증(정책별)
- strict + (`embed=timeF`, `timeenc=0`) => 실패 확인
- strict + (`embed=timeF`, `timeenc=1`) => 통과
- strict + (`embed=fixed`, `timeenc=1`) => 실패 확인
- auto + mismatch => 경고 + 보정 적용 확인

2. 스모크 검증
- TSLIB 1-batch train/eval forward 통과
- P0/M0 1-batch forward 불변

3. 실행 검증
- `TSLIB_baseline_compare_main` 최소 1개 모델(예: iTransformer) 1 epoch dry-run
- 로그에 조합 검증 메시지 및 실제 사용 timeenc 기록 확인

### 10.6 완료 기준 (Definition of Done)

1. TSLIB에서 `embed/timeenc` 불일치가 silent하게 지나가지 않음
2. 기본 설정(`strict`)에서 잘못된 구성은 즉시 에러로 차단
3. baseline plan은 명시적 `data.timeenc`로 재현 가능 상태
4. P0/M0/Patch 경로 회귀 없음
