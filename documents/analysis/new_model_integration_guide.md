# 새로운 베이스라인 모델 추가 가이드 (Pure Implementation)

## 0) 필수 원칙

- `external/Time-Series-Library` 및 `src/itransformer/models/tslib_adapter.py` 경로는 **사용하지 않는다**.
- `conf/model/TSLIB.yaml`, `conf/plan/TSLIB_*`는 과거 시도(실패/레거시)로 간주하고, 새 실험 파이프라인은 의존하지 않는다.
- 모든 베이스라인(PatchTST, Informer, DLinear, TiDE)은 이 코드베이스 내부(`src/itransformer/models`)에 직접 구현한다.

---

## 1) 최소 추가/수정 파일

새 모델 1개당 최소 아래 4곳이 필요하다.

1. `src/itransformer/models/<new_model>.py`  
   - 모델 클래스 구현
2. `src/itransformer/models/__init__.py`  
   - 클래스 export
3. `src/itransformer/models/factory.py`  
   - `cfg.model.variant` 분기 추가
4. `conf/model/<NewVariant>.yaml`  
   - Hydra 모델 설정 추가

실험까지 하려면 추가로:

5. `conf/plan/<your_plan>.yaml`  
   - `model=<NewVariant>`로 실행 플랜 작성

---

## 2) 모델 인터페이스 규약

새 모델 클래스는 아래 인터페이스를 맞춘다.

- `forward(self, x_enc, x_mark_enc=None, meta_emb=None)`
- 출력 shape: `[B, pred_len, N]`
- 가능하면 `forecast(...) -> (pred, attns)`도 제공  
  (`analysis_entry.py`의 일부 분석 코드 호환성 확보)

권장 사항:

- `use_norm` 처리 방식은 기존 `ITransformer`, `ITransformerM0`와 동일하게 유지
- `x_mark_enc`를 안 쓰는 모델도 인자 자체는 받되 내부에서 무시

---

## 3) 구현 절차 (체크리스트)

### Step A. 모델 구현

- `src/itransformer/models/<new_model>.py` 작성
- `cfg.data.seq_len`, `cfg.data.pred_len`, `cfg.model.d_model` 등 공통 필드 사용
- 마지막 출력은 반드시 `[B, pred_len, N]`

### Step B. 팩토리 연결

- `src/itransformer/models/__init__.py`에 import/export 추가
- `src/itransformer/models/factory.py`에 `if cfg.model.variant == "<NewVariant>"` 분기 추가

### Step C. 모델 설정 추가

- `conf/model/<NewVariant>.yaml` 생성
- 최소 필드:
  - `variant`
  - `d_model`, `n_heads`, `e_layers`, `d_ff`, `dropout`, `activation`, `use_norm`
  - `meta`, `patch` 블록(안 써도 기본 구조 유지 권장)

### Step D. 실행 플랜 작성

- `conf/plan/<plan>.yaml`에서 `model=<NewVariant>` 지정
- 우선 seed 1개 + 소규모 pred_len로 smoke test 후 sweep 확장

---

## 4) 학습/평가 호환 시 주의점

### (1) Downstream `ft/lp` 모드

- `src/itransformer/downstream.py`의 SSL ckpt remap/freeze 로직은 기존 계열(P/M) 가정이 있다.
- 새 베이스라인 초기 검증은 `train.mode=sl` 권장.
- 꼭 `ft/lp`를 쓰려면 아래를 모델 구조에 맞게 점검:
  - `_remap_ssl_state`
  - freeze 대상 모듈 탐지 로직

### (2) Analysis 집계

- `analysis_entry.py`의 기본 variant 집계는 P계열 중심이다.
- 새 variant를 기본 집계에 포함하려면 variants 필터/기본값을 조정하거나, plan에서 명시적으로 지정한다.

### (3) Eval

- `eval.py`는 기본적으로 `build_model(cfg)` + checkpoint 로딩 구조라, factory만 정상 연결되면 대체로 동작한다.
- patch 계열은 `patch_len` 등 필수 하이퍼파라미터 누락 여부를 반드시 점검한다.

---

## 5) 빠른 Smoke Test 절차

1. 모델 단독 forward shape 확인  
2. `downstream` 1 epoch 실행 (`train.mode=sl`)  
3. run 디렉토리에 `downstream_checkpoint.pt`, `downstream_metrics.json` 생성 확인  
4. `eval` BASE(T3) 1회 실행 확인  

---

## 6) 권장 도입 순서

복잡도 기준으로 아래 순서가 안전하다.

1. **DLinear** (가장 단순, 파이프라인 검증용)
2. **PatchTST** (patch 처리 검증)
3. **Informer**
4. **TiDE**

이 순서대로 붙이면, 파이프라인 문제와 모델 문제를 분리해서 디버깅하기 쉽다.

---

## 7) 요약

- 이 프로젝트에서 새 베이스라인은 **반드시 내부 직접 구현**으로 통합한다.
- TSLIB/external 의존 경로는 사용하지 않는다.
- 핵심은 `models 구현 + factory 연결 + conf/model + plan` 4단계다.
