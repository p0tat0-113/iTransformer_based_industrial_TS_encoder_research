# Eval이 체크포인트 cfg가 아닌 현재 cfg로 모델 생성

작성 목적: eval이 **현재 cfg 기반으로 모델을 생성**하기 때문에, patch_len 등 핵심 설정이 누락되면 **잘못된 모델로 평가**되거나 실패할 수 있는 문제를 정리한다.

---

## 1) 현재 동작 요약
- eval은 **체크포인트 cfg를 읽지 않고** 현재 Hydra cfg로 모델을 구성한다.
- checkpoint는 `load_state()`로 가중치만 로드한다.
- patch 모델은 `patch_len`이 반드시 필요하지만, eval 실행 예시에 patch_len 지정이 없다.

---

## 2) 근거 코드 라인

### 2.1 eval이 현재 cfg로 모델 생성
- `model = build_model(cfg)`
  - `src/itransformer/eval.py:131`
- `load_state(model, ckpt_path)`
  - `src/itransformer/eval.py:132`

### 2.2 문서 예시에서 patch_len 누락
- C-RB 예시에서 `model=P2`만 지정하고 `model.patch.patch_len` 없음
  - `documents/08_experiment_C.md:41-46`

---

## 3) 문제점
- patch_len이 누락되면:
  - 모델 생성 시 오류가 발생하거나
  - 잘못된 patch_len/구조로 평가가 수행될 가능성 존재
- 특히 **patch 기반 모델(P1~P4)**은 구조 파라미터가 필수이므로
  eval 결과의 신뢰성이 떨어질 수 있음.

---

## 4) 수정안(설계 방향)

### 4.1 체크포인트 cfg 적용
- ckpt 내부 `cfg`를 읽어 **model 설정을 보정**
  - 최소: `model.patch.patch_len`, `model.patch.mode`, `model.patch.local_win` 등

### 4.2 eval 실행 시 강제 지정
- eval 명령에 `model.patch.patch_len`을 반드시 포함하도록 문서/템플릿 수정

### 4.3 유효성 검사
- eval 시작 시 patch 모델인데 patch_len이 0/None이면 명시적으로 에러

---

## 5) 영향/검증 체크
- C-RB eval 실행이 patch_len 없이도 실패하지 않는지 확인
- patch_len을 ckpt와 동일하게 맞추면 결과가 안정되는지 확인

---

## 6) 관련 파일
- `src/itransformer/eval.py`
- `documents/08_experiment_C.md`

