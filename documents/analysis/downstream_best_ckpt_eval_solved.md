# Downstream 테스트 메트릭이 best ckpt 기준이 아님

작성 목적: downstream에서 **테스트 메트릭이 마지막 epoch 가중치로 계산**되는 문제를 정리하고, 수정 방향을 명확히 한다.

---

## 1) 현재 동작 요약
- 학습 루프에서 **val 기준 최고 성능**을 갱신할 때만 `downstream_checkpoint.pt`를 저장.
- 학습 종료 후 **test 메트릭을 즉시 계산**하지만, **best ckpt를 다시 로드하지 않음**.
- 결과적으로 test 메트릭은 **마지막 epoch 가중치**로 계산될 가능성이 높음.

---

## 2) 근거 코드 라인

### 2.1 best ckpt 저장
- `downstream_checkpoint.pt` 저장 위치
  - `src/itransformer/downstream.py:254-263`

### 2.2 test 메트릭 계산
- best ckpt 재로딩 없이 test 실행
  - `src/itransformer/downstream.py:273`

---

## 3) 문제점
- Early-stopping이 발생하거나 마지막 epoch가 best가 아닐 경우,
  **test 메트릭이 val 기준 best와 불일치**.
- 실험 리포트에서 **best-val 모델의 성능을 대표하지 못함**.
- 결과 해석/비교 시 신뢰성 하락.

---

## 4) 수정안(설계 방향)

### 4.1 가장 단순한 수정
- test 전 best ckpt를 다시 로드
  - 저장된 `downstream_checkpoint.pt`를 로드 후 `_evaluate()` 수행

### 4.2 대안
- best state_dict를 메모리에 보관 후 test 시 재사용
  - disk I/O 없이 즉시 적용 가능

### 4.3 부가 개선
- test 메트릭을 `best`와 `last` 두 개로 저장해 비교 가능하게 할 수도 있음

---

## 5) 영향/검증 체크
- early-stopping 활성화 실험에서 test 결과가 달라지는지 확인
- `best_epoch`와 test 결과가 일관되게 연결되는지 확인

---

## 6) 관련 파일
- `src/itransformer/downstream.py`

