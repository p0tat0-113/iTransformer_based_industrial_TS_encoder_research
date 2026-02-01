# Pretrain/Downstream에서 seed 설정 부재

작성 목적: pretrain/downstream에서 **seed가 RNG에 적용되지 않는 문제**를 정리하고, 재현성 개선 방향을 정한다.

---

## 1) 현재 동작 요약
- `runtime.seed`는 **run_id 생성**에만 사용됨.
- `random/np.random/torch.manual_seed` 호출이 없음.
- 따라서 동일 seed로 실행해도 **학습 결과가 달라질 수 있음**.

---

## 2) 근거 코드 라인

### 2.1 seed가 run_id에만 사용됨
- Pretrain: `src/itransformer/pretrain.py:51-60`
- Downstream: `src/itransformer/downstream.py:118-127`

### 2.2 RNG seed 설정 호출 없음
- Pretrain/Downstream 전체에 `manual_seed`, `np.random.seed`, `random.seed`가 없음
  - `src/itransformer/pretrain.py` 전체
  - `src/itransformer/downstream.py` 전체

---

## 3) 문제점
- 실험 반복(seed 0/1/2)을 수행해도 **진짜 재현성이 보장되지 않음**.
- 다른 조건 비교 시 **seed variance가 통제되지 않아** 결과 해석이 불안정해짐.

---

## 4) 수정안(설계 방향)

### 4.1 기본 수정
- pretrain/downstream 시작 지점에 `_set_seed()` 추가
  - `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`

### 4.2 결정론 옵션 연계
- `runtime.deterministic=true`일 때
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True)` (가능 시)

### 4.3 DataLoader 재현성
- DataLoader shuffle이 재현되도록 generator/worker_init_fn 설정 고려
  - 특히 `num_workers > 0`일 때 필수

---

## 5) 영향/검증 체크
- 동일 seed로 2회 실행 시 loss curve/metrics 일치 여부 확인
- seed가 다른 실행과는 통계적으로 분리되는지 확인

---

## 6) 관련 파일
- `src/itransformer/pretrain.py`
- `src/itransformer/downstream.py`

