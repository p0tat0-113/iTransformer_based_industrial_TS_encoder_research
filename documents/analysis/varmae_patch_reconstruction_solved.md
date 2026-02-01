# VarMAE patch 경로: patch 단위 복원(B 방식)으로 전환

작성 목적: VarMAE에서 P2~P4를 사용할 때 **변수 단위 복원(A)** 대신 **patch 단위 복원(B)**으로 전환하기로 한 결정과 근거/수정 방향을 정리한다.

---

## 1) 설명
- 기존 VarMAE patch 경로는 encoder 출력 후 patch 축을 평균해 `[B, N, E]`로 축소하고,
  `projector(d_model→seq_len)`로 **변수 전체 시계열 복원**을 수행한다.
- 이는 “변수 단위 복원” 목적에는 맞지만, patch 구조 차이를 약화시킬 수 있다.
- 이번 합의는 **patch 단위 복원(B)**을 사용하여
  **patch 구조(P2/P3/P4)의 차이가 복원 과정에도 반영**되도록 하는 것이다.

---

## 2) 근거

### 2.1 현재 patch 축 평균 후 복원
- `enc_out.reshape(...).mean(dim=1)` → patch 축 평균
  - `src/itransformer/ssl/var_mae.py:267-268`
- `projector(d_model → seq_len)`로 전체 시계열 복원
  - `src/itransformer/ssl/var_mae.py:166, 269`

---

## 3) 수정안(설계 방향)

### 3.1 patch 단위 복원 구조
- encoder 출력에서 **patch 축 평균을 제거**
- `projector`는 **patch_len 단위 복원**으로 변경
  - 예: `projector(d_model → patch_len)`
- 복원 결과를 `[B, P, N, patch_len]` 형태로 유지
- 필요 시 patch를 다시 연결해 `[B, L, N]` 복원

### 3.2 손실 계산
- 마스크된 변수/패치 위치에 대해
  **patch 단위 복원 결과와 원본 patch를 직접 비교**

---

## 4) 영향/검증 체크
- P2/P3/P4 간 loss/성능 차이가 더 분리되는지 확인
- VarMAE-only 실험에서 구조 차이가 명확히 드러나는지 확인

---

## 5) 관련 파일
- `src/itransformer/ssl/var_mae.py`

