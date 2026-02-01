# P4(local) 구조 차이가 약화되는 설정(local_win=2, patch_len=16)

작성 목적: B_urgent 설정에서 P4(local)가 **사실상 global처럼 동작**하게 되는 이유와, 이를 완화하기 위한 수정안을 정리한다.

---

## 1) 설명
- B_urgent에서 `seq_len=96`, `patch_len=16`이면 **patch 개수 P=6**.
- `local_win=2`는 각 patch가 **자기 기준 ±2 patch**까지 보도록 허용한다.
- P가 6일 때 ±2는 대부분의 patch를 포함하므로, **local mask가 거의 제한을 주지 못함**.
- 결과적으로 P4(local)가 **P2(global)와 유사한 정보 범위**를 보게 되어
  **구조 차이가 실험 결과에서 약화**될 수 있다.

---

## 2) 근거

### 2.1 B_urgent 설정
- `patch_len=16`, `local_win=2`
  - `conf/plan/B_urgent.yaml:34-57`

### 2.2 Patch attention mask 정의
- local_win 기준으로 mask를 만들어 **±local_win 이내는 모두 허용**
  - `src/itransformer/models/patch_utils.py:1-16`

---

## 3) 수정안 (설계 방향)

### 3.1 patch_len을 줄여 P를 늘리기 (추천)
- 예: `patch_len=8` → P=12
- `local_win=1`이면 3/12(25%)만 관찰 → local 효과가 뚜렷해짐

### 3.2 local_win 더 줄이기
- P=6일 때는 `local_win=0`이나 `local_win=1` 정도로 제한해야
  global과 차이가 의미 있게 드러남

### 3.3 기준 비율로 설계
- **coverage = (2*local_win+1) / P**
- local 효과를 보려면 coverage를 **0.3 이하** 수준으로 유지하는 것이 안전

---

## 4) 실제 변경 사항
- `B_urgent`에서 patch_len을 8로 낮추고 local_win을 1로 축소
  - pretrain: `conf/plan/B_urgent.yaml:34-57`
  - downstream: `conf/plan/B_urgent.yaml:82-104`

---

## 5) 영향/검증 체크
- P2 vs P4의 loss/성능 차이가 실제로 분리되는지 확인
- patch_len sweep과 local_win sweep을 함께 돌려 차이를 관찰

---

## 6) 관련 파일
- `conf/plan/B_urgent.yaml`
- `src/itransformer/models/patch_utils.py`
