# P2~P4에서 x_mark 사용 방식 (두 가지 테스트)

작성 목적: **P2/P3/P4** patch 모델에서 x_mark를 사용하는 두 가지 방식을 정의하고,
먼저 “내 방식”을 적용한 뒤 “대안 방식”을 적용해 비교하기로 한 합의 내용을 정리한다.

---

## 1) 설명
- P2/P3/P4는 patch 토큰을 사용하므로 x_mark도 patch 수준에서 다뤄야 함.
- 두 가지 설계를 모두 테스트한다:
  - **방식 A (내 방식)**: x_mark도 변수처럼 토큰화하여 attention에 함께 넣음
  - **방식 B (대안)**: x_mark를 patch 단위로 요약해 patch 토큰에 add/concat
- 테스트 순서: **A 먼저 → B 나중**
- VarMAE/PatchMAE 모두 동일한 방식 적용

---

## 2) 근거

### 2.1 P2~P4 patch token 구조
- patch_emb → `[B, P, N, d]` → `[B, P*N, d]`
  - `src/itransformer/models/patch_transformer.py:59-81`

### 2.2 P0의 x_mark 처리 방식(참조 기준)
- x_mark를 변수처럼 concat하여 token화
  - `src/itransformer/models/layers/embed.py:9-19`

---

## 3) 수정안(설계 방향)

### 3.1 방식 A (내 방식, 우선 적용)
- x_mark도 x_enc와 동일하게 patch 단위로 토큰화
- 최종 토큰: `[B, P*N, d]` (데이터 변수) + `[B, P*T, d]` (시간 변수)
- encoder에 **같이 입력하여 attention 수행**

### 3.2 방식 B (대안, 이후 적용)
- x_mark를 patch 단위로 요약하여 `[B, P, T]` 생성
- `time_proj(T → d)`로 변환 후 patch_emb에 add/concat
- positional embedding과 함께 적용

---

## 4) 영향/검증 체크
- A/B 방식에서 P2/P3/P4 성능 및 구조 차이 분리 여부 비교
- token 수 증가(A)로 인한 성능/비용 변화 확인

---

## 5) 관련 파일
- `src/itransformer/models/patch_transformer.py`
- `src/itransformer/ssl/var_mae.py`
- `src/itransformer/ssl/patch_mae.py`

