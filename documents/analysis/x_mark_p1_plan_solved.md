# P1(mean_pool)에서 x_mark 사용 방식

작성 목적: **P1(mean_pool)** 모델에서 x_mark를 사용하기로 한 합의 내용을 정리한다.

---

## 1) 설명
- P1은 patch를 mean pooling하여 **변수당 1개 토큰 [B, N, d]**으로 축소한다.
- 따라서 x_mark도 **patch 단위로 요약한 뒤**,
  최종적으로 **변수 토큰과 동일한 수준([B, T, d])**에서 결합하는 것이 정합적이다.
- 즉, **x_mark를 패치 단위로 요약 → 임베딩 → patch 축 평균 → 변수 토큰과 concat** 방식으로 사용한다.

---

## 2) 근거

### 2.1 P1(mean_pool) 구조
- patch_emb 평균 → `[B, N, E]` 토큰
  - `src/itransformer/models/patch_transformer.py:66-81`

### 2.2 P0의 x_mark 처리 방식
- x_mark를 변수처럼 concat하여 token화
  - `src/itransformer/models/layers/embed.py:9-19`

---

## 3) 수정안(설계 방향)

### 3.1 x_mark 처리
1. `x_mark [B, L, T]`를 patch 단위로 분할
2. `mean/last pooling`으로 patch 요약 → `[B, P, T]`
3. `time_proj(T → d)` 적용 → `[B, P, d]`
4. patch 축 평균 → `[B, d]` 혹은 `[B, T, d]` 형태로 변환
5. 최종적으로 **데이터 토큰 [B, N, d]과 concat**

### 3.2 결합 방식
- **concat 방식** 유지 (P0와 의미적으로 일치)

---

## 4) 영향/검증 체크
- P1(mean_pool)에서 x_mark가 없던 상태와 비교해 성능 변화 확인
- P0와의 공정 비교가 개선되는지 확인

---

## 5) 관련 파일
- `src/itransformer/models/patch_transformer.py`
- `src/itransformer/models/layers/embed.py`

