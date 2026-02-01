# VarMAE 마스킹 정책: mask_axis=var 고정

작성 목적: VarMAE에서 **변수 단위 마스킹(mask_axis=var)**만 사용하기로 한 결정과, 근거/수정 방향을 정리한다.

---

## 1) 설명
- 기존 VarMAE는 `mask_axis=var|token`을 지원한다.
- 그러나 `mask_axis=token`은 **마스킹과 손실 정의가 불일치**하여
  의도한 “변수 단위 마스킹”과 다른 결과를 만든다.
- 따라서 VarMAE에서는 **mask_axis=var만 사용**하는 것이 정합적이다.

---

## 2) 근거

### 2.1 token 마스킹 시 손실 불일치
- `mask_tokens`를 만든 뒤 `mask_var = mask_tokens.any(dim=1)`로 변수 단위로 축약
  - `src/itransformer/ssl/var_mae.py:242-245`
- 손실은 `mask_var` 기준으로 계산됨
  - `src/itransformer/ssl/var_mae.py:275-283`

---

## 3) 수정안(설계 방향)

### 3.1 정책 고정
- `mask_axis`를 `var`로 고정 (config에서도 기본값 유지)
- `token` 경로는 제거하거나 사용 금지

### 3.2 마스킹 구현
- `mask_var [B, N]`를 생성
- patch 경로에서는 해당 변수의 모든 patch 토큰을 동시에 마스크

---

## 4) 영향/검증 체크
- 동일 seed 조건에서 mask_axis=var로 결과가 안정적으로 재현되는지 확인
- P0/P1 vs P2/P3/P4 비교가 동일한 마스킹 의미를 가지는지 확인

---

## 5) 관련 파일
- `src/itransformer/ssl/var_mae.py`
- `conf/ssl/var_mae.yaml`

