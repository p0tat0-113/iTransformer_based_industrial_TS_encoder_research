# PatchITransformer 구조 재설계 계획 (PatchMAE 정합화)

작성 목적: **downstream PatchITransformer**를 **PatchMAE와 동일한 patch 토큰 구조**로 정렬해
P2~P4의 구조 차이가 downstream에서도 반영되도록 설계안을 정리한다.

---

## 1) 설명 (현 상태 요약)
- PatchMAE는 **patch 토큰을 끝까지 유지**하고, `projector(d→patch_len)`로 **patch 단위 복원**을 수행한다.
- PatchITransformer는 encoder 뒤에 **patch 축 평균(`mean(dim=1)`)**으로 **patch 구조를 눌러버린다.**
- 따라서 P2/P3/P4의 patch 구조 차이가 downstream에서 약화될 가능성이 높다.

---

## 2) 근거 코드 라인

### 2.1 PatchMAE: patch 토큰 유지 + patch 복원
- `src/itransformer/ssl/patch_mae.py:141-147`
  - `enc_out`를 `[B, P, N, E]`로 유지하고 `projector(d→patch_len)`로 복원

### 2.2 PatchITransformer: patch 평균으로 collapse
- `src/itransformer/models/patch_transformer.py:95-104`
  - `enc_out`를 `[B, P, N, E]`로 만든 뒤 `enc_out.mean(dim=1)` 수행

---

## 3) 설계안 (구체적 변경 방향)

### 목표
1. PatchMAE와 **동일한 patch 토큰 경로**를 downstream에도 유지
2. **patch 축 평균 제거**, 구조 차이가 출력에 반영되도록 설계
3. 예측 head만 downstream 목적에 맞게 별도 설계

### C안 (확정): **Temporal Concat Head**
- patch 축 평균을 제거하고, **patch를 시간 순서대로 concat**하여 예측 헤드에 입력
- 개념:
  - 입력: `enc_out` `[B, P, N, E]`
  - 시간순 concat: `[B, N, P*E]`
  - head: `Linear(P*E → pred_len)` → `[B, pred_len, N]`
- 장점:
  - patch 구조 차이가 **직접적으로 head 입력에 반영**
  - PatchMAE처럼 patch 토큰을 끝까지 유지

### 추가 요구사항 (PatchMAE 정합화)

#### A-1) x_mark 처리 정합화
- PatchMAE처럼 **x_mark도 patch token으로 유지**한다.
- 평균 풀링으로 요약하지 않으며, encoder 이후 **time token만 제거**한다.
- 결과적으로 downstream에서도 **patch 단위 시간 정보가 유지**된다.

#### A-2) meta 처리 정합화
- PatchMAE와 동일하게 `_apply_meta`를 구현/사용한다.
- `meta_emb`를 patch token과 동일한 경로로 결합한다.
- `meta_mode`(add/concat/fusion)도 동일한 방식으로 적용한다.

### B안 (대안): Patch-to-Sequence Head
- patch 출력 그대로 복원(`projector_patch: d→patch_len`) 후 `[B, L, N]` 재구성
- 이후 별도 예측 head로 `pred_len` 매핑
- 단점: **목표가 “미래 예측”인 downstream과 직접 정합성이 낮음**

### (참고) C안은 위에서 **확정안**으로 채택됨

---

## 4) 수정안 (추천: C안 기준)

### 4.1 PatchITransformer 수정
1) `enc_out.mean(dim=1)` 제거 (**mean_pool 제외**)  
2) **Temporal concat head** 추가 (**mean_pool 제외**)  
3) concat 결과를 `Linear(P*E→pred_len)`에 전달 (**mean_pool 제외**)  
4) **mean_pool(P1) 경로는 “초기 patch 임베딩 후 patch 평균”만 수행**  
   - 이후 encoder 출력은 `[B, N, E]`로 유지하고 **추가 pooling은 하지 않음**  
5) **x_mark patch token 유지** 후 encoder 출력에서 time token 제거  
6) **meta 적용 경로 추가** (`_apply_meta`)  

### 4.2 구조 정합성 체크
- PatchMAE와 동일한 입력 경로 유지:
  - patch_embed (patch_len→d)
  - x_mark patch 토큰 포함 (pooling 전까지 유지)
  - patch_mode 기반 attn mask 적용
  - meta 적용 위치 동일

---

## 5) 검증/실험 체크
- P2/P3/P4 downstream 성능 차이가 더 분리되는지 확인
- 기존 mean_pool 방식 대비 loss/metric 비교
- attention pooling 가중치가 patch 모드별로 달라지는지 확인
- time token 포함/제거 전후 성능 비교 (x_mark 유지 효과 확인)
- meta on/off 비교 (meta 결합 경로 정합성 확인)

---

## 6) 구현 계획 (구체 단계)

### 6.1 PatchITransformer 구조 변경 (핵심)
1) **mean_pool 경로(P1)**  
   - “초기 patch 임베딩 → patch 평균”만 유지  
   - 이후 encoder 출력은 `[B, N, E]` 그대로 사용  
   - **추가 pooling 없음** (learned pooling 적용 금지)

2) **non-mean_pool 경로(P2~P4)**  
   - patch token을 끝까지 유지  
   - encoder 후 `[B, P, N, E]` → **시간순 concat `[B, N, P*E]`**  
   - 기존 `enc_out.mean(dim=1)` 제거

### 6.2 Temporal Concat Head 설계 (C안)
- `enc_out` `[B, P, N, E]` → `permute(0,2,1,3)` → `[B, N, P, E]`
- `reshape` → `[B, N, P*E]`
- `Linear(P*E→pred_len)` 적용 → `[B, pred_len, N]`
- **P2~P4에서만 적용**, P1은 제외

### 6.3 x_mark 정합화
- **non-mean_pool**: x_mark도 patch token으로 유지 → encoder 통과 → **time token 제거**
- **mean_pool**: 초기 patch 평균 후 time token도 평균하여 concat (기존 흐름 유지)

### 6.4 meta 정합화
- PatchMAE와 동일한 `_apply_meta` 로직 추가  
  - `meta_proj`, `meta_fuse` 구조 동일  
  - data token에만 적용 (time token은 제외)

### 6.5 구현 위치/수정 포인트
- `src/itransformer/models/patch_transformer.py`
  - `__init__`: meta 관련 모듈 추가, concat head용 projector 추가
  - `forward`: mean_pool 분기와 non-mean_pool 분기를 명확히 분리
  - `enc_out.mean(dim=1)` 제거, temporal concat head로 대체

---

## 7) 관련 파일
- `src/itransformer/models/patch_transformer.py`
- `src/itransformer/ssl/patch_mae.py`
