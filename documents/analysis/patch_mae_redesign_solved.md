# PatchMAE 재설계 메모 (의도 정합성 기준)

작성 목적: PatchMAE가 **patch_len 벡터를 그대로 임베딩/복원**하도록 구조를 재정의하고, downstream PatchITransformer와 **가중치/구조 정합성**을 확보한다.

---

## 1) 현재 구현 요약 (문제 지점 포함)

### 1.1 입력 흐름 (현재)
- 입력: `x_enc [B, L, N]`
- 패치 분할: `patches [B, P, patch_len, N]`
- **시간축 평균**: `patch_mean = mean(patches, dim=2) [B, P, N]`
- 토큰화: `tokens [B, P*N, 1]`
- 임베딩: `value_proj(1 → d_model)`
- Encoder 통과
- 복원: `projector(d_model → 1)` → `recon [B, P, N]`
- 손실: **patch_mean** 기준 MSE/MAE

### 1.2 핵심 문제
1) **패치 내부 정보 소실**
   - patch_len 시계열이 평균 스칼라로 압축됨.
   - patch 내부 패턴/형태를 학습할 수 없음.

2) **downstream과 구조 불일치**
   - downstream PatchITransformer는 `patch_embed(patch_len → d)` 구조를 기대.
   - 현재 PatchMAE는 `value_proj(1 → d)`이므로 checkpoint 전이가 사실상 불가능.

3) **LP에서 patch_embed 동결**
   - SSL→downstream 로딩 시 `patch_embed`는 랜덤 상태로 남을 가능성이 높고,
   - LP 모드에서는 동결되어 성능 저하 및 P2/P3/P4 유사 현상 유발 가능.

---

## 2) 의도한 설계(목표 구조)

### 2.1 핵심 목표
- PatchMAE가 **patch_len 벡터 자체를 임베딩**하고
- **복원 헤드가 patch_len 벡터를 출력**하도록 변경
- downstream PatchITransformer와 **patch embedding 구조를 일치**시키기

### 2.2 목표 구조
- 임베딩: `patch_embed(patch_len → d_model)`
- 복원: `projector(d_model → patch_len)`
- 손실: **원래 patch 벡터 기준 MSE/MAE**

---

## 3) 구체 변경안

### 3.1 `__init__` 변경
- `value_proj(1 → d_model)` 제거
- `patch_embed(patch_len → d_model)` 추가
- `projector(d_model → patch_len)`로 변경
- patch_len은 `model.patch.patch_len` 우선, 없으면 `ssl.patch_len`

### 3.2 forward 변경
- **patch_mean 제거**
- `patches = x_enc.reshape(B, P, patch_len, N)`
- `patches = patches.permute(0,1,3,2)  # [B, P, N, patch_len]`
- `patch_emb = patch_embed(patches)     # [B, P, N, d_model]`
- 마스킹은 **patch×variable 단위** 추천 (`mask [B, P, N]`)
- `tokens = patch_emb.reshape(B, P*N, d_model)`
- `enc_out → projector → recon [B, P, N, patch_len]`

### 3.3 loss 변경
- target = 원본 patch 벡터 (`patches`)
- pred = `recon`
- 마스크된 위치만 loss 계산

---

## 4) 설계 관련 주의 사항

1) **patch_mode=mean_pool**
   - PatchMAE에서는 의미가 거의 없음 (패치 복원이 불가능)
   - PatchMAE에서 `mean_pool`은 금지하거나 attn_mask만 조정하는 방향 권장

2) **patch_len 정합성**
   - SSL과 downstream이 patch_len을 다르게 쓰면 구조 불일치 발생
   - patch_len은 downstream과 **완전 동일**하게 맞추는 것이 필수

3) **seq_len % patch_len != 0**
   - 현재 구조는 뒤쪽 truncate
   - 필요 시 pad 또는 drop 정책을 명시적으로 문서화할 것

---

## 5) 후속 확인 항목 (다음 단계)

- downstream LP에서 `patch_embed` 동결 여부 재검토
- SSL ckpt → downstream 로딩 시 `patch_embed`가 실제로 매칭되는지 확인
- P2/P3/P4 차이가 loss 곡선/성능에서 분리되는지 실험으로 검증

---

## 6) 파일 영향 범위

- 수정 대상: `src/itransformer/ssl/patch_mae.py`
- 연관 구조: `src/itransformer/models/patch_transformer.py`, `src/itransformer/downstream.py`

