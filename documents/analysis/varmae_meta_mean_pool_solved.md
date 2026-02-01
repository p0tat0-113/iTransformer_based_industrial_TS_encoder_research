# VarMAE mean_pool에서 meta 결합은 N 기준으로 처리

작성 목적: VarMAE에서 **patch_mode=mean_pool**일 때 메타데이터 결합은
**변수 개수 N 기준으로만** 수행해야 한다는 결정과 근거/수정 방향을 정리한다.

---

## 1) 설명
- mean_pool 모드는 patch 임베딩을 **patch 축 평균**하여 `[B, N, E]`로 축소한다.
- 따라서 meta 임베딩도 **N 기준**으로 결합해야 shape가 맞다.
- 현재 로직은 patch 경로라는 이유로 **P×N 확장**을 수행해
  mean_pool에서는 구조적으로 불일치가 발생할 수 있다.

---

## 2) 근거

### 2.1 mean_pool 시 tokens shape
- `tokens = patch_emb.mean(dim=1)  # [B, N, E]`
  - `src/itransformer/ssl/var_mae.py:250-252`

### 2.2 meta 확장 로직
- patch 경로에서는 meta를 P×N으로 확장
  - `src/itransformer/ssl/var_mae.py:174-181`

---

## 3) 수정안(설계 방향)

### 3.1 mean_pool 전용 분기
- `patch_mode == "mean_pool"`일 때는
  meta를 **N 기준으로만** 맞추어 결합

### 3.2 patch 모드 분기 유지
- `all / same_time / local`에서는
  기존처럼 P×N 확장이 정상

---

## 4) 영향/검증 체크
- mean_pool + meta 활성화 조합에서 shape 오류가 발생하지 않는지 확인
- P1(mean_pool) 실험이 meta 설정과 일관되게 동작하는지 확인

---

## 5) 관련 파일
- `src/itransformer/ssl/var_mae.py`
- `conf/model/P1.yaml`

