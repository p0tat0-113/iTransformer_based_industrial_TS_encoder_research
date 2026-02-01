# SSL ckpt 로딩 시 shape 불일치로 일부 가중치 미로딩

작성 목적: downstream에서 SSL 체크포인트 로딩 시 **shape 불일치로 가중치가 누락**될 수 있는 문제를 정리한다.

---

## 1) 현재 동작 요약
- `_remap_ssl_state()`에서 키를 매핑한 뒤 **shape가 같을 때만** 로드한다.
- shape가 다르면 해당 파라미터는 **조용히 제외**된다.
- patch 관련 파라미터는 구조가 다르면 쉽게 불일치가 발생한다.

---

## 2) 근거 코드 라인

### 2.1 remap + shape 체크
- `if new_key in target_keys and target_state[new_key].shape == value.shape: remapped[new_key] = value`
  - `src/itransformer/downstream.py:53-54`

### 2.2 로딩 후 missing/unexpected 출력
- `missing, unexpected = model.load_state_dict(filtered, strict=False)`
  - `src/itransformer/downstream.py:61-62`

---

## 3) 문제점
- shape mismatch가 발생해도 **에러가 아니라 조용히 스킵**됨.
- patch 관련 파라미터(`patch_embed` 등)가 누락될 경우
  downstream이 **랜덤 초기화 상태로 시작**할 수 있음.
- 특히 LP 모드에서는 해당 모듈이 **동결되어 학습되지 않을 가능성**이 높음.

---

## 4) 수정안(설계 방향)

### 4.1 로딩 결과를 명확히 기록
- 누락된 키/shape mismatch를 로그에 자세히 남기기
- 중요한 파라미터 누락 시 경고 또는 에러 처리

### 4.2 구조 정합성 확보
- SSL pretrain 구조와 downstream 구조를 동일하게 맞추기
- patch 관련 모듈의 입력/출력 차원을 일치시키기

### 4.3 LP 동결 정책 재검토
- 만약 patch 관련 모듈이 미로딩이면 LP에서 동결하지 않도록 처리

---

## 5) 실제 변경 사항

### 5.1 projector/meta는 허용, patch_embed는 즉시 실패
- `projector*` 키는 사전훈련/다운스트림 헤드 불일치를 허용하기 위해 스킵 처리
  - `src/itransformer/downstream.py:66-69`
- `patch_embed.*` shape mismatch가 있으면 즉시 `ValueError`로 중단
  - `src/itransformer/downstream.py:106-116`
- `meta_*`는 실험군 A에서 의도적으로 결측시키는 경우가 있어 mismatch/미로딩을 허용
  - `src/itransformer/downstream.py:117-127`
- 위 허용 범위 밖의 shape mismatch / unmapped key는 전부 `ValueError`로 중단
  - `src/itransformer/downstream.py:120-138`

---

## 6) 영향/검증 체크
- SSL ckpt 로딩 시 patch 관련 mismatch가 있으면 즉시 실패하므로, 실험 전 구조 정합성 확인 필요
- meta를 의도적으로 빼는 실험만 예외로 허용되며, 그 외는 모두 강제 실패

---

## 7) 관련 파일
- `src/itransformer/downstream.py`
