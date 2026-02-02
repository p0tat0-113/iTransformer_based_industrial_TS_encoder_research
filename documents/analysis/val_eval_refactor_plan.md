# Val 평가 안정화 리팩토링 계획

작성 목적: **validation 평가를 고정**하고, **마스크 랜덤성으로 인한 진동**을 줄이며,  
재현 가능한 val loss 곡선을 확보한다. (요구사항: *val 고정 + drop_last=False + mask 평균(2~4회)*)

---

## 1) 문제 요약
1) **pretrain val loss가 흔들림**
   - SSL 모델의 mask가 매번 랜덤 → 동일 batch라도 loss 변동
2) **Mix val 데이터가 매 epoch 달라짐**
   - MixBatchLoader가 epoch마다 schedule을 재구성
3) **val loader가 drop_last=True**
   - 매 epoch마다 버려지는 샘플이 달라져 val set이 변동

---

## 2) 목표
- validation을 **고정된 데이터 구성 + 고정된 샘플 순서**로 평가
- pretrain의 val loss는 **여러 mask 평균**으로 안정화

---

## 3) 변경 설계

### 3.1 val loader 고정 (shuffle=False, drop_last=False)
**수정 대상**
- `src/itransformer/data/factory.py`
  - `flag == "val"`일 때 `shuffle_flag=False`, `drop_last=False`
- `src/itransformer/data/mix.py` (`_build_loader`)
  - `flag == "val"`일 때 `shuffle=False`, `drop_last=False`

**의도**
- val set이 epoch마다 달라지는 현상 방지

---

### 3.2 Mix val schedule 고정
**수정 대상**
- `src/itransformer/data/mix.py` (`MixBatchLoader`)

**설계**
- val 전용 deterministic 모드 추가 (예: `deterministic=True`)
- deterministic 모드에서는 **epoch에 따른 schedule 변화 제거**
  - 옵션 A: `self._epoch`를 증가시키지 않음
  - 옵션 B: schedule을 최초 1회만 생성하여 캐시

**의도**
-(Mix) validation이 **항상 동일한 dataset 순서**로 평가되도록 고정

---

### 3.3 pretrain val mask 평균
**수정 대상**
- `src/itransformer/pretrain.py` (`_eval_val_loss`)

**설계**
- `val_mask_ensemble` 회수만큼 **mask 반복 평가 후 평균**
- config 추가:
  - `ssl.val_mask_ensemble: 1` (기본)
  - 필요 시 `2~4`로 설정

**선택 사항 (재현성 강화)**
- val mask용 `torch.Generator`에 고정 seed 적용
  - 예: `base_seed = cfg.runtime.seed + 10000`
  - batch/rep index로 오프셋 부여

---

## 3.4 (보류) deterministic mask vs ensemble 평균
- **현재 구현 상태**에서는 `val_mask_ensemble`을 2~4 수준으로 두면
  mask 랜덤성이 평균화되어 val loss 안정화에 도움됨.
- 다만 **재현성을 완전히 보장하지는 못함** (run 간 마스크가 달라질 수 있음).
- 필요 시 **deterministic mask(고정 seed)** 방식을 추가 구현하는 방향을 추후 검토.

---

## 4) 설정 추가 계획
**추가 위치**
- `conf/ssl/var_mae.yaml`
- `conf/ssl/patch_mae.yaml`

**추가 키**
```
val_mask_ensemble: 1
```

---

## 5) 검증 방법
1) 같은 seed로 두 번 실행 시 **val_loss 곡선이 동일하게 재현되는지**
2) val loader 길이가 epoch마다 고정되는지
3) val mask ensemble 적용 시 **val_loss 진동 감소** 확인

---

## 6) 영향 범위
- Train loss 계산은 변경 없음
- Downstream 평가 방식에는 영향 없음 (mask 없음)

---

## 7) 관련 파일
- `src/itransformer/data/factory.py`
- `src/itransformer/data/mix.py`
- `src/itransformer/pretrain.py`
- `conf/ssl/var_mae.yaml`
- `conf/ssl/patch_mae.yaml`
