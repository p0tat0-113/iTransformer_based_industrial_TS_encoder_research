# Baselines Integration Fix Checklist

`documents/analysis/baselines_integration_plan.md` 재검토에서 확인된 수정 필요 항목 체크리스트입니다.

## 체크리스트

- [x] **Informer `prob` 어텐션 구현 정합성**
  - 현재 `informer.attn=prob`가 실질적으로 `FullAttention` 래퍼로 동작함.
  - 관련 코드: `src/itransformer/models/informer.py:122`, `src/itransformer/models/informer.py:131`, `src/itransformer/models/informer.py:256`
  - 완료 기준:
    - `prob` 경로가 실제 ProbSparse 동작(샘플링/Top-k query 선택) 수행
    - `full` 경로와 연산/속도 차이가 재현됨

- [x] **Informer denorm 채널 브로드캐스트 안전화 (`c_out != enc_in`)**
  - `use_norm=True`일 때 denorm이 `enc_in` 통계를 기준으로 브로드캐스트되어 출력 채널이 의도와 다르게 확장될 수 있음.
  - 관련 코드: `src/itransformer/models/informer.py:237`, `src/itransformer/models/informer.py:303`, `src/itransformer/models/informer.py:352`
  - 완료 기준:
    - `c_out != enc_in` 설정에서 출력 shape이 항상 `[B, pred_len, c_out]`
    - denorm 경로가 채널 수 불일치 시 명시적으로 처리(예: 제한/분기/검증 에러)

- [x] **DLinear 이동평균 커널 크기 제약 명확화**
  - 현재 패딩 로직이 사실상 홀수 커널 전제라 짝수 커널에서 길이 불일치 가능성 존재.
  - 관련 코드: `src/itransformer/models/dlinear.py:19`, `src/itransformer/models/dlinear.py:48`
  - 완료 기준:
    - 정책 중 하나를 명확히 적용:
      - (권장) `kernel_size` 홀수만 허용하고 짝수 입력 시 즉시 `ValueError`
      - 또는 짝수 커널도 길이 보존되도록 패딩 로직 수정
    - 단위 테스트/스모크 테스트로 홀수·짝수 케이스 동작 확인

## 재현 커맨드

- 문법/컴파일 체크
  - `python -m compileall src/itransformer/models/informer.py src/itransformer/models/dlinear.py`

- ETTh1 스모크 (forward+backward)
  - 아래 스니펫 실행:

```bash
PYTHONPATH=/workspace/src python - <<'PY'
import torch
from hydra import initialize, compose
from itransformer.data import data_provider
from itransformer.models.factory import build_model

for model_name in ["DLinear", "PatchTST", "Informer", "TiDE"]:
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model_name}", "data=ETTh1", "runtime.device=cpu", "train.num_workers=0"],
        )
    _, loader = data_provider(cfg, flag="train")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    x = torch.as_tensor(batch_x, dtype=torch.float32)
    x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32) if batch_x_mark is not None else None
    y_mark = torch.as_tensor(batch_y_mark, dtype=torch.float32) if batch_y_mark is not None else None
    model = build_model(cfg)
    if getattr(model, "needs_y_mark_dec", False):
        pred = model(x, x_mark, None, y_mark_dec=y_mark)
    else:
        pred = model(x, x_mark, None)
    true = torch.as_tensor(batch_y, dtype=torch.float32)[:, -cfg.data.pred_len :, :]
    loss = torch.mean((pred - true) ** 2)
    loss.backward()
    print(model_name, tuple(pred.shape), float(loss.detach()))
PY
```

- Informer `prob/full` 속도/성능 미니 벤치
  - 체크포인트 기반:
    - `PYTHONPATH=/workspace/src python scripts/benchmark_informer_attn_modes.py --run-id <RUN_ID> --split test --device cpu --num-workers 0 --max-batches 200 --warmup-batches 10`
  - 설정 기반(체크포인트 없이):
    - `PYTHONPATH=/workspace/src python scripts/benchmark_informer_attn_modes.py --data ETTh1 --split test --device cpu --num-workers 0 --max-batches 200 --warmup-batches 10`
