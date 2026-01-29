# 10. Outputs & Metrics

## 1) 디렉터리 구조
- Run: `artifacts/runs/<run_id>/`
  - config.yaml
  - pretrain_metrics.json 또는 downstream_metrics.json
  - checkpoint (pretrain_checkpoint.pt / downstream_checkpoint.pt)
  - status.json

- Op: `artifacts/ops/<op_id>/`
  - op_results.json
  - config.yaml
  - status.json

- CMP: `artifacts/cmp/<cmp_id>/`
  - cmp.json
  - config.yaml
  - status.json

- AGG: `artifacts/agg/<agg_id>/`
  - agg.json
  - config.yaml
  - status.json

## 2) Run 메트릭 스키마
```
{
  "summary": {
    "best_epoch": 7,
    "best_train": {"loss": 0.12, "mse": 0.12, "mae": 0.21},
    "best_val": {"loss": 0.13},
    "early_stopped": true,
    "patience": 3,
    "stopped_epoch": 9
  },
  "curves": {
    "train_loss": [...],
    "val_loss": [...],
    "grad_norm": [...],
    "lr": [...]
  },
  "cost": {
    "wall_time_sec_total": 123.4,
    "time_sec_per_epoch_mean": 12.3,
    "time_sec_per_step_mean": 0.012,
    "gpu_mem_peak_mb": 1024.0,
    "params_count": 12345678
  },
  "notes": {
    "val_metric_basis": "masked_only"
  }
}
```

## 3) op_results.json
- S1/S2/S3: op_params 포함
- T1: base_metrics / shuffled_metrics / delta
- T2/R1/R2: metrics_by_rate
- meta 섹션: run_id/run_code/dataset/model_variant 기록

## 4) cmp.json
- left/right mse/mae + delta_mse/delta_mae
- meta: analysis_code/dataset

## 5) agg.json
- rows: 원본
- agg: mean/std 집계
- meta: analysis_code/dataset/run_code 필터
