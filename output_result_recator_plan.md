# Output Result Refactor Plan (Metrics)

이 문서는 **산출물 개선에 대해 합의한 내용**을 기록한다.
실제 구현은 나중에 일괄 적용한다.

## 공통 정책
- 결과는 **op/cmp/agg로 분리**한다. (CMP/AGG 구조에 맞춤)
  - op_results.json: 단일 평가/진단 결과
  - cmp.json: 두 run 비교 결과
  - agg.json: 다수 run 집계 결과

## A. Run 변경사항 (훈련 메트릭)
### A-1. 공통 원칙
- pretrain/downstream 모두 동일한 메트릭 스키마를 지향한다.
- **best epoch 정보는 반드시 저장**한다.
- per-epoch 기준의 학습/검증 곡선을 저장한다.
- 메트릭은 JSON으로 저장한다.

### A-2. 검증 로직
- 매 epoch마다 **validation loss**를 계산한다.
- pretrain과 downstream 모두에 적용한다.

### A-3. pretrain 검증 메트릭 정의
- **검증은 loss만 기록**한다. (val_mse/val_mae 기록하지 않음)
- loss는 **마스크된 부분 기준**으로 계산한다.
- 메트릭 파일에 이 기준을 **명시**한다.
  - 예: `notes.val_metric_basis = "masked_only"`

### A-4. early stopping
- `conf/train/*.yaml`의 `patience`를 사용한다.
- early stopping 로직을 구현한다.
- 아래 항목을 summary에 기록한다.
  - `best_epoch`
  - `early_stopped` (true/false)
  - `patience`
  - `stopped_epoch` (조기 종료 시)

### A-5. 학습 곡선 (per-epoch)
- train_loss
- val_loss
- grad_norm (epoch 평균)
- lr (epoch 단위 기록)

### A-6. 요약 메트릭 (summary)
- `best_epoch`
- `best_train`: {loss, mse, mae}
- `best_val`: {loss}
- `early_stopped`, `patience`, `stopped_epoch`

### A-7. 비용/자원 (cost)
- `wall_time_sec_total`
- `time_sec_per_epoch_mean`
- `time_sec_per_step_mean`
- `gpu_mem_peak_mb` (CUDA일 때만)
- `params_count`

### A-8. 파일 구조 예시
```json
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
    "train_loss": [ ... ],
    "val_loss": [ ... ],
    "grad_norm": [ ... ],
    "lr": [ ... ]
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

### A-9. 적용 대상
- pretrain 메트릭 파일: `pretrain_metrics.json`
- downstream 메트릭 파일: `downstream_metrics.json`

### A-10. 구현 시 유의사항
- pretrain의 검증 loss는 **마스크된 영역 기준**임을 명시한다.
- best epoch 저장은 필수다.
- grad_norm은 epoch 평균으로 저장한다.
- lr은 epoch 기준으로 저장한다.

## B. Op 변경사항
| Op 코드 | 변경 내용 | 영향 범위 | 비고 |
| ------ | -------- | -------- | ---- |
| A-EV-1~3 | op_results.json에 시나리오 하이퍼파라미터 기록 (S1 level / S2 downsample factor / S3 level) | eval | op_hparams_tag 파싱 결과를 명시 저장 |
| A-DIAG-1 | baseline+shuffle 동시 평가 결과를 1회 기록 | eval | op_results.json에 base_metrics/shuffled_metrics/delta 저장 |
| A-DIAG-2 | 메타 결측률 sweep을 **단일 op**로 실행 | eval | op_results.json에 결측률별 {mse, mae} 기록 |
| A-DIAG-3 | CMP 결과를 좌/우 {mse, mae} + {delta_mse, delta_mae}로 정리 | analysis | cmp.json에 left/right {mse, mae}와 {delta_mse, delta_mae} 기록 |
| B-EV-1 | 비용(학습 시간/메모리/파라미터) 집계는 **agg.json**으로 저장 | analysis | run 메트릭(cost) 집계 결과 |
| B-EV-2 | P1~P4 SSL ckpt로 LP downstream 학습 후 성능 집계, **P0 제외** | analysis | cost는 pretrain 기준(A안), 성능은 LP 결과 집계(variant×patch_len 기준). P1은 patch_len 영향 없음 |
| B-EV-4 | CKA는 **같은 모델끼리(P1~P4)** 집계 | analysis | agg.json에 first_layer_cka, last_layer_cka, delta_cka 저장 |
| B-EV-5 | 실험 취소 | - | 실행하지 않음 |
| C-RB-1/2 | R1/R2 missing rate sweep을 **단일 op**로 실행 | eval | op_results.json에 결측률별 {mse, mae} 기록 |
| C-RB-1/2 (집계) | 집계는 **같은 모델(P0~P4)끼리** 단순 집계 | analysis | agg.json에 rate별 mean/std 저장 |

## C. 실험군 A 변경사항
| 실험 코드 | 변경 내용 | 산출물 영향 | 비고 |
| -------- | -------- | -------- | ---- |
| A-TR-* | (예: best epoch 기록 추가) | (metrics 스키마 변경) | |

## D. 실험군 B 변경사항
| 실험 코드 | 변경 내용 | 산출물 영향 | 비고 |
| -------- | -------- | -------- | ---- |
| B-TR-* | | | |

## E. 실험군 C 변경사항
| 실험 코드 | 변경 내용 | 산출물 영향 | 비고 |
| -------- | -------- | -------- | ---- |
| C-PR-* | | | |
| C-DS-* | | | |
| C-RB-* | | | |
