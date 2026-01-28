# Refactor Plan: iTransformer Experimental Platform (Hydra-first)

## 0. 목표 (Goals)
- iTransformer 계열만 남기고 코드베이스 단순화
- Hydra 기반으로 실험 설정/재현/추적이 일관되게 동작
- exp_plan.md의 반복 규칙/ID 규칙을 시스템적으로 자동화
- A/B/C 실험군을 배치 처리 가능한 오케스트레이터로 실행

## 1. 원칙 (Principles)
- **Hydra-first**: 설정은 모두 conf/에 선언, 코드에 하드코딩 금지
- **Single source of truth**: Run/Op/CMP/AGG ID는 utils에서 1회 정의
- **Minimal viable pipeline**: 먼저 최소 기능으로 end-to-end 동작, 이후 확장
- **Artifact-driven**: 모든 결과는 아티팩트 폴더에 저장하고 상태로 재시도 가능

## 2. 리팩토링 우선순위 (Milestones)

### M1. 구조 설계 + 스캐폴딩
- [x] 새 디렉토리 구조 생성 (`conf/`, `src/itransformer/`)
- [x] Hydra 기본 config 스키마 구축
- [x] train/eval/analysis 엔트리포인트 생성
- [x] Run/Op/CMP/AGG ID 생성 유틸 생성
  - 메모: `analysis.py`는 패키지 `analysis/`와 충돌을 피하기 위해 `analysis_entry.py`로 사용 중

### M2. iTransformer 핵심 이관
- [ ] iTransformer 기본 모델 이관
- [ ] A0/A1/A2 (메타 결합) 변형 구현
- [ ] 데이터 로더 이관 (전체 데이터셋 유지)

### M3. SSL 최소 기능 구현
- [ ] Var-MAE 최소 버전 (variate masking + MSE 복원)
- [ ] Patch-MAE 최소 버전 (patch token masking + MSE 복원)

### M4. 평가/진단/분석
- [ ] Scenario eval (S1/S2/S3)
- [ ] Diagnostics (T1/T2/T3)
- [ ] Robustness (R1/R2)
- [ ] 분석 (CKA, attention map, trade-off)

### M5. 오케스트레이터
- [ ] exp_plan.yaml 스펙 → Run/Op/CMP/AGG 생성
- [ ] DAG 기반 실행 (train → eval → analysis)
- [ ] resume/skip 지원

### M6. 정리
- [ ] legacy 코드 삭제
- [ ] 문서화 (Quickstart + 실험 예시)

## 3. 타깃 디렉토리 구조 (Target Layout)
```
/workspace
  conf/
    config.yaml
    data/
    model/
    ssl/
    train/
    optim/
    eval/
    analysis/
    runtime/
    paths/
    ids/
    experiment/
    plan/
  src/
    itransformer/
      train.py
      eval.py
      analysis_entry.py
      data/
      models/
      ssl/
      evals/
      analysis/
      utils/
      orchestrator/
  dataset/
  exp_plan.md
  refactor_plan.md
```

## 4. Hydra Config 스키마 요약
- `data`: dataset, path, seq_len, pred_len, enc_in, freq, features, timeenc
- `model`: variant(A0/A1/A2/P0~P4), d_model, n_heads, e_layers, meta/patch params
- `ssl`: enabled, type(var_mae/patch_mae), mask_ratio, pretrain_epochs
- `train`: epochs, batch_size, patience, seed
- `optim`: lr, scheduler, weight_decay
- `eval`: mode, op_code, op_hparams, checkpoint
- `analysis`: cka/attention/tradeoff flags
- `runtime`: device, precision, deterministic
- `paths`: artifacts root
- `ids`: run/op/cmp/agg id templates

## 5. Artifacts 규칙
- Run: `config.yaml`, `metrics.json`, `metrics.csv`, `checkpoint.pt`, `status.json`
- Op: `op_results.json`, `plots/`, `status.json`
- CMP/AGG: `cmp.json`/`agg.json`, `plots/`

## 6. 오케스트레이터 설계 연결
- exp_plan.yaml로 Run/Op/CMP/AGG 스펙 자동 생성
- 각 스펙은 Hydra entrypoint 호출
- 완료된 스펙은 status.json 기반 스킵

## 7. 위험/주의사항
- 기존 코드 삭제는 M2~M4 안정화 후에 수행
- SSL은 최소 기능부터 시작하고, 추가 목표는 별도 마일스톤으로 확장
- 데이터셋 유지로 인해 config가 방대해질 수 있음 → dataset별 default config 분리

## 8. 다음 작업 (Next Step)
- M2: iTransformer 기본 모델 + 데이터 로더 이관 시작
