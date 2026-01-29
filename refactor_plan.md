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
- [x] iTransformer 기본 모델 이관
- [x] A0/A1/A2 (메타 결합) 변형 구현
- [x] 센서 메타데이터 스키마/저장 포맷 확정 (dataset별)
- [x] 메타 텍스트 직렬화 + UNK 템플릿 규칙 정의
- [x] 메타 임베딩 파이프라인(외부 API + 캐시 + projection) 설계/구현
  - 메모: Gemini 클라이언트 구현 및 캐시 빌더/검증기 추가 완료
- [ ] 데이터셋별 메타데이터 매핑 규칙(수동 매핑) 적용
  - 메모: metadata.jsonl 템플릿 생성 완료, 실제 수동 매핑은 작성 필요
- [x] 데이터 로더 이관 (전체 데이터셋 유지)
- [x] 실험군 B: P0~P4 구조 구현 (P1~P4 포함)

### M3. SSL 최소 기능 구현
- [x] Var-MAE 최소 버전 (variate masking + MSE 복원)
- [x] Patch-MAE 최소 버전 (patch token masking + MSE 복원)
  - 메모: `itransformer.pretrain` 엔트리포인트 추가, ETTh1에서 1 epoch 테스트 성공
  - 메모: 경고 제거( pandas `apply` → `dt`, `loss.item` → `detach().item`, `utcnow` → `timezone.utc`)
- [x] C-DS: SL/FT/LP 학습 로직 구현
  - FT: SSL checkpoint 로드 후 전체 미세조정
  - LP: encoder+embedding freeze, pred_len projector만 학습
  - Patch-MAE downstream은 patch 기반 유지
  - 메모: `itransformer.downstream` 엔트리포인트 추가, SL/patch SL 테스트 성공

### M3.5. 산출물/집계 스키마 정비 (output_result_recator_plan 기반)
- [ ] **op/cmp/agg 분리 정책 확정 및 반영**
  - op_results.json: 단일 평가/진단 결과
  - cmp.json: 두 run 비교 결과
  - agg.json: 다수 run 집계 결과
- [ ] Run 메트릭 스키마 정비 (pretrain/downstream 공통)
  - 스키마 정의: summary/curves/cost/notes 구조 고정
  - 매 epoch validation loss 계산
  - best_epoch 저장 (summary)
  - early stopping 도입 (patience 기반)
  - grad_norm epoch 평균 기록
  - lr epoch 기록
  - cost 기록: wall_time, time/epoch, time/step, gpu_mem_peak, params_count
  - pretrain val_loss는 **masked_only** 기준 명시
  - 문서에 스키마 예시 추가
- [ ] **downstream 메타 전달 누락 해결**
  - metadata.enabled=true일 때 meta_emb를 로드/캐시 후 model에 전달
  - train/val/test 모두 동일 meta_emb 사용
- [ ] A-EV-1~3: op_results.json에 시나리오 하이퍼파라미터 기록
  - S1/S3: level
  - S2: downsample factor
  - op_results.json에 op_params 섹션 명시 저장
- [ ] A-DIAG-1: baseline+shuffle 동시 평가 기록
  - op_results.json: base_metrics / shuffled_metrics / delta
- [ ] A-DIAG-2: 메타 결측률 sweep을 단일 op로 실행
  - op_results.json: 결측률별 {mse, mae}
- [ ] A-DIAG-3: CMP 결과 정규화
  - cmp.json: left/right {mse, mae} + {delta_mse, delta_mae}
- [ ] CMP 스키마 고정
  - cmp.json에 left/right/delta만 저장, key 이름 통일
- [ ] AGG 스키마 고정
  - agg.json에 rows(원본) + agg(집계) 구조 사용
- [ ] B-EV-1: 비용 집계는 agg.json으로 저장
  - run 메트릭(cost) 집계 (seed별 raw + mean/std)
- [ ] B-EV-2: P1~P4 SSL ckpt → LP downstream 성능 집계 (P0 제외)
  - cost는 pretrain 기준(A안)
  - 성능 집계는 variant×patch_len 기준 (P1은 patch_len 영향 없음 주석)
- [ ] B-EV-4: CKA 집계 (P1~P4 같은 모델끼리)
  - agg.json: first_layer_cka / last_layer_cka / delta_cka
- [ ] B-EV-5: 실험 취소 (실행하지 않음) — 기록/정책만 유지
- [ ] C-RB-1/2: R1/R2 missing rate sweep을 단일 op로 실행
  - op_results.json: 결측률별 {mse, mae}
  - agg.json: 같은 모델(P0~P4)끼리 단순 집계

### M4. 평가/진단/분석
- [x] Scenario eval (S1/S2/S3)
  - S1: 입력에 Gaussian noise 추가 (레벨 sweep)
  - S2: 데이터 변형만 적용, sr_tag 미사용
  - S3: 입력에 scale + bias drift (레벨 sweep)
- [x] Diagnostics (T1/T2/T3)
  - T1: 센서↔메타 매칭 셔플 (A 실험군 전용)
  - T2: metadata missing sweep (A 실험군 전용)
  - T3: A1 vs A2 비교 (결합 방식별 CMP 생성)
- [x] Robustness (R1/R2)
  - R1: 센서 raw 측정값 부분 결측 (값 마스킹)
  - R2: 센서 채널 자체 결측 (채널 drop/mask)
- [x] C-RB: SSL downstream 모델 기준 R1/R2 curve 생성
- [x] 분석 (F1~F5)
  - F1: 비용(시간/메모리/파라미터) 집계
  - F2: 성능-비용 trade-off
  - F4: CKA (첫/마지막 블록)
  - F5: attention map
  - 메모: 기본 구현 완료, 스모크 테스트 미실행

### M5. 오케스트레이터
- [ ] exp_plan.yaml 스펙 → Run/Op/CMP/AGG 생성
- [ ] DAG 기반 실행 (train → eval → analysis)
- [ ] resume/skip 지원
- [ ] patch_len sweep 실행/로그 구조 구현 (B 실험군)

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
- 외부 API(메타 임베딩) 비용/속도 이슈 → 사전 임베딩 + 캐시 필수
- 메타데이터 스키마가 데이터셋마다 상이 → 템플릿/매핑 기반의 유연한 로딩 필요
- 실험군 B/C는 **메타데이터 미사용**, A에서만 사용

## 8. 다음 작업 (Next Step)
- M3.5: 산출물/집계 스키마 정비 + CMP/AGG 집계

## 9. 실험군 A 메타데이터 파이프라인(갱신)
- 저장 포맷: `dataset/<name>/metadata.jsonl` (수동 매핑 파일 기반)
- 메타데이터 스키마는 **dataset마다 상이**함 → 고정 필드 가정 금지
- 텍스트 직렬화는 **템플릿 기반**으로 처리 (dataset별 설정)
  - 예: `"{type}; {unit}; {quality}; {sr_tag}"` 혹은 `"{json}"` 등
  - 미존재 필드는 `UNK_<field>` 혹은 `UNK`로 치환
- 템플릿/매핑 설정 위치: `conf/metadata/<dataset>.yaml`
- `sensor_id`는 데이터 컬럼명과 **직접 매칭**
- 메타 임베딩 생성
  - 모델: `gemini-embedding-001` (3072-dim)
  - API key: 환경변수 사용
  - **사전 임베딩 + 캐시** 방식 (실험 중 API 호출 없음)
  - 캐시 포맷: `.pt` 또는 `.npy` + jsonl 인덱싱
- 현 상태 메모
  - metadata.jsonl 템플릿 파일 생성 완료 (각 dataset 폴더)
  - `conf/metadata/<dataset>.yaml` 기본 템플릿 추가
  - 메타 캐시 빌더 CLI 스켈레톤 추가 (`itransformer.tools.build_metadata_cache`)
  - 메타데이터 검증 CLI 추가 (`itransformer.tools.validate_metadata`)
  - Gemini 임베딩 클라이언트 구현 완료 + 테스트 성공 (ETTh1 캐시 생성)

## 10. 간과/미구현 항목 정리 (exp_plan 기준)
### 실험군 A (메타 임베딩)
- S1~S3 시나리오 평가 코드 구현 완료 (M4)
- T1~T3 진단 코드 구현 완료 (M4)
- CMP/AGG 집계 로직 미구현 → M3.5에 체크리스트 반영

### 실험군 B (패칭)
- P0~P4 구조 구현 미완 (P1~P4 필요) → M2 체크리스트에 반영
- patch_len sweep 실행/로그 구조 미구현 → M5 체크리스트에 반영
- F1/F2/F4/F5 분석 구현 완료 (M4)
- B/C 실험군은 메타데이터 미사용

### 실험군 C (SSL)
- C-DS: SL/FT/LP 학습 로직 미구현 → M3 체크리스트 반영
- C-RB: R1/R2 robustness curve 평가 구현 완료 (M4)
- Patch-MAE downstream은 patch 기반 유지
