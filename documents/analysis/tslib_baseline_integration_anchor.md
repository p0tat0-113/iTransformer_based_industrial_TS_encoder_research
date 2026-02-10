# TSLib Baseline Integration Anchor (M0 Fair Comparison)

목표: `M0`와 공정 비교를 위해 **데이터 공급/학습/평가 파이프라인은 현재 코드베이스를 유지**하고, baseline 모델 구현만 `Time-Series-Library`(TSLib)에서 가져와 통합한다.

앵커 원칙:
- 단일 진실원천(SSOT): 이 문서를 구현 체크리스트/진행 로그로 사용
- 파이프라인 공정성: `itransformer.data`, `downstream.py`, `eval.py`, metrics 포맷 고정
- 모델 모듈성: baseline 추가/제거가 `model=` 오버라이드만으로 가능해야 함

---

## Phase 0. Anchor & Scope

- [x] 앵커 문서 생성
- [x] TSLib 소스 확보 (`external/Time-Series-Library`)
- [ ] baseline 1차 대상 모델 확정 (권장: `iTransformer`, `DLinear`, `PatchTST`, `TimesNet`, `TimeMixer`, `Transformer`)
- [ ] 공정 비교 규약 확정 (val split, scheduler, epochs, seed, pred_len sweep)

완료 기준:
- 이 문서 기준으로 단계/상태를 추적 가능
- 소스/경로가 고정되어 재현 가능한 상태

---

## Phase 1. Integration Skeleton (Factory/Wrapper/Config)

- [x] TSLib 모델 브리지(wrapper) 추가
  - 파일: `src/itransformer/models/tslib_adapter.py`
  - 입력 계약 통일: `forward(x_enc, x_mark_enc, meta_emb=None) -> [B, pred_len, N]`
  - 내부에서 TSLib `Model` 호출 형식(`x_dec`, `x_mark_dec`) 흡수
- [x] 모델 팩토리 분기 추가 (`variant=TSLIB`)
  - 파일: `src/itransformer/models/factory.py`
- [x] 모델 export 등록
  - 파일: `src/itransformer/models/__init__.py`
- [x] Hydra 모델 설정 추가
  - 파일: `conf/model/TSLIB.yaml`
  - 핵심: `tslib.model_name`, `tslib.repo_root`, 모델별 기본 하이퍼파라미터
- [x] downstream freeze-rule 호환성 패치
  - 파일: `src/itransformer/downstream.py`
  - 목적: TSLib adapter처럼 `enc_embedding/patch_embed/value_proj`가 없는 모델에서도 LP/FT 분기 크래시 없이 진행

완료 기준:
- `python -m itransformer.downstream model=TSLIB ...`로 인스턴스 생성 가능

---

## Phase 2. Baseline Coverage & Compatibility

- [x] 1차 대상 모델별 호환성 점검
  - `iTransformer`, `DLinear`, `PatchTST`, `TimesNet`, `TimeMixer`, `Transformer`
- [x] 모델별 필요 config 키 누락 방지 (wrapper 기본값 제공)
- [x] 디코더형 모델(Transformer류) `x_dec/x_mark_dec` 생성 정책 고정
- [x] 실패 모델 graceful fallback (명확한 에러 메시지)

완료 기준:
- 1차 대상 모델 모두 1-step forward 통과

---

## Phase 3. Experiment Plan & Smoke Runs

- [x] baseline 스모크 plan 추가
  - 파일: `conf/plan/TSLIB_baseline_compare_smoke.yaml`
- [x] baseline 본실험 plan 추가
  - 파일: `conf/plan/TSLIB_baseline_compare_main.yaml`
- [x] 스모크 실행 (dataset 1개, pred_len 1~2개, epoch=1)
- [x] 산출물 검증 (`config.yaml`, `downstream_metrics.json`, `status.json`)
- [x] 실패 케이스 수정 후 재스모크

완료 기준:
- orchestrator로 baseline sweep이 최소 1개 dataset에서 완료

---

## Phase 4. Fair Benchmark Execution

- [x] M0와 동일 조건 run matrix 확정 (v1)
  - `dataset=[AppliancesEnergy]`, `pred_len=[96,192,336]`, `seed=[0]`
  - `model_name=[iTransformer, DLinear, PatchTST, TimesNet, TimeMixer, Transformer]`
  - 학습 규약: `e20`, `stop_epoch=10`, `val_flag=test`, `adam+onecycle`
- [ ] baseline 전체 sweep 실행
- [ ] 결과 집계 표 생성 (MSE/MAE + cost)
- [ ] M0 대비 delta 리포트 작성

완료 기준:
- 모델별/예측길이별 비교표 + 실행 로그/설정 재현 가능

---

## Phase 5. Quality Gates

- [ ] 재현성 점검 (동일 seed 재실행 편차 체크)
- [ ] 데이터 파이프라인 동일성 점검 (split/scale/time feature)
- [ ] 비용 지표 수집 일관성 점검
- [ ] 문서화 완료 (사용법 + 알려진 제약)

---

## Execution Log

### 2026-02-10
- [x] Phase 0: 앵커 생성
- [x] Phase 0: TSLib clone 완료 (`external/Time-Series-Library`)
- [x] Phase 1: wrapper/factory/config 연동 완료 (`variant=TSLIB`)
- [x] Phase 1: `downstream.py` freeze-rule 호환성 수정 (TSLib adapter 크래시 해결)
- [x] Phase 2: 6개 baseline forward 호환성 확인
- [x] Phase 3: 스모크 실행 완료
  - `SMK-AppliancesEnergy.TSLIB.DLinear.pr96.sd0`
  - `SMK-AppliancesEnergy.TSLIB.iTransformer.pr96.sd0`
  - 결과 파일: `artifacts/runs/<run_id>/downstream_metrics.json`
- [x] Phase 3: 본실험 plan 추가
  - `conf/plan/TSLIB_baseline_compare_main.yaml`
- [x] Phase 4: 본실험 단건 검증 실행
  - `SUP-AppliancesEnergy.TSLIB.DLinear.e20.vt.pr96.sd0`
  - `best_test_mse=0.574283`, `best_test_mae=0.550361`
  - 로그: `artifacts/plans/TSLIB_baseline_compare_main/logs/run__SUP-AppliancesEnergy.TSLIB.DLinear.e20.vt.pr96.sd0.log`
- [x] Phase 4: DLinear horizon sweep(3개) 실행 완료
  - `SUP-AppliancesEnergy.TSLIB.DLinear.e20.vt.pr96.sd0` -> `mse=0.574283`, `mae=0.550361`
  - `SUP-AppliancesEnergy.TSLIB.DLinear.e20.vt.pr192.sd0` -> `mse=0.745821`, `mae=0.651466`
  - `SUP-AppliancesEnergy.TSLIB.DLinear.e20.vt.pr336.sd0` -> `mse=1.002309`, `mae=0.770457`

---

## Step-by-Step Execution Plan (v1)

1. 환경 점검
   - 명령: `PYTHONPATH=src python -m itransformer.orchestrator.run plan=conf/plan/TSLIB_baseline_compare_smoke.yaml --resume`
   - 판정: `artifacts/plans/TSLIB_baseline_compare_smoke/manifest.json`에 스펙 2개 존재

2. 본실험 시작 (전체 baseline)
   - 명령: `PYTHONPATH=src python -m itransformer.orchestrator.run plan=conf/plan/TSLIB_baseline_compare_main.yaml`
   - 판정: `artifacts/plans/TSLIB_baseline_compare_main/manifest.json`에서 failed=0
   - 참고: `--filter` 실행 시 manifest에는 해당 subset만 기록됨 (run별 최종 상태는 `artifacts/runs/<run_id>/status.json` 확인)

3. 실패 재시도
   - 명령: `PYTHONPATH=src python -m itransformer.orchestrator.run plan=conf/plan/TSLIB_baseline_compare_main.yaml --resume`
   - 판정: 이전 failed run 재실행 후 완료

4. 결과 수집
   - 입력: `artifacts/runs/SUP-AppliancesEnergy.TSLIB.*.downstream_metrics.json`
   - 산출: 모델/예측길이별 `test.mse`, `test.mae`, `cost` 테이블

5. M0 대비 비교표 작성
   - 입력: M0 run의 `downstream_metrics.json`
   - 산출: `(TSLib - M0)` delta 표와 승패 매트릭스
