# 04. Configuration (Hydra)

## 1) 기본 개념
- 모든 설정은 `conf/` 아래에 그룹별 YAML로 존재
- 실행 시 `python -m <entry> key=value ...` 형태로 오버라이드

## 2) 핵심 그룹
- data: 데이터셋 설정
- model: 모델 변형(A0/A1/A2/P0~P4)
- ssl: Var-MAE / Patch-MAE 설정
- train: downstream 학습 설정
- eval: 평가 설정
- analysis: 집계/분석 설정
- metadata: 메타데이터 설정
- ids: run/op/cmp/agg ID 템플릿

## 3) 자주 쓰는 파라미터
- data.name / data.root_path / data.data_path
- model.variant
- train.mode: sl | ft | lp
- ssl.type: var_mae | patch_mae
- eval.op_code: S1/S2/S3/T1/T2/T3/R1/R2
- runtime.device: cpu | cuda

## 4) ID 템플릿
`conf/ids/default.yaml`
```
run_id: {I}.{D}.{V}.{H}.sd{seed}
op_id: {I}__OP={code}{op_hparams}__ON={RunID}
cmp_id: CMP.{D}.{I}__L={RunID_left}__R={RunID_right}
agg_id: AGG.{D}.{I}__ON={RunIDs}
```
- 실행 시 내부에서 `build_run_id`, `build_op_id`, `build_cmp_id`, `build_agg_id`로 생성됨

## 5) op_hparams_tag 주의
- zsh에서는 `eval.missing_rates=[...]` 반드시 따옴표로 감싸기
- 숫자를 넣어도 내부에서 문자열로 처리됨

## 6) 계획 파일(plan)
- `conf/plan/*.yaml` 사용
- 자세한 형식은 `09_orchestrator.md` 참고
