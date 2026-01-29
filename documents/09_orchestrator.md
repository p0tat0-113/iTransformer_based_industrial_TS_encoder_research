# 09. Orchestrator

## 1) 목적
- run/op/cmp/agg를 계획 파일(plan)로 정의하고 자동 실행
- 실행 순서: run -> op -> cmp/agg
- 결과는 `artifacts/plans/<plan_id>/`에 저장

## 2) 기본 실행
```
python -m itransformer.orchestrator.run plan=conf/plan/exp_plan.yaml
```

옵션:
- `--resume`: 완료된 spec skip
- `--only=run|op|cmp|agg`: 특정 타입만 실행
- `--filter=<substr>`: id substring 필터

## 3) plan 스키마
```
plan_id: <string>

runs:
  - id: <run_id>
    entry: pretrain|downstream|train
    overrides:
      - data=ETTh1
      - model=A0
      - ...

ops:
  - id: <op_id>
    entry: eval
    overrides:
      - data=ETTh1
      - eval.op_code=S1
      - eval.on_run_id=<run_id>

cmps:
  - id: <cmp_id>
    entry: analysis
    overrides:
      - data=ETTh1
      - analysis.left=<run_id>
      - analysis.right=<run_id>

aggs:
  - id: <agg_id>
    entry: analysis
    overrides:
      - data=ETTh1
      - analysis.code=B-EV-1
```

## 4) sweep 규칙
- plan 내 각 아이템에 `sweep:`을 추가하면 Cartesian product로 확장됨
- `id_template`와 `overrides` 안에 `{}` 템플릿 변수 사용 가능

## 5) 검증 규칙
- run: `data`, `model` 필수
- op: `data`, `eval.op_code` 필수 + `eval.on_run_id` 또는 `eval.ckpt_path` 필수
- cmp/agg: `data`, `analysis.code` 필수

## 6) 결과
- manifest: `artifacts/plans/<plan_id>/manifest.json`
- 로그: `artifacts/plans/<plan_id>/logs/*.log`
