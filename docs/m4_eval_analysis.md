# M4 평가/진단/분석 실행 가이드

이 문서는 M4 단계에서 추가된 평가/진단/분석 기능의 실행 방법을 정리한다.

## 1. 공통 전제
- 실행은 Hydra 기반으로 동작한다.
- 평가(eval)는 `itransformer.eval`, 분석(analysis)은 `itransformer.analysis_entry`로 실행한다.
- `run_id` 또는 `ckpt_path`가 필요하다.
- 메타데이터는 실험군 A에서만 사용한다.

## 2. Scenario / Robustness / Diagnostics 실행 (eval)

### 2.1 기본 실행
```bash
PYTHONPATH=/workspace/src python -m itransformer.eval \
  data=ETTh1 model=A0 eval.op_code=S1 eval.on_run_id=<RUN_ID> \
  metadata.enabled=false runtime.device=cuda
```

### 2.2 op_code 목록
- S1: 입력에 Gaussian noise 추가 (레벨 sweep)
- S2: 데이터 변형(다운샘플링 후 보간)
- S3: 입력에 scale + bias drift
- R1: 값 레벨 결측 (raw 값 마스킹)
- R2: 채널 결측 (채널 drop/mask)
- T1: 메타 매칭 셔플 (A 실험군 전용)
- T2: 메타 결측 sweep (A 실험군 전용)
- T3: baseline 평가 (A1 vs A2 비교용)

### 2.3 레벨 지정 방식
- `eval.op_hparams_tag`에 레벨을 준다.
- S1/S3는 `l1/l2/l3` 혹은 숫자 사용 가능
  - `l1=0.05`, `l2=0.1`, `l3=0.2`
- R1/R2는 `eval.op_hparams_tag`를 숫자로 주면 결측률로 사용
- S2는 다운샘플링 배수로 사용
  - `d2`, `d4` 또는 숫자 (`2`, `4`)

예시:
```bash
# S1 레벨2
PYTHONPATH=/workspace/src python -m itransformer.eval \
  data=ETTh1 model=A0 eval.op_code=S1 eval.op_hparams_tag=l2 eval.on_run_id=<RUN_ID>

# R2 채널 결측 30%
PYTHONPATH=/workspace/src python -m itransformer.eval \
  data=ETTh1 model=A0 eval.op_code=R2 eval.op_hparams_tag=0.3 eval.on_run_id=<RUN_ID>

# S2 다운샘플 x4
PYTHONPATH=/workspace/src python -m itransformer.eval \
  data=ETTh1 model=A0 eval.op_code=S2 eval.op_hparams_tag=d4 eval.on_run_id=<RUN_ID>
```

### 2.4 메타 진단 (T1/T2)
```bash
PYTHONPATH=/workspace/src python -m itransformer.eval \
  data=ETTh1 model=A1Add metadata.enabled=true \
  eval.op_code=T1 eval.on_run_id=<RUN_ID>
```

## 3. 분석 실행 (analysis_entry)

### 3.1 F1/F2/F4/F5
```bash
PYTHONPATH=/workspace/src python -m itransformer.analysis_entry \
  data=ETTh1 model=A0 analysis.code=F1 analysis.on=<RUN_ID>
```

- F1: 파라미터 수 + 추론 시간(기본)
- F2: F1 + 성능 지표 로드(가능 시)
- F4: CKA (첫/마지막 블록)
- F5: attention map 저장

### 3.2 CMP (A1 vs A2 비교)
```bash
PYTHONPATH=/workspace/src python -m itransformer.analysis_entry \
  data=ETTh1 analysis.code=T3 \
  analysis.left=<RUN_ID_LEFT> analysis.right=<RUN_ID_RIGHT>
```

CMP는 `cmp.json`에 좌/우 메트릭과 delta를 함께 저장한다.

## 4. 산출물 위치
- eval 결과: `artifacts/ops/<op_id>/op_results.json`
- analysis 결과: `artifacts/agg/<agg_id>/analysis.json`
- CMP 결과: `artifacts/cmp/<cmp_id>/cmp.json`

## 5. 참고 사항
- eval은 `eval.on_run_id` 또는 `eval.ckpt_path` 중 하나가 필요하다.
- 분석은 `analysis.on`에 단일 run_id를 넣는다.
- CKA/attention은 모델 구조에 따라 값이 없을 수 있다.
