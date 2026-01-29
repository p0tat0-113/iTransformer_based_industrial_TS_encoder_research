# 02. Quickstart

## 1) 환경 준비
```
# 권장
python>=3.10
pip install -r requirements.txt
export PYTHONPATH=/workspace/src
```

## 2) 데이터 준비
- 데이터는 `dataset/` 아래에 둔다.
- 예: `dataset/ETT-small/ETTh1.csv`

## 3) GPU 확인 (선택)
```
python legacy/gpu_acceleration_chk.py
```

## 4) A 실험군 최소 실행 (1 epoch)
```
python -m itransformer.downstream \
  data=ETTh1 model=A0 train=downstream train.mode=sl \
  train.epochs=1 train.batch_size=16 train.num_workers=0 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 \
  run.code=A-TR-1 run.hparams_tag=base
```

## 5) 오케스트레이터 실행 (plan)
```
python -m itransformer.orchestrator.run plan=conf/plan/exp_plan.yaml
```
- 실행 로그: `artifacts/plans/<plan_id>/logs/`
- 결과 상태: `artifacts/plans/<plan_id>/manifest.json`

## 6) 자주 겪는 오류
- zsh에서 `eval.missing_rates=[...]`는 반드시 따옴표로 감싸야 함
- metadata cache가 없으면 `metadata.cache.build=true` 필요
- GPU 사용 시 `runtime.device=cuda`

상세한 실험 방법은 실험군 문서(06~08) 참고.
