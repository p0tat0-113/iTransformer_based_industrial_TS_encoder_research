# 06. Experiment A (Metadata)

## 1) 학습 (A-TR)
- A0: 메타 미사용
- A1: 실제 메타 (Add/Concat/Fusion)
- A2: constant/UNK 메타 (Add/Concat/Fusion)

예시 (A0, 1 epoch):
```
python -m itransformer.downstream \
  data=ETTh1 model=A0 train=downstream train.mode=sl \
  train.epochs=1 train.batch_size=16 train.num_workers=0 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 \
  run.code=A-TR-1 run.hparams_tag=base
```

예시 (A1Add, cache build 1회):
```
python -m itransformer.downstream \
  data=ETTh1 model=A1Add train=downstream train.mode=sl \
  train.epochs=1 train.batch_size=16 train.num_workers=0 \
  metadata.enabled=true metadata.cache.build=true \
  runtime.device=cpu runtime.seed=0 \
  run.code=A-TR-2 run.hparams_tag=base
```

## 2) 시나리오 평가 (A-EV)
- S1: noise
- S2: downsample
- S3: bias/scale drift

예시 (S1):
```
python -m itransformer.eval \
  data=ETTh1 model=A0 metadata.enabled=false \
  eval.code=A-EV-1 eval.op_code=S1 eval.op_hparams_tag=0.1 \
  eval.on_run_id=A-TR-1.ETTh1.A0.base.sd0 \
  runtime.device=cpu
```

## 3) 진단 (A-DIAG)
- T1: shuffle (A1만)
- T2: missing rate sweep (A1만)
- T3: A1 vs A2 비교 (CMP)

예시 (T2, zsh 주의):
```
python -m itransformer.eval \
  data=ETTh1 model=A1Add metadata.enabled=true metadata.cache.build=false \
  eval.code=A-DIAG-2 eval.op_code=T2 \
  "eval.missing_rates=[0.0,0.5,1.0]" \
  eval.on_run_id=A-TR-2.ETTh1.A1Add.base.sd0 \
  runtime.device=cpu
```

예시 (CMP: A1 vs A2):
```
python -m itransformer.analysis_entry \
  data=ETTh1 analysis.code=A-DIAG-3 \
  analysis.left=A-TR-2.ETTh1.A1Add.base.sd0 \
  analysis.right=A-TR-5.ETTh1.A2Add.base.sd0
```
