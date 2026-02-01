# 08. Experiment C (SSL)

## 1) SSL Pretrain (C-PR)
- Var-MAE: `ssl=var_mae`
- Patch-MAE: `ssl=patch_mae`

예시 (Var-MAE, 1 epoch):
```
python -m itransformer.pretrain \
  data=ETTh1 model=P2 ssl=var_mae ssl.mask_ratio=0.25 ssl.pretrain_epochs=1 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 train.num_workers=0 \
  run.code=C-PR-1 run.hparams_tag=mr0.25
```

예시 (Patch-MAE, 1 epoch):
```
python -m itransformer.pretrain \
  data=ETTh1 model=P2 ssl=patch_mae ssl.patch_len=8 ssl.mask_ratio=0.25 ssl.pretrain_epochs=1 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 train.num_workers=0 \
  run.code=C-PR-2 run.hparams_tag=mr0.25.pl8
```

## 2) Downstream (C-DS)
- SL: 랜덤 init
- FT: SSL ckpt 로드 + 전체 미세조정
- LP: SSL ckpt 로드 + encoder freeze

예시 (FT):
```
python -m itransformer.downstream \
  data=ETTh1 model=P2 train=downstream train.mode=ft train.epochs=1 \
  train.batch_size=16 train.num_workers=0 \
  train.ssl_ckpt_path=./artifacts/runs/C-PR-1.ETTh1.P2.mr0.25.sd0/pretrain_checkpoint.pt \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 \
  run.code=C-DS-2 run.hparams_tag=ft
```

## 3) Robustness (C-RB)
R1/R2는 eval op로 실행하고 agg로 집계한다.

예시 (R1 sweep):
```
python -m itransformer.eval \
  data=ETTh1 model=P2 model.patch.patch_len=8 metadata.enabled=false \
  eval.code=C-RB-1 eval.op_code=R1 "eval.missing_rates=[0.0,0.5]" \
  eval.on_run_id=C-DS-1.ETTh1.P2.sl.sd0
```

집계:
```
python -m itransformer.analysis_entry data=ETTh1 analysis.code=C-RB-1
python -m itransformer.analysis_entry data=ETTh1 analysis.code=C-RB-2
```
