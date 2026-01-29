# 07. Experiment B (Patching)

## 1) 모델 변형
- P0: 패칭 없음 (baseline)
- P1: patch -> mean pooling (1 token)
- P2: all patch tokens (global)
- P3: same timestep attention
- P4: local window attention

## 2) Pretrain (B-TR)
예시 (P2, patch_len=8, 1 epoch):
```
python -m itransformer.pretrain \
  data=ETTh1 model=P2 ssl=patch_mae ssl.patch_len=8 ssl.pretrain_epochs=1 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 train.num_workers=0 \
  run.code=B-TR-3 run.hparams_tag=pl8
```

P0은 패칭이 없으므로 `ssl=var_mae` 사용을 권장:
```
python -m itransformer.pretrain \
  data=ETTh1 model=P0 ssl=var_mae ssl.pretrain_epochs=1 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 train.num_workers=0 \
  run.code=B-TR-1 run.hparams_tag=base
```

## 3) Downstream LP (B-DS)
P1~P4는 LP 평가를 수행한다. P0는 baseline으로 별도 그룹(patch_len=none)으로 집계한다.

예시 (P2, LP):
```
python -m itransformer.downstream \
  data=ETTh1 model=P2 train=downstream train.mode=lp train.epochs=1 \
  train.batch_size=16 train.num_workers=0 \
  train.ssl_ckpt_path=./artifacts/runs/B-TR-3.ETTh1.P2.pl8.sd0/pretrain_checkpoint.pt \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 \
  run.code=B-DS-2 run.hparams_tag=pl8
```

## 4) 집계 (B-EV)
```
python -m itransformer.analysis_entry data=ETTh1 analysis.code=B-EV-1
python -m itransformer.analysis_entry data=ETTh1 analysis.code=B-EV-2
python -m itransformer.analysis_entry data=ETTh1 analysis.code=B-EV-4
```
- B-EV-1: 비용 집계
- B-EV-2: 성능/비용 trade-off
- B-EV-4: 첫/마지막 encoder layer CKA

## 5) P0 baseline 처리
- 집계에서 patch_len="none"으로 별도 그룹 처리
- P0는 패칭 구조 비교의 기준선으로 포함
