# iTransformer Experimental Platform

이 저장소는 iTransformer 기반 실험 플랫폼이다. 전체 문서는 `documents/README.md`에 정리되어 있다.

## Quick Links
- 문서 인덱스: `documents/README.md`
- 실험 계획: `documents/plans/exp_plan.md`
- 리팩토링 진행: `documents/plans/refactor_plan.md`
- 메타데이터 가이드: `documents/guides/metadata_guide.md`

## Quickstart (짧은 실행)
```
export PYTHONPATH=/workspace/src
python -m itransformer.downstream data=ETTh1 model=A0 train=downstream train.mode=sl \
  train.epochs=1 train.batch_size=16 train.num_workers=0 \
  metadata.enabled=false runtime.device=cpu runtime.seed=0 \
  run.code=A-TR-1 run.hparams_tag=base
```

## Legacy
기존 iTransformer fork 코드는 `legacy/`로 이동했으며, 삭제하지 않고 보관한다.
