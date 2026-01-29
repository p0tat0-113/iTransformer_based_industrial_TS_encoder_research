# 03. Repository Layout

```
/workspace
  conf/                      # Hydra 설정 그룹
    analysis/                # analysis.* 설정
    data/                    # dataset 설정
    eval/                    # eval.* 설정
    ids/                     # ID 템플릿
    metadata/                # metadata 설정
    model/                   # 모델 변형(A0/A1/A2/P0~P4)
    optim/                   # optimizer
    plan/                    # 오케스트레이터 plan
    runtime/                 # device, seed
    ssl/                     # SSL pretrain 설정
    train/                   # downstream 학습 설정
  src/itransformer/          # 현재 사용하는 코드
    data/                    # 데이터 로더/팩토리
    models/                  # 모델 구현
    ssl/                     # Var-MAE, Patch-MAE
    evals/                   # eval 유틸
    analysis/                # 분석 유틸
    orchestrator/            # plan 실행
    utils/                   # id/metadata/metrics 등
  artifacts/                 # 실행 결과 저장 (run/op/cmp/agg)
  dataset/                   # 원본 데이터셋
  legacy/                    # 기존 코드(보관용, 사용하지 않음)
  documents/                 # 문서 모음
  requirements.txt
  Dockerfile
```

## legacy 디렉터리
- 원본 iTransformer fork에서 사용하던 코드 보관용
- 현재 실행은 `src/itransformer`만 사용

## documents 디렉터리
- 본 문서 전체 + 기존 문서/계획/분석 기록
