# 01. Overview

## 1) 목적
이 코드는 iTransformer 기반 실험 플랫폼을 구축하기 위한 코드베이스다. 주요 목적은 다음과 같다.
- iTransformer 계열 모델만 남기고 실험 구조를 단순화
- Hydra 기반 설정 관리로 재현성 확보
- 실험군 A/B/C를 일관된 규칙으로 반복 실행
- 결과(op/cmp/agg) 분리 및 집계 자동화

## 2) 실험군 요약
- 실험군 A (메타 임베딩)
  - A0: 메타 미사용
  - A1: 실제 메타 사용 (Add/Concat/Fusion)
  - A2: constant/UNK 메타 사용 (Add/Concat/Fusion)
  - A-EV: S1/S2/S3 시나리오 평가
  - A-DIAG: T1/T2/T3 진단

- 실험군 B (패칭)
  - P0: 패칭 없음 (baseline)
  - P1~P4: 패칭 구조 변형
  - B-EV-1: 비용 집계
  - B-EV-2: patch_len별 성능-비용
  - B-EV-4: CKA (첫/마지막 encoder layer)

- 실험군 C (SSL)
  - C-PR: Var-MAE / Patch-MAE 사전학습
  - C-DS: SL/FT/LP 다운스트림 학습
  - C-RB: R1/R2 강건성 평가

## 3) 실행 구조
- **Run**: 학습 실행 단위 (pretrain/downstream)
- **Op**: 평가 실행 단위 (eval)
- **CMP**: 두 run 비교 (analysis)
- **AGG**: 다수 run 집계 (analysis)

각 결과는 `artifacts/` 아래에 분리 저장된다. 자세한 스키마는 `10_outputs_metrics.md` 참고.

## 4) 핵심 특징
- Hydra 기반 설정/오버라이드 (conf/)
- run/op/cmp/agg ID 규칙 통일 (ids/default.yaml)
- meta embedding 캐시/검증/빌드 파이프라인
- 오케스트레이터로 실험 플랜 실행 가능
