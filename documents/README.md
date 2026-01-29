# iTransformer Experimental Platform Documentation

이 문서는 `/workspace` 코드베이스의 구조, 실험 방법, 설정 방식, 산출물 규칙을 처음 보는 사람도 이해할 수 있도록 정리한 안내서다.

## 문서 구성
- 01_overview.md: 프로젝트 목표와 전체 구성 요약
- 02_quickstart.md: 설치/실행/스모크 테스트 빠른 시작
- 03_repository_layout.md: 디렉터리 구조와 역할
- 04_configuration.md: Hydra 설정 체계 상세
- 05_metadata_pipeline.md: 메타데이터 규칙/캐시/임베딩
- 06_experiment_A.md: 실험군 A (메타 임베딩) 실행 방법
- 07_experiment_B.md: 실험군 B (패칭) 실행 방법
- 08_experiment_C.md: 실험군 C (SSL) 실행 방법
- 09_orchestrator.md: 오케스트레이터/플랜 사용법
- 10_outputs_metrics.md: 산출물/메트릭 스키마
- 11_troubleshooting.md: 자주 발생하는 문제와 해결

## 참고 문서(기존 자료)
- plans/exp_plan.md: 실험 설계 원본
- plans/refactor_plan.md: 리팩토링 진행 로그
- guides/metadata_guide.md: metadata.jsonl 작성 가이드
- analysis/output_result_recator_plan.md: 산출물 스키마 합의안
- analysis/output_result_analysis.md: 산출물 검증용 노트
- analysis/misalign_analysis.md: 계획-구현 불일치 점검
- legacy/README_original.md: 기존 README 원본
- legacy/docs/legacy_docs: 이전 docs 디렉터리 내용
