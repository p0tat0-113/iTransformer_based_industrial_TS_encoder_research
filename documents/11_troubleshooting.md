# 11. Troubleshooting

## 1) zsh: no matches found
- 예: `eval.missing_rates=[0.0,0.5,1.0]`
- 해결: 따옴표로 감싸기
  - "eval.missing_rates=[0.0,0.5,1.0]"

## 2) metadata cache not found
- `metadata.cache.build=true` 사용
- 또는 `itransformer.tools.build_metadata_cache` 실행

## 3) meta와 토큰 수 mismatch
- 시간 feature가 추가되면 토큰 수 증가
- 코드에서 meta padding 처리됨 (A1/A2)

## 4) GPU 사용 안됨
- `runtime.device=cuda` 설정
- 컨테이너가 CUDA 사용 가능해야 함

## 5) patch_len 오류
- P1(mean_pool)은 내부적으로 patch_len=seq_len로 처리
- Patch downstream에서 patch_len이 없으면 ckpt에서 자동 주입

## 6) B/C 집계가 섞임
- analysis에서 run.code 필터 적용됨
- 필요 시 `analysis.run_code_prefixes` 또는 `analysis.run_codes`로 제한
