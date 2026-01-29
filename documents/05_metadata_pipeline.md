# 05. Metadata Pipeline

## 1) 기본 규칙
- metadata.jsonl은 데이터셋별로 존재
- CSV 기반 데이터(ETT/Traffic/Weather/ECL/Exchange): sensor_id = 컬럼명
- PEMS/Solar: sensor_id = "0".."N-1" 문자열

## 2) metadata.jsonl 형식
예시:
```
{"sensor_id": "OT", "type": "temp", "unit": "C"}
{"sensor_id": "HUFL", "type": "humidity", "unit": "%"}
```

## 3) 템플릿 직렬화
- `conf/metadata/<dataset>.yaml`에서 template 지정
- 기본은 `{json}` (전체 JSON 직렬화)
- 필드 템플릿을 쓰는 경우 누락 필드는 UNK 처리

## 4) constant/UNK 메타 (A2)
- `model.meta.source=constant`일 때
- 실제 metadata.jsonl을 읽지 않고 **모든 sensor에 동일한 UNK 토큰** 사용
- 캐시 파일도 별도로 저장됨

## 5) 캐시 빌드
```
python -m itransformer.tools.build_metadata_cache data=ETTh1 metadata.cache.build=true
```
또는 downstream/pretrain에서 `metadata.cache.build=true`를 주면 자동 빌드

## 6) 검증
```
python -m itransformer.tools.validate_metadata data=ETTh1
```

## 7) Gemini 설정
- 기본 모델: `gemini-embedding-001`
- 차원: 3072
- API 키: `GEMINI_API_KEY`
