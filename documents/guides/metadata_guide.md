# 메타데이터 JSONL 가이드 (실험군 A)

이 프로젝트는 각 데이터셋의 센서/변수에 텍스트 메타데이터를 연결하기 위해 `metadata.jsonl`을 사용합니다.
로더는 **한 줄에 하나의 JSON 객체**가 들어오는 형식을 기대하며, 최소 필드로 `sensor_id`가 필요합니다.

## 1) 파일 위치
- 경로: `dataset/<name>/metadata.jsonl`
- 예시: `dataset/traffic/metadata.jsonl`

## 2) sensor_id 규칙
- **CSV 기반 데이터셋 (ETT/Traffic/Weather/ECL/Exchange)**: `sensor_id`는 **컬럼명과 동일**해야 합니다.
- **PEMS/Solar**: 문자열 인덱스 사용: `"0" ... "N-1"`
  - PEMS 파일별로 센서 수가 다르면, 존재하지 않는 ID는 자동으로 무시됩니다.

## 3) JSONL 최소 형식
```json
{"sensor_id": "OT"}
{"sensor_id": "var_1"}
```

## 4) 권장 필드 (synthetic metadata)
스키마는 **데이터셋마다 다를 수 있으므로 자유롭게 확장** 가능합니다.
실험군 A에서 자주 사용하는 필드:
- `type`, `unit`, `quality`, `sr_tag`

예시:
```json
{"sensor_id":"OT","type":"power","unit":"kW","quality":"1","sr_tag":"orig"}
{"sensor_id":"var_1","type":"temp","unit":"C","quality":"0","sr_tag":"downsample"}
```

## 5) 템플릿 기반 텍스트 직렬화
메타데이터 텍스트는 `conf/metadata/<dataset>.yaml`의 `template`으로 직렬화됩니다.

### 옵션 A: 전체 JSON 사용
```yaml
template: "{json}"
```
모든 JSON 필드를 문자열로 직렬화합니다.

### 옵션 B: 특정 필드만 사용
```yaml
template: "type: {type}; unit: {unit}; quality: {quality}; sr_tag: {sr_tag}"
```
존재하지 않는 필드는 `UNK_<field>`로 치환됩니다.

### 옵션 C: 커스텀 text 필드 사용
```json
{"sensor_id":"OT","text":"power sensor; kW; quality=1"}
```
```yaml
template: "{text}"
```

## 6) UNK 처리
`conf/metadata/<dataset>.yaml`에서 설정:
- `unk_token`: 기본 `UNK`
- `unk_template`: 기본 `UNK_{field}`

## 7) Gemini 임베딩 캐시
임베딩은 **사전 생성 후 캐시**하는 방식입니다.
- 모델: `gemini-embedding-001` (3072차원)
- API 키: 환경변수 사용 (예: `GEMINI_API_KEY`)
- 캐시 파일: `./artifacts/metadata/<data.name>_<data.data_path>.pt`

텍스트만 덤프:
```
PYTHONPATH=/workspace/src python -m itransformer.tools.build_metadata_cache \
  data=Traffic metadata_builder.dump_texts=true metadata.cache.build=false
```

API로 임베딩 생성 (클라이언트 구현 완료 시):
```
PYTHONPATH=/workspace/src python -m itransformer.tools.build_metadata_cache \
  data=Traffic metadata.cache.build=true
```
