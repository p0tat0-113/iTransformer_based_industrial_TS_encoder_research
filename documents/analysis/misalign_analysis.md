아래는 `documents/plans/exp_plan.md`, `documents/analysis/output_result_recator_plan.md`, `documents/plans/refactor_plan.md`를 기준으로 현재 코드가 어떻게 구현되어 있는지 **광범위·심층 점검한 보고서**야. (코드 기준: `src/itransformer/*`, `conf/*` 전반)

**요약**
- A/B/C 실험의 **핵심 루프(Pretrain/Downstream/Eval/Analysis, op/cmp/agg 분리)**는 대부분 계획과 부합.
- 그러나 **A2(상수/UNK 메타) 구현**, **op_id 형식**, **B‑EV 집계 대상 범위**, **오케스트레이터 검증 단계**에서 **문서와 실제 구현 간 불일치**가 분명함.
- M6 전에 반드시 정리/보완해야 할 항목이 있음.

---

**1) ID 규칙 및 오케스트레이터 정합성**
- **Run ID 규칙**은 `conf/ids/default.yaml` 및 `src/itransformer/utils/ids.py`와 **일치**.
  - 예: `${run.code}.${data.name}.${model.variant}.${run.hparams_tag}.sd${runtime.seed}`
- **Op ID 규칙**은 **부분 불일치**.
  - 문서: `{I}__OP={code}[.{op_hparams}]__ON={RunID}`
  - 코드: `conf/ids/default.yaml` → `${eval.code}__OP=${eval.op_code}${eval.op_hparams_tag}__ON=${eval.on_run_id}`
  - 문제: `eval.op_hparams_tag`는 **parse_level/parse_downsample에 바로 쓰이므로 점(.) 없이 들어가야 정상**인데, **op_id 형식에는 점이 필요**.
  - `build_op_id()` 유틸이 존재하나 `eval.py`에서 사용하지 않음. (`src/itransformer/utils/ids.py`, `src/itransformer/eval.py`)
- **CMP/AGG ID 규칙**은 대체로 맞지만, `build_agg_id()`가 `ids.agg_id` 템플릿을 실제로 활용하지는 않음(이미 Hydra interpolation된 문자열이 들어옴). 실질적으로는 문제가 없지만 **single source of truth** 관점에서 혼선 가능.

- **오케스트레이터 검증 기능**: `refactor_plan.md`에 “필수 필드/타입 검증”이 명시돼 있으나, `src/itransformer/orchestrator/plan.py`는 **sweep id_template만 검증**. dataset/variant/op_code 등 검증 미구현.

---

**2) 실험군 A (메타 임베딩)**
구현 현황:
- A0/A1/A2 모델 구조는 `conf/model/A*.yaml`, `src/itransformer/models/itransformer.py`에 구현되어 있음.
- **메타 임베딩 로딩 및 캐시**는 `src/itransformer/utils/metadata.py`에서 완료.
- **downstream/eval에서 meta_emb 전달**은 구현됨 (`src/itransformer/downstream.py`, `src/itransformer/eval.py`).

중요 불일치:
- **A2(Constant/UNK baseline 미구현)**
  - `conf/model/A2*.yaml`에는 `meta.source: constant`가 있으나,
  - 실제 로직에서 `meta.source`를 **어디에서도 사용하지 않음**.
  - 결과적으로 A2는 **A1과 동일하게 실제 metadata 임베딩을 사용**할 가능성이 높음.
  - 이는 **A-DIAG-3의 핵심 실험 목적(semantic ablation)**을 깨뜨리는 심각한 불일치.

시나리오/진단:
- **A‑EV‑1~3 (S1/S2/S3)**: `src/itransformer/eval.py`에서 적용 및 `op_results.json`에 `op_params` 저장됨.
- **A‑DIAG‑1 (T1 Shuffle)**: base/shuffled/delta 저장 구현됨.
- **A‑DIAG‑2 (T2 Missing sweep)**: `missing_rates`로 단일 op sweep 구현됨.
- **A‑DIAG‑3 (T3 A1 vs A2 CMP)**: cmp.json left/right/delta 저장 구현됨.
- 단, T3는 **특별한 조작 없이 일반 eval 수행**이므로, **A2가 제대로 “상수 메타”로 동작하지 않으면 비교 무의미**.

---

**3) 실험군 B (Patching)**
구현 현황:
- P0~P4 구조는 `conf/model/P*.yaml` + `src/itransformer/models/patch_transformer.py`에 구현.
  - P1: mean_pool (patch_len=seq_len)
  - P2: all tokens (global)
  - P3: same_time (time-only)
  - P4: local window
- Patch-MAE 사전학습: `src/itransformer/ssl/patch_mae.py`
- Patch downstream: `src/itransformer/models/patch_transformer.py`

주의/불일치:
- **B‑EV 집계 범위 오염 위험**
  - `src/itransformer/analysis_entry.py`는 **variant와 dataset만 기준으로 runs_dir 전체를 스캔**.
  - **run.code 또는 실험군 구분 없이** 집계 → C 실험의 P2/P3 run까지 섞일 위험.
  - B‑EV‑1(비용), B‑EV‑4(CKA)은 **pretrain 기준**이어야 하는데, downstream run이 섞일 수 있음.
- **P1 patch_len 값 문제**
  - `conf/model/P1.yaml`에 `patch_len: 0`, mean_pool은 코드에서 `seq_len`로 해석하지만
  - 집계 시 `_patch_len_from_cfg()`는 0을 반환 → agg 그룹 키가 `patch_len=0`으로 기록됨.
- **B‑EV‑2는 P0 제외** 조건을 코드에서 지킴(variants 기본값 P1~P4).
- B‑EV‑4 CKA는 “first_layer_cka, last_layer_cka, delta_cka”로 저장됨.
  - 다만 계산 방식은 **embedding ↔ layer output** 기준임.
  - 문서 의도가 “first vs last output의 CKA”였다면 구현과 약간 다름. (명시 확인 필요)

---

**4) 실험군 C (SSL)**
구현 현황:
- **Var‑MAE / Patch‑MAE pretrain** 구현 (`src/itransformer/ssl/var_mae.py`, `patch_mae.py`)
- **C‑DS (SL/FT/LP)** 구현 (`src/itransformer/downstream.py`)
  - FT에서 `freeze_epochs` 지원
  - LP는 encoder/embedding freeze + projector 학습
- **C‑RB (R1/R2)** 단일 op sweep 및 agg 집계 구현됨 (`eval.py`, `analysis_entry.py`)

주의/불일치:
- **다운스트림 patch_len 연계 문제**
  - Patch‑MAE pretrain checkpoint를 downstream에 연결할 때,
  - `model.patch.patch_len`이 downstream config에 명시되지 않으면 오류 발생.
  - SSL ckpt에서 patch_len을 자동 복구하는 로직이 없음.
- **C‑RB 집계는 variant 기준(P0~P4)으로 분류**
  - C 실험이 실제로 P0~P4 variant로 운영되는지 문서와 일치하는지 확인 필요.

---

**5) 산출물 스키마 (output_result_recator_plan.md) 준수 여부**
대부분 구현됨:
- pretrain/downstream metrics 스키마: summary/curves/cost/notes → **코드와 일치**.
  - pretrain val_loss는 masked_only, notes에 명시.
  - grad_norm, lr, early stopping 모두 기록.
- op/cmp/agg 분리: **완료**.
- A‑EV‑1~3 op_params 기록, A‑DIAG‑1/2/3 저장 규칙 **구현됨**.
- C‑RB op_results에 metrics_by_rate 기록 및 agg 집계 **구현됨**.

불완전/주의:
- `op_results.json` 자체에는 `op_code`/`op_hparams_tag`가 포함되지 않음. (config.yaml에만 존재)
- “집계는 같은 모델끼리” 요구는 C‑RB/B‑EV에서 대부분 반영되었으나,
  **run.code 필터가 없어** 실험군 간 섞일 가능성 있음.

---

**6) refactor_plan.md 대비 누락 또는 불일치**
- “Plan 로더 검증(필수 필드/타입)” 미구현.
- “B‑EV‑1/2/4 집계 로직 실험군별 분리”는 문서와 달리 **전역 스캔 방식**.
- “A2 constant/UNK baseline”은 설계만 있고 코드 반영 없음.
- `train.py`는 여전히 stub (M6에서 제거 대상).

---

**핵심 리스크 / 우선순위 이슈**
1) **A2 상수/UNK baseline 미구현**
   - A‑DIAG‑3 실험 의미 붕괴.
   - `meta.source`에 따른 분기 또는 A2 전용 metadata.jsonl 경로/템플릿 강제 필요.

2) **B‑EV 집계 오염**
   - `analysis_entry.py`에서 run.code 필터 부재로 B/C 실험 결과 혼합 가능.
   - B‑EV‑1/4는 pretrain 기반, B‑EV‑2는 downstream LP 기반이라는 구분 필요.

3) **Op ID 형식 불일치**
   - op_hparams_tag parsing vs op_id 구성 규칙 충돌.
   - `build_op_id()`를 실제 eval 단계에서 사용하거나 ids 템플릿에서 dot 처리 분기 필요.

4) **Patch_len 연동 자동화 미흡**
   - downstream에서 patch_len을 수동 지정하지 않으면 실행 실패 위험.
   - SSL ckpt에 patch_len 저장 → downstream에서 자동 주입하는 방식 필요.

---

**추가 확인이 필요한 질문**
1) A2 baseline을 **“실제 메타를 쓰지 않고 상수/UNK 텍스트를 넣는 방식”**으로 명확히 하려면,
   - `metadata.jsonl`을 A2 전용으로 둘 건지,
   - 아니면 코드에서 `meta.source=constant` 분기를 구현할 건지 결정 필요.
2) B‑EV‑4의 CKA 정의:
   - 현재는 **input embedding ↔ first/last layer** 기준.
   - 의도는 **first layer ↔ last layer**인가?
3) B‑EV 집계에서 **run.code 기준 필터링**을 넣을지?
   - 예: `B-TR-*`만 집계하도록.

---

원하면 위 내용을 기준으로 **M6 직전 정리 체크리스트**를 만들어서 바로 수정 작업에 들어갈 수 있어.
