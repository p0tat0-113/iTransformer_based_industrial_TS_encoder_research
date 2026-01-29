요청하신 대로 스크린샷의 촬영 시간 순서(17:59 → 18:14 → 18:25 → 18:35 → 18:40)와 내용의 논리적 흐름(공통 규칙 → 실험군 A → B → C → 요약)에 맞춰 텍스트를 추출하여 정리했습니다.

---

아래는 **현재 구현된 코드 기준**으로, exp_plan.md의 실험 식별 코드별로 생성되는 산출물(체크포인트/로그/메트릭 등)을 정리한 거야.
경로 패턴도 같이 적었고, **구현상 제약/누락**은 별도로 명시했어.

### 공통 산출물 위치 규칙

* Run(학습/사전학습): `artifacts/runs/<RunID>/`
* Op(평가/진단): `artifacts/ops/<OpID>/`
* CMP: `artifacts/cmp/<CmpID>/`
* AGG: `artifacts/agg/<AggID>/`
* Hydra 로그/설정: `artifacts/_hydra/<date>/<time>/.hydra/` + 실행 로그 파일

---

### 실험군 A (메타)

**A-TR-1 ~ A-TR-7 (학습)**
**현재 구현 기준:**

* **실제 학습 루프는 `itransformer.downstream` (train.mode=sl)에 있음.**
* `itransformer.train`은 **config/status만 저장하는 스텁**이므로 체크포인트/메트릭 없음.

**`itransformer.downstream` 사용 시 산출물:**

* config.yaml
* downstream_checkpoint.pt
* downstream_metrics.json (val/test MSE/MAE)
* status.json

**중요 제약:**

* `downstream`은 metadata.enabled=true일 때 **meta_emb를 모델에 전달하도록 수정됨** (A1/A2 메타 사용 가능).

**A-EV-1 ~ A-EV-3 (S1/S2/S3 평가)**
**엔트리포인트:** `itransformer.eval`
**산출물:**

* config.yaml
* op_results.json (mse/mae)
* status.json

**비고:**

* plot 생성 없음
* S2는 **sr_tag 사용 없이** 다운샘플링 factor로만 동작

**A-DIAG-1 ~ A-DIAG-2 (T1/T2)**
**엔트리포인트:** `itransformer.eval`
**산출물:**

* 동일하게 `op_results.json`, `status.json`, `config.yaml`

**비고:**

* T1/T2는 metadata.enabled=true 필요
* 셔플/마스크 결과는 **저장되지 않음** (메트릭만 저장)

**A-DIAG-3 (T3 비교)**
**구현 방식:** eval로 T3 baseline 결과 생성 + `analysis_entry` CMP 비교
**CMP 산출물:**

* cmp.json (좌/우 메트릭 + delta)
* status.json
* config.yaml

**비고:**

* 그래프/집계 없음 (M5 단계 필요)

---

### 실험군 B (Patching)

**B-TR-1 ~ B-TR-5 (SSL pretrain)**
**엔트리포인트:** `itransformer.pretrain`
**산출물:**

* config.yaml
* pretrain_checkpoint.pt
* pretrain_metrics.json (epoch별 loss)
* status.json

**비고:**

* loss curve 이미지는 없음
* patch_len 등은 config.yaml에 저장

**B-EV-1 ~ B-EV-5 (F1~F5 분석)**
**엔트리포인트:** `itransformer.analysis_entry`
**산출물 (F1/F2/F4/F5):**

* analysis.json
* status.json
* config.yaml
* F5는 `attn_first_layer.pt` 저장 시도

**중요 제약:**

* `analysis_entry`는 체크포인트를 **downstream_checkpoint.pt / checkpoint.pt만** 찾음
→ **pretrain_checkpoint.pt는 인식하지 않음**
→ B-TR pretrain 결과에 대해 F1~F5를 바로 실행하면 실패함
(코드 수정 또는 체크포인트 파일명 정리가 필요)

---

### 실험군 C (SSL)

**C-PR-1 / C-PR-2 (SSL pretrain)**

* **B-TR과 동일하게 pretrain 산출물 생성**

**C-DS-1 ~ C-DS-3 (SL/FT/LP downstream)**
**엔트리포인트:** `itransformer.downstream`
**산출물:**

* config.yaml
* downstream_checkpoint.pt
* downstream_metrics.json
* status.json

**C-RB-1 / C-RB-2 (R1/R2 강건성)**
**엔트리포인트:** `itransformer.eval`
**산출물:**

* op_results.json
* status.json
* config.yaml

**비고:**

* curve는 **자동 집계/그래프 없음**
* 여러 rate를 개별 Op으로 실행해야 함

**CMP / AGG**

* **CMP:** 구현됨 (cmp.json + status + config)
* **AGG:** 아직 미구현 (status/config만 생성, agg.json 없음)

---

### 요약: "실험 코드 ↔ 산출물" 핵심 정리

* **학습/사전학습 실행 결과** → `artifacts/runs/<RunID>/`
* pretrain: `pretrain_checkpoint.pt`, `pretrain_metrics.json`
* downstream: `downstream_checkpoint.pt`, `downstream_metrics.json`


* **평가/진단 실행 결과** → `op_results.json`
* **비교/분석 결과** → `cmp.json` 또는 `analysis.json`
* **로그/설정** → `artifacts/_hydra/<date>/<time>/`

필요하면, 위 내용을 바로 **exp_plan.md에 삽입 가능한 표 형태**로 정리해줄게.
특히 아래 항목은 exp_plan에 "주의/제약"으로 명시하는 걸 추천해:

* B-EV 분석은 **pretrain_checkpoint를 직접 사용 불가**
* A1/A2 학습에서 메타 입력이 **downstream에서도 사용되도록 수정됨**
* AGG(집계/그래프)는 아직 미구현

원하는 포맷(표/섹션/주석) 알려줘.
