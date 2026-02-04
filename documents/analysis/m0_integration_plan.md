# M0 (Multi-scale Latent Slots) 통합 구현 계획

목표: `documents/analysis/new_model_struc.md`에 정의된 **M0(iTransformer_MultiSlot)** 를 현재 코드베이스의 **훈련 프레임워크/설정(Hydra)/실험 플랜(orchestrator)** 에 “자연스럽게” 녹여서, 기존과 동일한 방식으로 실험을 반복 실행할 수 있게 만든다.

본 문서는 **구현(코딩) 전에** 필요한 변경 포인트/설계 결정을 정리한 계획서다.

---

## 0) 스코프 / 비스코프

### 스코프(1차 실험 확정 사항 반영)
- Patchify: **right-pad(0) + non-overlap(stride=p)**, padding mask는 1차에서 생략
- 입력 처리: 순정 iTransformer처럼 `x`와 `x_mark`를 variate 축으로 concat 후 inversion
- `x_mark`도 **동일 경로(멀티스케일 slot embedding)** 로 처리 후 encoder에 함께 투입, 출력에서 제거
- Temporal encoder: **scale별 별도(conv_p)**, `m_p==1`이면 bypass
- PMA: `pma_heads=8`, slotizer 뒤에 **FFN+residual(ON)**
- slot concat 순서 고정: `[p=8 (2 slots)] -> [p=32 (1)] -> [p=L (1)]`
- slot-id embedding: 1차에서는 미사용(Seeds로만 구분)
- `use_norm`: `x`만 normalize, `x_mark`는 그대로

### 비스코프(이번 구현에서 하지 않음)
- diversity loss / slot-id embedding / slot 분화 유도 손실
- “순정 iTransformer로 환원” 디버그 모드
- SSL(pretrain)과의 weight transfer 정합성(VarMAE/PatchMAE → M0)

---

## 1) 기존 프레임워크/설정 시스템 요약(통합 관점)

### 1.1 모델 선택(Factory)
- `src/itransformer/models/factory.py:build_model(cfg)` 가 `cfg.model.variant` 및 `cfg.model.patch.enabled`에 따라
  - `ITransformer`(P0) 또는 `PatchITransformer`(P1~P4)를 생성한다.
- M0는 **새 variant**이므로 `build_model()`에 분기를 추가하여 M0 클래스를 반환해야 한다.

### 1.2 학습 엔트리포인트
- Supervised(예측) 학습: `src/itransformer/downstream.py`
  - 모델 인터페이스: `model(x_enc, x_mark, meta_emb=None)` 형태를 사용
  - 출력 shape: `[B, pred_len, N]` 기대
- Eval: `src/itransformer/eval.py` 또한 `build_model(cfg)` 기반

### 1.3 Hydra 설정 구조
- 모델 설정은 `conf/model/*.yaml` (variant별)로 관리되고, `conf/config.yaml` defaults로 조립됨.
- “struct 모드” 제약 때문에 **override할 키는 기본 config에 정의되어 있어야** `+` 없이 override 가능.
  - (단, 이번 M0 실험에서는 필요 키를 `conf/model/M0.yaml`에 “정식 키”로 정의해서 안전성을 확보)

---

## 2) M0 통합 전략(최소 침습 + 실험 친화)

핵심 원칙:
1. **훈련 루프(`downstream.py`)는 그대로 둔다.**
2. M0는 **새 모델 클래스 + config + factory 등록**만으로 선택 가능해야 한다.
3. 기존 출력 계약을 지킨다:
   - 입력: `x_enc: [B,L,N]`, `x_mark: [B,L,C] (optional)`
   - 출력: `[B,pred_len,N]`
4. “변수 간 mixing”은 encoder self-attn에서만 발생하도록, embedding 단계는 토큰 독립 연산으로 구현한다.

---

## 3) 구현 단계별 계획(파일 단위)

### Step A) Config: `conf/model/M0.yaml` 추가
목표: M0를 `model=M0`로 선택 가능 + 하이퍼파라미터 override 가능하게 구성.

권장 구조(예시):
- iTransformer 기본 하이퍼파라미터는 P0와 동일 유지
- `patch.enabled: false`로 두어 PatchITransformer로 가지 않게 함(Factory에서 `variant=="M0"`로 직접 선택)
- M0 전용 설정은 `multislot:` 네임스페이스로 분리

예시 키(초안):
```yaml
variant: M0
d_model: 512
n_heads: 8
e_layers: 2
d_ff: 2048
dropout: 0.1
activation: gelu
use_norm: true
meta:
  enabled: false
  mode: none
patch:
  enabled: false
  patch_len: 0
  mode: none
  local_win: 0
  use_pos_emb: false

multislot:
  scales: [8, 32, ${data.seq_len}]
  k_per_scale: [2, 1, 1]   # scales와 1:1 매칭
  pad_value: 0.0           # right-pad 값
  temporal_conv:
    kernel_size: 3
    layers: 2
    dilation: 1
  pma:
    n_heads: 8
    ffn: true
  slot_fuse:
    hidden: ${model.d_ff}
```

추가 설계 메모:
- `scales`에 `${data.seq_len}`을 포함하면 dataset에 따라 L이 자동 반영됨.
- `k_per_scale`은 `scales`와 같은 길이로 맞추고, slot concat 순서는 리스트 순서로 고정.

---

### Step B) Model code: 새 파일 추가
목표: M0 모델을 `src/itransformer/models` 아래에 추가하고, 기존 모델과 동일한 forward 계약을 지킴.

권장 파일/클래스 구분(1파일에 같이 두어도 무방):
- 파일: `src/itransformer/models/m0.py` (가칭)
- 주요 클래스:
  1) `MultiScaleSlotEmbedding`
     - 입력: `x_enc [B,L,N]`, `x_mark [B,L,C]`
     - 내부:
       - concat → inversion → `x_inverted [B,T,L]` (`T=N+C`)
       - scale별 right-pad → patchify(non-overlap) → proj(p→d) → add pos → causal conv(p별) → PMA slotize(K_p) → add scale-id
       - 출력: `Z [B,T,K_total,d]`
  2) `PMA`(slotizer)
     - Q=seeds(K_p,d), K/V=patch tokens(m_p,d)
     - multi-head attention(head=8) + (옵션 확정) FFN+residual
  3) `CausalConvBlock1D`(scale별)
     - shape 유지: (B,T,m_p,d)에서 m_p 축 conv
     - 구현은 `B*T`를 batch로 합쳐 conv 적용(토큰 간 mixing 방지)
  4) `SlotFuseMLP`
     - 입력: `U [B,T,K_total,d]`
     - `concat(K_total*d)` → MLP → `fused [B,T,d]`
  5) `ITransformerM0` (또는 `ITransformerMultiSlot`)
     - embedding 교체 + encoder K회 반복(가중치 공유) + projector 동일

성능/구현 포인트(중요):
- 문서 수도코드처럼 토큰 t를 파이썬 for-loop로 돌리면 매우 느릴 수 있으므로,
  **scale 단위로 벡터화**해서 구현한다.
  - 예: `x_inverted [B,T,L]`를 scale별 pad 후 `[B,T,m_p,p]`로 reshape
  - `patch_proj[p]`는 마지막 차원(p)에만 linear 적용
  - conv/PMA는 `B*T`를 batch로 합쳐서 처리

---

### Step C) 모델 등록: `__init__.py`, `factory.py`
목표: `model=M0` 설정만으로 훈련/평가 파이프라인에서 자동 생성.

수정 대상:
- `src/itransformer/models/__init__.py`
  - `from itransformer.models.m0 import ITransformerM0` 추가
  - `__all__`에 추가
- `src/itransformer/models/factory.py`
  - `if cfg.model.variant == "M0": return ITransformerM0(cfg)` 분기 추가
  - 기존 P0/P1~P4 분기와 충돌하지 않도록 우선순위는 variant 우선으로 둠

---

### Step D) 실험 플랜(Plan YAML) 추가
목표: orchestrator로 “baseline(P0) vs M0”를 동일 조건에서 반복 실행.

권장 플랜(초안):
1) Smoke test (1~2 epoch, 작은 batch)로 forward/shape/학습 루프 검증
2) 본 실험: ETTh1/Exchange에서 supervised scratch로 비교

예: `conf/plan/M0_etth1_exchange.yaml` (가칭)
- baseline: `SUP-ETTh1.P0...`, `SUP-Exchange.P0...`
- 비교군: `SUP-ETTh1.M0...`, `SUP-Exchange.M0...`
- 공통: `train.ssl_ckpt_path=null` (scratch), `train.mode=ft`, `freeze_epochs=0`
- `metadata.enabled=false` 고정(1차)

---

### Step E) 최소 검증 체크리스트(구현 후)
목표: “틀렸는데 돌아가는” 상황을 초기에 차단.

1) Shape assertion
- `x_inverted: [B,T,L]`
- 각 scale p:
  - `L_pad = ceil(L/p)*p`
  - `patches: [B,T,m_p,p]`
  - `E/F: [B,T,m_p,d]`
  - `slots_p: [B,T,K_p,d]`
- `Z: [B,T,K_total,d]`, `U: [B,T,K_total,d]`, `fused: [B,T,d]`
- 최종: `[B,pred_len,N]`

2) “변수 간 mixing 금지” 규칙 확인
- embedding 단계에서 `B*T`를 batch로 합쳐 conv/PMA 적용(토큰 간 mixing 방지)
- encoder에 들어갈 때만 `[B,T,d]`에서 T축 self-attn(변수 간 mixing) 허용

3) 성능/메모리
- K_total=4로 encoder가 4번 호출되므로 속도 저하가 정상.
- 로깅에 `params_count`, `gpu_mem_peak` 확인(기존과 동일 포맷 유지 가능)

---

## 4) 리스크/주의점(1차 실험 해석)

1) right-pad(0) + mask 없음
- 패딩 구간이 “실제 데이터”처럼 처리되어 학습에 영향을 줄 수 있음.
- 1차는 단순성 우선이므로 감수하되, 결과 해석 시 “padding 영향” 가능성을 명시.

2) covariates(x_mark) zero pad
- 시간 특징이 0으로 pad되는 것은 의미적으로 어색할 수 있음.
- 1차는 baseline 조건을 맞추기 위해 동일 경로로 처리(문서 합의).

---

## 5) 구현 순서 제안(체크포인트 포함)

1) `conf/model/M0.yaml` 추가 (Hydra로 `model=M0` 선택 가능 확인)
2) `src/itransformer/models/m0.py` 추가 (forward shape만 먼저 통과)
3) `factory.py`/`__init__.py` 등록
4) `downstream.py`로 1 epoch smoke run (ETTh1, seed=0)
5) plan yaml 추가 후 orchestrator로 다중 런 실행

