이 문서는 **네가 이미 가지고 있는 “순정 iTransformer” 코드베이스**를 최대한 그대로 유지하면서,
우리가 설계한 **Multi-scale Latent Slots (spike-focused)** 모듈만 “끼워 넣는 방식”으로 수도코드를 작성한 거야.

핵심 의도:

* 기존 iTransformer의 큰 틀(Preprocess/Inversion → Embedding → Encoder → Projection)을 유지한다.
* 다만 Embedding을 “멀티스케일 patch → temporal encoder → slotize”로 바꾸고,
* Encoder는 slot index k를 고정한 채 **K번 반복 호출**(가중치 공유)한다.
* 그 후 slot들을 fuse해서 기존 projector(Linear d_model→pred_len)에 다시 연결한다.

---

## -1) 모델 철학(구현자가 먼저 알아야 하는 전제)

### -1.1 iTransformer가 원래 하려던 역할 분담

* iTransformer의 핵심은 “토큰을 시간축이 아니라 **변수축(variate)**에 둔다”는 것.
* 그래서 Transformer의 self-attention은 본질적으로 **변수 간 상호작용(상관관계)**을 학습하는 데 쓰이고,
* 시간적(temporal) 정보는 토큰을 만드는 단계(embedding/projection)와 단순한 head(Linear)에서 처리되는 구조가 된다.

### -1.2 우리가 추가로 강제하려는 원칙(이 프로젝트의 핵심)

산업 센서 데이터에서는 센서별 지연/비동기/리샘플링 차이 때문에,
"같은 시점 t"끼리 변수를 직접 섞으면 alignment noise를 함께 학습할 위험이 있다.

그래서 우리는 아래를 **구조적으로 강제**한다.

1. **변수 간 mixing은 오직 variate-attention에서만** 일어난다.

   * time-axis token(패치 인덱스)이 변수를 가로질러 attention에 들어가는 것을 금지한다.
2. **시간적 패턴(특히 spike)은 변수 내부에서만** 다룬다.

   * patchify / conv 같은 모듈은 각 variate(또는 covariate token) 내부에서만 적용.
3. **spike-focused fine-grained 표현을 확보하되**, iTransformer 철학을 깨지 않는다.

   * 변수당 토큰 1개는 spike가 평균화되기 쉬워서,
   * 변수당 K개의 “latent slot(view)”로 분해하되,
   * attention은 여전히 “변수 축”에서만 수행한다.

### -1.3 이 설계가 만들어내는 효과(직관)

* 멀티스케일/slotize는 “변수 내부에서 spike를 놓치지 않는 표현”을 만들고,
* variate-attention은 “변수 간 상관 구조”만 학습한다.
* 결과적으로 lag/비동기에서 오는 time-alignment noise 경로를 줄이면서,
  spike 같은 고주파 이벤트를 더 선명하게 모델이 볼 수 있게 한다.

---

## -2) 구성요소별 역할(왜 넣었는지, 무엇을 하면 철학이 깨지는지)

### (A) Inversion + covariates concat (순정과 동일)

* 역할: (변수 + 시간 covariates)를 모두 **토큰 후보**로 만들기 위한 전처리.
* 왜 유지?: 기존 코드/실험과 최대한 동일한 조건에서 비교하려고.
* 주의: covariates도 encoder에서 같이 섞이므로 embedding 단계도 동일 규칙으로 처리해야 일관적임.

### (B) Multi-scale Patchify

* 역할: lookback 시계열을 여러 창 크기(p)로 잘라서 토큰 시퀀스를 만든다.
* 왜 필요?: spike는 짧은 구간에 몰리므로 작은 p에서 더 잘 드러나고,
  큰 p(L 포함)는 레벨/컨텍스트(저주파)를 준다.
* 주의: patch 토큰을 "그대로" variate-attention에 넣으면 철학이 깨진다(시간축 토큰 mixing).

### (C) Patch projection (p → d_model)

* 역할: patch(길이 p)를 d_model 차원의 token으로 변환.
* p=L & m_p=1이면 Linear(L→d_model) 한 번이므로, 순정 iTransformer embedding과 동일한 의미를 가진다.

### (D) Patch positional embedding (변수 내부)

* 역할: patch 토큰에 "몇 번째 patch인지"(순서 단서)를 주입.
* 왜 필요?: slotizer(PMA)는 set-like pooling 성격이 있어 순서를 직접 쓰기 어렵다.
  따라서 순서 단서를 token 내용에 미리 넣어야 한다.

### (E) Causal inter-patch Conv temporal encoder (변수 내부)

* 역할: patch 인덱스 축(m_p)에서 로컬 맥락(스파이크 전후, 회복)을 token에 주입.
* 왜 causal?: forecasting에서 미래 patch를 보면 정보 누수(leakage) 위험.
* 구현 팁: p=L처럼 m_p=1인 경우 inter-patch conv는 의미가 없으므로 identity 처리.
* 주의: Conv/TCN은 **압축이 아니라 feature extractor**로 쓰는 게 기본(길이 유지).

### (F) PMA slotizer (변수 내부 bottleneck)

* 역할: m_p개의 patch token을 K_p개의 slot으로 압축.
* 왜 필요?:

  * 토큰 수 폭발(m_p가 큼)을 막고,
  * patch token을 그대로 variate-attn에 넣지 않기 위해 "변수 내부"에서 bottleneck을 만든다.
* 주의: PMA의 K>1 slot 분화는 자동 보장이 약하므로,
  2차 실험부터 diversity loss를 추가하는 것을 권장(별도 문서 참고).

### (G) Scale-id embedding

* 역할: slot이 어떤 스케일(p)에서 왔는지 명시.
* 왜 필요?: 멀티스케일 concat만 하면 모델이 스케일을 추론해야 해서 학습이 흔들릴 수 있음.

### (H) Encoder를 K번 반복 호출 (가중치 공유)

* 역할: "k번째 view"에서 변수들 간 상호작용을 학습.
* 왜 공유?: 파라미터 수를 K배로 늘리지 않고, view만 늘리기 위함.
* 직관: k=1은 “짧은 스케일의 spike-view”, k=K는 “전역 컨텍스트-view”처럼,
  view별로 다른 상관 구조를 보게 만든다.
* 주의: 여기서 encoder 입력의 토큰 수는 항상 T=N+Covariates 뿐이다(시간축 토큰 금지).

### (I) Slot fuse (변수 내부)

* 역할: K개 slot 출력을 하나의 d_model 벡터로 합쳐서, 순정 projector에 연결.
* 왜 MLP?: attention이 아니라 MLP/gating으로 fuse하면 "변수 간 mixing" 경로를 만들지 않는다.

### (J) Projector (순정과 동일)

* 역할: d_model → pred_len 복원.
* 왜 유지?: 기존 baseline과 공정 비교.

---

## 0) 순정 iTransformer와의 대응 관계(구현자 혼란 방지)

순정 iTransformer:

* Embedding: Linear(seq_len → d_model)
* Encoder: 1회
* Projector: Linear(d_model → pred_len)

우리 모델은 다음처럼 “특수 케이스 포함” 관계를 가진다:

* scales={L} (p=L만 사용), K_L=1, TemporalEnc=Identity, Slotize=Identity
  → **정확히 순정 iTransformer와 동일**

즉, 우리는 iTransformer를 “일반화”한 형태로 구현할 수 있다.

---

## 1) 실험 #1에서 고정할 iTransformer 기본 하이퍼파라미터(그대로 사용)

```text
--d_model  512
--n_heads  8
--e_layers 2
--d_ff     2048
--dropout  0.1
```

추가로 필요한(우리 모듈 전용) 하이퍼파라미터(실험 #1 권장 기본값):

```text
--scales {8, 32, L}              # 멀티스케일 patch size
--K_8    2
--K_32   1
--K_L    1                        # p=L은 순정 embedding의 일반화
--K_total = 4

--temporal_enc causal inter-patch Conv
   kernel_size = 3
   n_conv_layers = 2
   dilation = 1
```

---

## 2) 전체 구조 개요(순정과 동일한 흐름 유지)

* (1) Pre-processing (Inversion): 기존과 동일
* (2) Embedding: **기존 Linear(seq_len→d_model)** 대신

  * 멀티스케일 patchify → proj → (pos) → causal conv → PMA slotize
  * 결과: (B, TokenCount, K_total, d_model)
* (3) Encoder: **기존 encoder를 K_total번 반복 호출(가중치 공유)**

  * k 고정, n(변수)만 섞는 variate-only attention
* (4) Projection: 기존 projector를 그대로 사용

  * 단, projector 전에 slot fuse로 (B, TokenCount, d_model)로 복구

---

## 3) 수도코드

### 3.1 보조 함수/모듈(구현자가 바로 코드로 내리기 쉬운 단위)

```text
function PATCHIFY_1D(x: (B, L), p: int) -> (B, m_p, p)
    # non-overlap patch (stride=p)
    # L이 p로 안 나눠떨어지면: padding/trim 정책이 필요함
    m_p = L / p
    return reshape(x, (B, m_p, p))

function ADD_PATCH_POS(E: (B, m_p, d), p: int) -> (B, m_p, d)
    # patch index positional embedding (변수 내부)
    pos = LearnedPos[p][0:m_p]           # (m_p, d)
    return E + pos

function CAUSAL_CONV_BLOCK(E: (B, m_p, d)) -> (B, m_p, d)
    # 목적: (패치 인덱스 축 m_p)에서 local spike 전후 맥락을 토큰에 주입
    # forecasting 누수 방지 위해 causal padding(왼쪽 패딩) 사용

    if m_p == 1:
        # p=L 같은 경우: patch가 1개라 inter-patch conv 의미가 없음
        return E

    Y = LN(E)
    Y = CausalConv1D(Y, kernel=3, dilation=1)   # (B, m_p, d)
    Y = GELU(Y)
    Y = Dropout(Y)
    Y = CausalConv1D(Y, kernel=3, dilation=1)   # (B, m_p, d)
    Y = Dropout(Y)
    return E + Y

function PMA_SLOTIZE(F: (B, m_p, d), Seeds: (K_p, d)) -> (B, K_p, d)
    # learnable query pooling (변수 내부)
    Z = MHA(Q=Seeds, K=F, V=F)            # (B, K_p, d)
    Z = Z + FFN(LN(Z))                   # 안정화(선택이지만 권장)
    return Z

function ADD_SCALE_ID(Z: (B, K_p, d), p: int) -> (B, K_p, d)
    return Z + ScaleEmbedding[p]         # (d) broadcast

function VARIATE_ENCODER(tokens: (B, T, d)) -> (B, T, d)
    # 순정 iTransformer encoder와 동일한 TransformerEncoder
    # 여기서 T = N + Covariates
    return TransformerEncoder(tokens)

function FUSE_SLOTS(U: (B, T, K, d)) -> (B, T, d)
    # slot들을 하나로 합쳐서 순정 projector에 넣을 수 있게 만든다.
    # 첫 실험은 concat + MLP

    H = concat(U along K)                # (B, T, K*d)
    H = Linear(K*d -> d)(H)
    H = GELU(H)
    H = Dropout(H)
    H = Linear(d -> d)(H)
    return H
```

---

### 3.2 메인 클래스: 순정 iTransformer를 “부분 수정”한 형태

```text
class iTransformer_MultiSlot(Model):
    """
    iTransformer × Multi-scale Latent Slots (spike-focused)

    기존 iTransformer와 최대한 동일하게 구성:
    - projector는 그대로
    - encoder도 그대로(단, K번 반복 호출)
    - embedding만 MultiSlotEmbedding으로 교체
    """

    def __init__(self, config):
        # ===== (순정 iTransformer와 동일) 기본 hyperparams =====
        self.d_model  = config.d_model     # 512
        self.n_heads  = config.n_heads     # 8
        self.e_layers = config.e_layers    # 2
        self.d_ff     = config.d_ff        # 2048
        self.dropout  = config.dropout     # 0.1

        # ===== (우리 추가) 멀티스케일/슬롯 설정 =====
        self.scales = [8, 32, config.seq_len]          # {8,32,L}
        self.K_per_scale = {8:2, 32:1, config.seq_len:1}
        self.K_total = sum(self.K_per_scale[p] for p in self.scales)   # 4

        # ===== (우리 추가) 스케일별 patch projection =====
        # p=L이면 Linear(L → d_model)이라서, 순정 embedding과 사실상 같은 역할
        for p in self.scales:
            self.patch_proj[p] = Linear(in_features=p, out_features=self.d_model)
            self.pos_emb[p]    = LearnedPosEmbedding(max_len=config.seq_len // p, d=self.d_model)
            self.scale_emb[p]  = Parameter(shape=(self.d_model,))
            self.seeds[p]      = Parameter(shape=(self.K_per_scale[p], self.d_model))

        # ===== (우리 추가) temporal encoder: causal inter-patch conv =====
        self.temporal_enc = CausalConvBlock(d_model=self.d_model, kernel=3, layers=2)

        # ===== (순정 iTransformer와 동일) encoder / projector =====
        self.encoder = TransformerEncoder(
            layers=self.e_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout
        )
        self.projector = Linear(in_features=self.d_model, out_features=config.pred_len)

        # ===== (우리 추가) slot fuse =====
        self.slot_fuse = MLP(in_features=self.K_total * self.d_model,
                             hidden=self.d_ff,
                             out_features=self.d_model,
                             dropout=self.dropout)


    def forward(self, x, x_mark):
        """
        Input:
            x:      (B, L, N)
            x_mark: (B, L, C)   # covariates 개수 = C
        Output:
            y_hat:  (B, pred_len, N)

        주의:
        - 순정과 동일하게 x와 x_mark를 variate 축으로 concat해서 token으로 만든다.
        - covariates 토큰도 "변수 토큰"처럼 다룬다(encoder에 같이 들어가야 하므로).
        """

        # === 1) Pre-processing (Inversion) : 순정과 동일 ===
        x_combined  = concat([x, x_mark], axis=Variate)       # (B, L, N+C)
        x_inverted  = transpose(x_combined, (B, N+C, L))      # (B, T, L) where T=N+C

        B, T, L = shape(x_inverted)

        # === 2) MultiSlot Embedding (순정 embedding 대체) ===
        # 목표: (B, T, L) → (B, T, K_total, d_model)

        Z = zeros((B, T, self.K_total, self.d_model))

        for t in 0..T-1:
            # token 하나(=변수 또는 covariate)의 lookback 전체
            seq = x_inverted[:, t, :]                       # (B, L)

            slots_list = []

            for p in self.scales:
                # (A) patchify
                P = PATCHIFY_1D(seq, p)                     # (B, m_p, p)

                # (A) patch projection
                E = self.patch_proj[p](P)                   # (B, m_p, d)

                # (A) positional embedding (변수 내부)
                E = ADD_PATCH_POS(E, p)                     # (B, m_p, d)

                # (B) temporal encoder (변수 내부), causal
                F = self.temporal_enc(E)                    # (B, m_p, d)

                # (C) slotize by PMA
                Zp = PMA_SLOTIZE(F, self.seeds[p])          # (B, K_p, d)

                # (C) add scale id
                Zp = ADD_SCALE_ID(Zp, p)                    # (B, K_p, d)

                slots_list.append(Zp)

            # (D) concat scales → token t의 K_total slots
            Z_t = concat(slots_list, axis=Slot)             # (B, K_total, d)

            Z[:, t, :, :] = Z_t

        # === 3) Encoder: k 고정, variate 축만 attention (K_total번 반복) ===
        # 순정은 encoder를 1번 호출하지만, 우리는 view(slot)마다 1번씩 호출
        # 중요: encoder 가중치는 공유(같은 self.encoder를 반복 사용)

        U = zeros_like(Z)                                   # (B, T, K_total, d)

        for k in 0..K_total-1:
            tokens_k = Z[:, :, k, :]                        # (B, T, d)

            # (순정과 동일) Transformer encoder
            enc_k = self.encoder(tokens_k)                  # (B, T, d)

            U[:, :, k, :] = enc_k

        # === 4) Slot Fuse: (B, T, K, d) → (B, T, d) ===
        fused = FUSE_SLOTS(U)                               # (B, T, d)

        # === 5) Prediction Head: 순정 projector 그대로 ===
        dec_out = self.projector(fused)                     # (B, T, pred_len)

        # covariates 토큰 제거 및 원상복구
        dec_var = dec_out[:, 0:N, :]                        # (B, N, pred_len)
        y_hat  = transpose(dec_var, (B, pred_len, N))       # (B, pred_len, N)

        return y_hat
```

---

## 4) 구현 시 특히 헷갈리기 쉬운 포인트(명시)

1. **순정 embedding과의 정확한 연결고리**

* p=L 스케일에서 patchify하면 m_p=1이고 patch 길이가 L이다.
* patch_proj[L] = Linear(L→d_model) 이므로, 이 경로는 순정 embedding과 동일한 역할을 한다.

2. **TemporalEnc(Conv/TCN)는 압축이 아니다**

* stride=1, pooling 없음
* 입력/출력 shape이 (B, m_p, d)로 유지되어야 한다.

3. **Encoder는 K번 호출하지만 가중치는 공유**

* 모델 파라미터 수가 K배로 늘지 않게 한다.
* 계산량만 K배(대신 K는 작게 잡음).

4. **x_mark(covariates)도 token으로 같이 처리**

* 순정 iTransformer처럼 encoder에서 covariates 토큰과 variate 토큰이 같이 상호작용한다.
* 따라서 embedding 단계에서도 covariates 토큰을 동일하게 처리해주는 것이 구현상 단순하다.

---

## 5) (다음 실험) 최소 수정으로 파생 가능한 버전

* Perceiver-resampler slotizer: PMA_SLOTIZE 부분만 교체
* TokenLearner slotizer: PMA_SLOTIZE 부분만 교체
* diversity loss ON: 학습 loss에 L_div 항 추가

이 문서의 구조를 그대로 두고, 모듈 함수만 바꾸면 된다.
