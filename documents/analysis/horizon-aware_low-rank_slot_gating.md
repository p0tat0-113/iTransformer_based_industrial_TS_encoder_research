## 0) 왜 현재 `score_mlp`가 이렇게 커졌는지 (핵심 진단)

너의 `score_mlp([u_L, u_i, u_i-u_L])`는 사실상 **"(3d → 2048)"**의 거대한 선형층(혹은 그에 준하는)으로 보입니다.

* 라면 입력 차원 = 
*  파라미터
* 네가 보고한 `slot_fuse` 과 거의 일치

이게 큰 이유는 보통 둘 중 하나(혹은 둘 다)입니다.

1. **가 horizon에 따라 바뀌지 않는다**
* 현재는 $\alpha_{i}$가 슬롯별 스칼라(혹은 슬롯별 1개)이고, horizon 조절은  하나로만 함
* 즉, '짧은 horizon은 local 슬롯을, 긴 horizon은 global 슬롯을' 같은 horizon-조건부 선택을 구조적으로 못 함
* 이 부족한 표현력을 `score_mlp`가 억지로 보상하려고 키웠을 가능성이 큼


2. **gating이 latent(u)만 보고 결정한다**
* 실제로 "어떤 슬롯의 예측에 baseline을 얼마나/어떤 방향으로 고쳐야 하는지"는  (output 공간)가 더 직접적인 정보인데
* 만으로 그걸 맞추려면 복잡한 비선형 매핑이 필요해짐 → 큰 MLP로 수렴



따라서 "MLP를 줄이는" 방향이 아니라, **gating이 가져야 할 자유도(특히 horizon 조건부 선택)를 작은 파라미터로 제공**하는 쪽이 맞습니다.

---

## 1) (최우선 추천) Horizon-aware Low-Rank Slot Gating (HLSG)

**Horizon-aware Low-Rank Slot Gating (HLSG)** = "슬롯-가중치()를 horizon별로 만들되, '낮은 랭크(r)'로 factorize해서 파라미터를 줄이는 게이팅 방식"

* **Relation to 네 기존 slot_fuse:** replacement. 구체적으로, 기존의 `score_mlp` + `scalar a_i`를 low-rank로 만든 $\alpha_{i,h}$로 대체
* **Rationale:** 표현력(=horizon별 슬롯 선택)을 늘리면서도, 파라미터는  같은 거대 행렬 대신 $(3d \to r) + (r \to H)$로 쪼개서 크게 줄일 수 있음

### 핵심 아이디어

* 'horizon마다 어떤 scale/slot을 쓸지'는 매우 중요한데, 이걸 현재는  하나로 뭉뚱그려 처리합니다.
* HLSG는 를 **horizon별로** 만들되,  전체를 직접 예측하지 않고 **저차원 basis**로 만듭니다.

### 형태 (가정: baseline 제외 extras 개수 = E, horizon 길이 = H)

* baseline slot embedding: 
* extra slot embedding:  ()
* baseline prediction: 
* extra prediction: 
* residual: 

**(1) 슬롯별 게이팅 코드 생성 (저차원 r)**


* $f = $ 작은 MLP 또는 단층 Linear (권장 시작: **Linear(3d→r) + GELU**)
* LN = LayerNorm

**(2) horizon embedding / basis로 horizon별 logit 생성**


*  = 학습되는 horizon embedding ()

**(3) horizon별 softmax**


*  (초기에는 1~2 권장)

**(4) horizon별 보정**


### 기호 정의 (이 블록에서)

*  = slot embedding dim (너는 512)
*  = pred horizon length (너는 96)
*  = extra slot 수 (너는 5)
*  = low-rank 차원 (예: 16/32/64)
*  = concat
* LN = LayerNorm
*  = 번째 extra의 시점 residual
*  = 시점에서 번째 extra에 주는 가중치

### 파라미터가 얼마나 줄어드나?

* 기존: 대략 
* HLSG (단층 ):
* 
* horizon embedding 
* 합 = **52k** (bias 포함해도 비슷)


* **즉, 3.15M → ~0.05M 수준.**

### 왜 성능이 유지/개선될 가능성이 큰가?

* 네 모델이 워낙 의도한 '전역/중간/국소'는 **horizon에 따라 중요도가 달라지는 경우가 많습니다.**
* 기존은  하나로만 horizon 반응성을 주는데, 이건 슬롯 선택을 바꾸지 못합니다.
* HLSG는 **슬롯 선택 자체를 horizon별로 하게** 만들어서, 작은 파라미터로도 더 맞는 inductive bias를 줍니다.

### 안정화 팁 (성능 유지에 중요)

* ** smoothness regularization:** horizon에 따라 가 너무 요동치지 않게


* 시점의 슬롯 가중치 벡터


* **entropy floor/temperature:** 초반 학습에 한 슬롯으로 붕괴하는 걸 막기 위해 를 1~2로 두고 필요하면 낮추기


---

### **HLSG 수식(코드와 1:1 대응)**

*  = baseline(전역, p=L) 슬롯의 latent embedding
*  = i번째 extra 슬롯의 latent embedding
*  = Linear(3d  r) (d= `d_model`, r= `hls_rank`)
*  = h번째 horizon의 embedding (shape: `pred_len x r`)
*  = `hls_rank` (Low-rank dimension)
*  = `hls_tau` (temperature)
*  = horizon h에서 extra slot i가 baseline을 얼마나 보정할지(가중치)

---

## 내가 코드에 반영한 “HLSG 아이디어 수정점”

HLSG를 그냥 넣으면 초기 학습 초기에 slot이 랜덤하게 섞이면서 불안정해질 수 있어서, 기존 네 `horizon_alpha_mlp` 처럼 **초기엔 거의 uniform alpha로 시작하게** 만들었어.

* `HorizonLowRankScorer.in_proj` 를 **0으로 초기화** → 초기  →  → 
* 대신 `h_emb` 는 작은 랜덤으로 시작 → gradient는 `h_emb` 를 통해 `in_proj` 로 흘러서 학습이 막히지 않게 구성

이건 네가 관찰한 “짧은 horizon에선 local slot이 유익하지만, 긴 horizon에선 노이즈”라는 현상에서, 초반부터 모델이 local에 과하게 의존하는 쪽으로 깨지는 걸 줄이려는 안정화 장치야.