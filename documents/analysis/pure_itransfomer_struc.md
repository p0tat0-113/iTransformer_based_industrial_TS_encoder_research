# Pure iTransformer Structure (P0)

아래 수도코드는 현재 코드베이스의 P0(iTransformer) 핵심 아이디어(시간축을 임베딩하고, 변수축을 토큰으로 사용하는 “inverted” 방식)를 충분히 추상화해서 잘 표현하고 있습니다.

## Hyperparameters (예시)

```text
--d_model    512 (or 256)     토큰 임베딩 차원 크기
--n_heads    8                Multi-head Attention 헤드 개수
--e_layers   2                인코더 레이어(Block) 수
--d_ff       2048 (or 512)    FeedForward Network 내부 차원
--dropout    0.1              드롭아웃 비율
```

## Pseudocode

```python
class iTransformer(Model):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting
    핵심 아이디어: 시간 축(Time)을 임베딩하고, 변수 축(Variate)을 토큰으로 사용한다.
    """
    def __init__(self, config):
        # 1. Inverted Embedding Layer
        #    - 시간 흐름(seq_len) 전체를 하나의 벡터(d_model)로 압축
        self.embedding = Linear(in_features=config.seq_len, out_features=config.d_model)

        # 2. Standard Transformer Encoder
        #    - 변수(Variate)들 간의 상관관계(Correlation)를 학습
        self.encoder = TransformerEncoder(
            layers=config.e_layers,
            d_model=config.d_model,
            n_heads=config.n_heads
        )

        # 3. Prediction Head (Projector)
        #    - 임베딩된 벡터(d_model)를 미래 시계열(pred_len)로 복원
        self.projector = Linear(in_features=config.d_model, out_features=config.pred_len)

    def forward(self, x, x_mark):
        """
        Input:
            x: [Batch, Time(L), Variate(N)]       # 일반적인 시계열 데이터 형태
            x_mark: [Batch, Time(L), Covariates]  # 시간 정보 (시, 분, 요일 등)
        """

        # === 1. Pre-processing (Inversion) ===
        # 변수(Variate)와 시간 정보(Covariates)를 모두 토큰으로 취급하기 위해 합침
        # Shape: [Batch, Time(L), N + Covariates]
        x_combined = concatenate([x, x_mark], axis=Variate)

        # 시간 축과 변수 축을 뒤집음 (Transpose)
        # Shape: [Batch, N + Covariates, Time(L)]
        x_inverted = x_combined.transpose(Time, Variate)

        # === 2. Embedding ===
        # 전체 길이 L의 시계열을 d_model 크기의 벡터 하나로 투영
        # Shape: [Batch, N + Covariates, d_model]
        tokens = self.embedding(x_inverted)

        # === 3. Encoder (Multivariate Correlation) ===
        # 각 변수 토큰들이 서로 Attention을 수행하며 관계를 학습
        # Shape: [Batch, N + Covariates, d_model]
        enc_out = self.encoder(tokens)

        # === 4. Prediction ===
        # 학습된 표현을 다시 미래 시계열 값으로 디코딩
        # Shape: [Batch, N + Covariates, pred_len]
        dec_out = self.projector(enc_out)

        # 시간 정보(Covariates) 토큰은 예측 대상이 아니므로 버림
        # 순서를 다시 [Batch, Time, Variate]로 원상복구
        # Shape: [Batch, pred_len, N]
        final_output = dec_out[:, :N, :].transpose(Time, Variate)

        return final_output
```

## Codebase Mapping (참고)

- 모델 정의: `src/itransformer/models/itransformer.py`
- inverted embedding 구현: `src/itransformer/models/layers/embed.py` (`DataEmbeddingInverted`)

## Notes (구현 디테일 차이)

- 실제 구현은 `use_norm`이 켜져 있으면 입력을 정규화한 뒤(평균/분산) 예측 후 다시 복원합니다.
- 실제 구현은 `x_mark`가 없는 경우도 지원합니다.
