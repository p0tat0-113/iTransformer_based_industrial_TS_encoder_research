아래는 네 `new_model` 브랜치 코드베이스에 **PatchTST / Informer / DLinear / TiDE**를 **pure implementation**으로 통합하기 위한 “실제로 붙여 넣어 바로 돌아가게 만드는” 수준의 구체 패치안이야. (가이드의 원칙/파일 구조/인터페이스 규약을 그대로 따름) 
또한 Informer/TiDE는 **미래 time mark(y_mark_dec)**를 쓰는 것이 표준이라, **downstream.py / eval.py에 최소 변경 2곳**이 필요하다(아래에 패치 포함). 이는 현재 downstream/eval이 y_mark_dec를 TSLIB에만 넘기도록 되어 있기 때문. ([GitHub][1])

---

## 0) 추가 파일 요약

가이드에 맞춰 “모델 1개당 최소 4곳”을 채운다.

* `src/itransformer/models/patchtst.py`
* `src/itransformer/models/informer.py`
* `src/itransformer/models/dlinear.py`
* `src/itransformer/models/tide.py`

그리고 연결:

* `src/itransformer/models/__init__.py` 수정
* `src/itransformer/models/factory.py` 수정
* `conf/model/PatchTST.yaml` 추가
* `conf/model/Informer.yaml` 추가
* `conf/model/DLinear.yaml` 추가
* `conf/model/TiDE.yaml` 추가

추가로(권장, 사실상 필요):

* `src/itransformer/downstream.py`에서 Informer/TiDE에 y_mark_dec 전달
* `src/itransformer/eval.py`에서 Informer/TiDE에 y_mark_dec 전달

---

## 1) 모델 코드 추가

### 1-1) `src/itransformer/models/dlinear.py`

DLinear은 LTSF-Linear 계열 구현(이동평균 분해 + seasonal/trend 선형) 형태를 따라가되, 이 코드베이스의 `use_norm` 규약을 동일 적용. ([GitHub][2])

```python
# src/itransformer/models/dlinear.py
import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1 (got {self.kernel_size})")
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self.kernel_size == 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        tail = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, tail], dim=1)  # [B, L+2p, C]
        y = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, C]
        return y


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.mavg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        trend = self.mavg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.pred_len = int(cfg.data.pred_len)
        self.n_vars = int(cfg.data.enc_in)

        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        dcfg = getattr(cfg.model, "dlinear", None)
        self.kernel_size = int(getattr(dcfg, "kernel_size", 25) or 25)
        self.individual = bool(getattr(dcfg, "individual", True))

        self.decomp = SeriesDecomposition(self.kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.n_vars)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.n_vars)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        # x_enc: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc

        seasonal, trend = self.decomp(x)  # [B, L, N]
        seasonal = seasonal.permute(0, 2, 1)  # [B, N, L]
        trend = trend.permute(0, 2, 1)

        if self.individual:
            out_s = []
            out_t = []
            for i in range(self.n_vars):
                out_s.append(self.linear_seasonal[i](seasonal[:, i, :]).unsqueeze(1))  # [B,1,H]
                out_t.append(self.linear_trend[i](trend[:, i, :]).unsqueeze(1))
            out_s = torch.cat(out_s, dim=1)  # [B, N, H]
            out_t = torch.cat(out_t, dim=1)
        else:
            out_s = self.linear_seasonal(seasonal)  # [B, N, H]
            out_t = self.linear_trend(trend)

        y = (out_s + out_t).permute(0, 2, 1)  # [B, H, N]

        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return y, None

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, **kwargs)
        return y
```

---

### 1-2) `src/itransformer/models/patchtst.py`

PatchTST는 TSLib의 “PatchEmbedding + Transformer encoder + FlattenHead” 구조를 그대로 따라가되, 이 코드베이스의 Encoder/Attention 구현을 재사용한다. ([GitHub][3])

```python
# src/itransformer/models/patchtst.py
import math
import torch
import torch.nn as nn

from itransformer.models.layers.attention import FullAttention, AttentionLayer
from itransformer.models.layers.transformer import Encoder, EncoderLayer


class Transpose(nn.Module):
    def __init__(self, dim0, dim1, contiguous: bool = False):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.contiguous = contiguous

    def forward(self, x):
        y = x.transpose(self.dim0, self.dim1)
        return y.contiguous() if self.contiguous else y


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [1, L, D]
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(padding)
        self.pad = nn.ReplicationPad1d((0, self.padding))
        self.value_proj = nn.Linear(self.patch_len, d_model, bias=False)
        self.pos_emb = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, N, L]
        n_vars = x.size(1)
        x = self.pad(x)  # [B, N, L+padding]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, N, P, patch_len]
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3))  # [B*N, P, patch_len]
        x = self.value_proj(x) + self.pos_emb(x)  # [B*N, P, D]
        x = self.dropout(x)
        patch_num = x.size(1)
        return x, n_vars, patch_num


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float):
        super().__init__()
        self.n_vars = int(n_vars)
        self.flatten = nn.Flatten(start_dim=-2)  # (D, P) -> (D*P)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D, P]
        x = self.flatten(x)  # [B, N, D*P]
        x = self.linear(x)   # [B, N, H]
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.pred_len = int(cfg.data.pred_len)
        self.n_vars = int(cfg.data.enc_in)

        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        d_model = int(cfg.model.d_model)
        n_heads = int(cfg.model.n_heads)
        e_layers = int(cfg.model.e_layers)
        d_ff = int(cfg.model.d_ff)
        dropout = float(cfg.model.dropout)
        activation = str(getattr(cfg.model, "activation", "gelu"))

        pcfg = getattr(cfg.model, "patchtst", None)
        patch_len = int(getattr(pcfg, "patch_len", 16) or 16)
        stride = int(getattr(pcfg, "stride", 8) or 8)
        padding = int(getattr(pcfg, "padding", stride) or stride)
        head_dropout = float(getattr(pcfg, "head_dropout", dropout) or dropout)

        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        attn_layers = []
        for _ in range(e_layers):
            attn = AttentionLayer(
                FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
                d_model=d_model,
                n_heads=n_heads,
            )
            attn_layers.append(EncoderLayer(attn, d_model, d_ff=d_ff, dropout=dropout, activation=activation))

        norm_layer = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.encoder = Encoder(attn_layers, norm_layer=norm_layer)

        # patch_num = floor((L + padding - patch_len)/stride) + 1
        patch_num = (self.seq_len + self.padding - self.patch_len) // self.stride + 1
        self.head_nf = d_model * int(patch_num)
        self.head = FlattenHead(self.n_vars, self.head_nf, self.pred_len, head_dropout=head_dropout)

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        # x_enc: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc

        x = x.permute(0, 2, 1)  # [B, N, L]
        enc_in, n_vars, _ = self.patch_embedding(x)  # [B*N, P, D]
        enc_out, attns = self.encoder(enc_in)        # [B*N, P, D]

        B = x_enc.size(0)
        P = enc_out.size(1)
        D = enc_out.size(2)

        enc_out = enc_out.view(B, n_vars, P, D).permute(0, 1, 3, 2)  # [B, N, D, P]
        dec_out = self.head(enc_out).permute(0, 2, 1)                # [B, H, N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, **kwargs)
        return y
```

---

### 1-3) `src/itransformer/models/informer.py`

Informer는 encoder-decoder이며, 표준 구현은 **encoder=Prob/Full**, **decoder self-attn=Prob/Full(causal mask)**, **cross-attn=Full** 구성을 쓴다. ([GitHub][4])
또한 decoder의 입력 `x_dec`는 통상 “마지막 label_len + 0 padding(pred_len)”인데, **label part는 x_enc의 마지막 label_len과 동일**하므로(데이터 프로토콜상 overlap) 모델 내부에서 생성 가능. 그래서 downstream에서 **batch_y를 추가로 넘길 필요는 없다**. ([GitHub][1])

```python
# src/itransformer/models/informer.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itransformer.models.layers.attention import FullAttention, AttentionLayer


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device):
        # True = masked
        self.mask = torch.triu(torch.ones(B, 1, L, L, dtype=torch.bool, device=device), diagonal=1)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, L, D]
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        emb = nn.Embedding(c_in, d_model)
        emb.weight = nn.Parameter(w, requires_grad=False)
        self.emb = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    # expects x_mark order = [month, day, weekday, hour, minute?]
    def __init__(self, d_model: int, embed_type: str = "fixed", freq: str = "h"):
        super().__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding

        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

        self.freq = freq

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        x = x_mark.long()
        minute_x = self.minute_embed(x[:, :, 4]) if (self.freq == "t" and hasattr(self, "minute_embed")) else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, freq: str = "h"):
        super().__init__()
        # NOTE: timeenc=1(time_features)일 때 쓰는 경로. timeenc=0이면 TemporalEmbedding이 보통 더 안전.
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = int(freq_map.get(freq, 4))
        self.proj = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        return self.proj(x_mark)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, embed: str, freq: str, dropout: float):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        if embed == "timeF":
            self.temporal_embedding = TimeFeatureEmbedding(d_model, freq=freq)
        else:
            # embed in {"fixed","learned"}
            self.temporal_embedding = TemporalEmbedding(d_model, embed_type=embed, freq=freq)
        self.embed = embed
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        x_val = self.value_embedding(x)
        x_pos = self.position_embedding(x_val)
        if x_mark is None:
            return self.dropout(x_val + x_pos)
        x_tmp = self.temporal_embedding(x_mark)
        return self.dropout(x_val + x_tmp + x_pos)


class ConvDistillLayer(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=2, padding_mode="circular")
        self.norm = nn.BatchNorm1d(c_in)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        y = self.conv(x.transpose(1, 2))
        y = self.norm(y)
        y = self.act(y)
        y = self.pool(y)
        return y.transpose(1, 2)


class ProbAttention(nn.Module):
    """
    Informer의 probabilistic attention 구현.
    - 입력/출력 텐서 shape은 이 프로젝트의 AttentionLayer와 호환되게 유지:
      queries/keys/values: [B, L, H, E] -> out: [B, L, H, E]
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.mask_flag = bool(mask_flag)
        self.factor = int(factor)
        self.scale = scale
        self.output_attention = bool(output_attention)
        self.dropout = nn.Dropout(attention_dropout)

    def _sample_scores(self, Q, K):
        # Q: [B, H, Lq, E], K: [B, H, Lk, E]
        B, H, Lq, E = Q.shape
        _, _, Lk, _ = K.shape

        # sample_k, n_top
        sample_k = min(Lk, max(1, int(self.factor * math.log(Lk + 1))))
        n_top = min(Lq, max(1, int(self.factor * math.log(Lq + 1))))

        # shared per-layer sampling over keys for each query position (expand view + gather)
        index_sample = torch.randint(Lk, (Lq, sample_k), device=Q.device)  # [Lq, sample_k]
        K_expand = K.unsqueeze(2).expand(B, H, Lq, Lk, E)                  # view
        idx = index_sample.view(1, 1, Lq, sample_k, 1).expand(B, H, Lq, sample_k, E)
        K_sample = torch.gather(K_expand, dim=3, index=idx)                # [B,H,Lq,sample_k,E]

        # QK_sample: [B,H,Lq,sample_k]
        Q_ = Q.unsqueeze(-2)                                               # [B,H,Lq,1,E]
        scores = torch.matmul(Q_, K_sample.transpose(-2, -1)).squeeze(-2)  # [B,H,Lq,sample_k]

        # sparsity measure
        M = scores.max(dim=-1).values - scores.mean(dim=-1)                # [B,H,Lq]
        top_idx = M.topk(n_top, dim=-1, sorted=False).indices              # [B,H,n_top]
        return top_idx, n_top

    def forward(self, queries, keys, values, attn_mask=None):
        # queries/keys/values: [B, L, H, E]
        B, Lq, H, E = queries.shape
        _, Lk, _, _ = keys.shape
        scale = self.scale or (1.0 / math.sqrt(E))

        Q = queries.permute(0, 2, 1, 3)  # [B,H,Lq,E]
        K = keys.permute(0, 2, 1, 3)     # [B,H,Lk,E]
        V = values.permute(0, 2, 1, 3)   # [B,H,Lk,E]

        # select top queries
        top_idx, n_top = self._sample_scores(Q, K)  # top_idx: [B,H,n_top]

        # build initial context
        if self.mask_flag:
            # for causal self-attn: cumulative sum context
            context = V.cumsum(dim=2)  # [B,H,Lk,E] (expects Lq==Lk in self-attn)
        else:
            context = V.mean(dim=2, keepdim=True).expand(B, H, Lq, E)  # [B,H,Lq,E]

        # gather Q_top: [B,H,n_top,E]
        idx_q = top_idx.unsqueeze(-1).expand(-1, -1, -1, E)
        Q_top = torch.gather(Q, dim=2, index=idx_q)

        # full scores for selected queries: [B,H,n_top,Lk]
        scores_top = torch.matmul(Q_top, K.transpose(-2, -1))  # [B,H,n_top,Lk]

        # apply mask if needed
        if self.mask_flag:
            if attn_mask is not None:
                # attn_mask: [B,1,Lq,Lk] -> select rows for top queries
                m = attn_mask.expand(B, H, Lq, Lk)
                idx_m = top_idx.unsqueeze(-1).expand(-1, -1, -1, Lk)
                mask_top = torch.gather(m, dim=2, index=idx_m)  # [B,H,n_top,Lk]
            else:
                # causal by indices
                key_pos = torch.arange(Lk, device=queries.device).view(1, 1, 1, Lk)
                q_pos = top_idx.unsqueeze(-1)  # [B,H,n_top,1]
                mask_top = key_pos > q_pos
            scores_top = scores_top.masked_fill(mask_top, -np.inf)

        attn = self.dropout(torch.softmax(scores_top * scale, dim=-1))     # [B,H,n_top,Lk]
        out_top = torch.matmul(attn, V)                                    # [B,H,n_top,E]

        # scatter updates back into context at top_idx positions
        context_out = context.clone()
        context_out.scatter_(dim=2, index=idx_q, src=out_top)

        out = context_out.permute(0, 2, 1, 3).contiguous()  # [B,Lq,H,E]
        return out, (None if not self.output_attention else attn)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if str(activation).lower() == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        out = self.norm2(x + y)
        return out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None and len(self.conv_layers) > 0:
            for layer, conv in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = layer(x, attn_mask=attn_mask)
                x = conv(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for layer in self.attn_layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if str(activation).lower() == "gelu" else F.relu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_sa, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = self.norm1(x + self.dropout(x_sa))

        x_ca, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        y = self.norm2(x + self.dropout(x_ca))

        y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
        out = self.norm3(y + y_ff)
        return out


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class Informer(nn.Module):
    # downstream/eval에서 y_mark_dec를 넘겨줄지 판단하는 플래그
    needs_y_mark_dec = True

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.label_len = int(cfg.data.label_len)
        self.pred_len = int(cfg.data.pred_len)
        self.enc_in = int(cfg.data.enc_in)
        self.c_out = int(cfg.data.c_out)

        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        d_model = int(cfg.model.d_model)
        n_heads = int(cfg.model.n_heads)
        e_layers = int(cfg.model.e_layers)
        d_layers = int(getattr(cfg.model, "d_layers", 1) or 1)
        d_ff = int(cfg.model.d_ff)
        dropout = float(cfg.model.dropout)
        activation = str(getattr(cfg.model, "activation", "gelu"))

        icfg = getattr(cfg.model, "informer", None)
        attn_type = str(getattr(icfg, "attn", "prob") or "prob").lower()
        factor = int(getattr(icfg, "factor", getattr(cfg.model, "factor", 5)) or 5)
        embed = str(getattr(icfg, "embed", "fixed") or "fixed")
        freq = str(getattr(icfg, "freq", getattr(cfg.data, "freq", "h")) or "h")
        distil = bool(getattr(icfg, "distil", True))
        mix = bool(getattr(icfg, "mix", True))
        output_attention = bool(getattr(icfg, "output_attention", False))

        self.output_attention = output_attention

        # Embeddings
        self.enc_embedding = DataEmbedding(self.enc_in, d_model, embed=embed, freq=freq, dropout=dropout)
        self.dec_embedding = DataEmbedding(self.enc_in, d_model, embed=embed, freq=freq, dropout=dropout)

        # Attention choice
        if attn_type == "prob":
            Attn = ProbAttention
        elif attn_type == "full":
            Attn = FullAttention
        else:
            raise ValueError(f"Unsupported informer.attn: {attn_type} (expected prob|full)")

        # Encoder
        enc_layers = []
        for _ in range(e_layers):
            inner = Attn(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=output_attention) \
                if Attn is ProbAttention else \
                Attn(mask_flag=False, attention_dropout=dropout, output_attention=output_attention)
            attn = AttentionLayer(inner, d_model=d_model, n_heads=n_heads)
            enc_layers.append(EncoderLayer(attn, d_model, d_ff=d_ff, dropout=dropout, activation=activation))

        conv_layers = None
        if distil and e_layers > 1:
            conv_layers = [ConvDistillLayer(d_model) for _ in range(e_layers - 1)]

        self.encoder = Encoder(enc_layers, conv_layers=conv_layers, norm_layer=nn.LayerNorm(d_model))

        # Decoder
        dec_layers = []
        for _ in range(d_layers):
            inner_self = Attn(mask_flag=True, factor=factor, attention_dropout=dropout, output_attention=False) \
                if Attn is ProbAttention else \
                Attn(mask_flag=True, attention_dropout=dropout, output_attention=False)
            self_attn = AttentionLayer(inner_self, d_model=d_model, n_heads=n_heads)

            cross_inner = FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False)
            cross_attn = AttentionLayer(cross_inner, d_model=d_model, n_heads=n_heads)

            dec_layers.append(DecoderLayer(self_attn, cross_attn, d_model, d_ff=d_ff, dropout=dropout, activation=activation))

        self.decoder = Decoder(
            dec_layers,
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True),
        )

        self.mix = mix  # kept for config completeness (not used explicitly here)

    def _build_dec_inp(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_dec: [B, label_len + pred_len, N]
        B, _, N = x_enc.shape
        zeros = torch.zeros((B, self.pred_len, N), device=x_enc.device, dtype=x_enc.dtype)
        return torch.cat([x_enc[:, -self.label_len :, :], zeros], dim=1)

    def _build_mark_dec(self, x_mark_enc: torch.Tensor | None, y_mark_dec: torch.Tensor | None, device, dtype):
        if y_mark_dec is not None:
            return y_mark_dec
        if x_mark_enc is None:
            return None
        B = x_mark_enc.size(0)
        Dm = x_mark_enc.size(-1)
        zeros = torch.zeros((B, self.label_len + self.pred_len, Dm), device=device, dtype=dtype)
        # 최소한 shape만 맞추는 fallback (미래 mark를 정확히 제공하려면 downstream/eval에서 y_mark_dec 전달 필요)
        zeros[:, : self.label_len, :] = x_mark_enc[:, -self.label_len :, :]
        return zeros

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        # x_enc: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc

        x_dec = self._build_dec_inp(x)  # [B, label+pred, N]
        x_mark_dec = self._build_mark_dec(x_mark_enc, y_mark_dec, device=x.device, dtype=x.dtype)

        enc_out = self.enc_embedding(x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_mask = TriangularCausalMask(x.size(0), dec_out.size(1), device=x.device).mask
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_mask, cross_mask=None)

        out = dec_out[:, -self.pred_len :, :]  # [B, H, N]

        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, y_mark_dec=y_mark_dec, **kwargs)
        return y
```

---

### 1-4) `src/itransformer/models/tide.py`

TiDE는 TSLib 구현의 MLP encoder-decoder(ResBlock, feature encoder, temporal decoder, residual projection) 구조를 그대로 따르되, 이 코드베이스의 `use_norm` 관례를 유지하고 `y_mark_dec`를 받을 수 있게 한다. ([GitHub][5])

```python
# src/itransformer/models/tide.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasLayerNorm(nn.Module):
    # LayerNorm with optional bias, matching TiDE-style blocks
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.skip = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.ln = BiasLayerNorm(output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y + self.skip(x)
        y = self.ln(y)
        return y


class TiDE(nn.Module):
    needs_y_mark_dec = True

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.label_len = int(cfg.data.label_len)
        self.pred_len = int(cfg.data.pred_len)
        self.n_vars = int(cfg.data.enc_in)

        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        # core dims
        hidden_dim = int(cfg.model.d_model)
        res_hidden = int(cfg.model.d_model)
        encoder_num = int(getattr(cfg.model, "e_layers", 2) or 2)
        decoder_num = int(getattr(cfg.model, "d_layers", 1) or 1)
        temporal_hidden = int(cfg.model.d_ff)
        dropout = float(cfg.model.dropout)

        tcfg = getattr(cfg.model, "tide", None)
        bias = bool(getattr(tcfg, "bias", True))
        feature_encode_dim = int(getattr(tcfg, "feature_encode_dim", 2) or 2)
        decode_dim = int(getattr(tcfg, "decode_dim", int(cfg.data.c_out)) or int(cfg.data.c_out))

        # time feature dim 결정: timeenc=0이면 dataset이 month/day/weekday/hour(+minute) 형태로 주는 걸 가정
        feature_dim_override = getattr(tcfg, "feature_dim", None)
        if feature_dim_override is not None:
            feature_dim = int(feature_dim_override)
        else:
            timeenc = int(getattr(cfg.data, "timeenc", 0) or 0)
            freq = str(getattr(cfg.data, "freq", "h") or "h")
            if timeenc == 0:
                feature_dim = 5 if freq == "t" else 4
            else:
                freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
                feature_dim = int(freq_map.get(freq, 4))

        self.feature_dim = int(feature_dim)
        self.feature_encode_dim = int(feature_encode_dim)
        self.decode_dim = int(decode_dim)

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        # modules
        self.feature_encoder = ResBlock(self.feature_dim, res_hidden, self.feature_encode_dim, dropout, bias)

        enc_blocks = [ResBlock(flatten_dim, res_hidden, hidden_dim, dropout, bias)]
        for _ in range(max(0, encoder_num - 1)):
            enc_blocks.append(ResBlock(hidden_dim, res_hidden, hidden_dim, dropout, bias))
        self.encoders = nn.Sequential(*enc_blocks)

        dec_blocks = []
        for _ in range(max(0, decoder_num - 1)):
            dec_blocks.append(ResBlock(hidden_dim, res_hidden, hidden_dim, dropout, bias))
        dec_blocks.append(ResBlock(hidden_dim, res_hidden, self.decode_dim * self.pred_len, dropout, bias))
        self.decoders = nn.Sequential(*dec_blocks)

        self.temporal_decoder = ResBlock(self.decode_dim + self.feature_encode_dim, temporal_hidden, 1, dropout, bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def _build_batch_mark(self, x_mark_enc: torch.Tensor | None, y_mark_dec: torch.Tensor | None, device, dtype):
        B = int(y_mark_dec.size(0) if y_mark_dec is not None else (x_mark_enc.size(0) if x_mark_enc is not None else 1))
        if x_mark_enc is None:
            if y_mark_dec is None:
                return torch.zeros((B, self.seq_len + self.pred_len, self.feature_dim), device=device, dtype=dtype)
            # y_mark_dec: [B, label+pred, D] -> history marks unknown, fallback zeros
            return torch.zeros((B, self.seq_len + self.pred_len, y_mark_dec.size(-1)), device=device, dtype=dtype)

        if y_mark_dec is None:
            zeros_future = torch.zeros((B, self.pred_len, x_mark_enc.size(-1)), device=device, dtype=dtype)
            return torch.cat([x_mark_enc, zeros_future], dim=1)

        # y_mark_dec contains [label_len + pred_len]; use only future pred_len part
        return torch.cat([x_mark_enc, y_mark_dec[:, -self.pred_len :, :]], dim=1)

    def _forecast_single(self, x_series: torch.Tensor, batch_mark: torch.Tensor) -> torch.Tensor:
        # x_series: [B, L], batch_mark: [B, L+H, Dm]
        if self.use_norm:
            means = x_series.mean(dim=1, keepdim=True).detach()
            x = x_series - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_series
            means = None
            stdev = None

        feat = self.feature_encoder(batch_mark)                      # [B, L+H, k]
        flat = torch.cat([x, feat.reshape(x.size(0), -1)], dim=-1)   # [B, L + (L+H)*k]
        hid = self.encoders(flat)                                    # [B, hidden_dim]

        decoded = self.decoders(hid).reshape(x.size(0), self.pred_len, self.decode_dim)  # [B,H,decode_dim]
        out = self.temporal_decoder(torch.cat([feat[:, self.seq_len :, :], decoded], dim=-1)).squeeze(-1)  # [B,H]
        out = out + self.residual_proj(x)

        if self.use_norm:
            out = out * stdev.squeeze(1)[:, None].repeat(1, self.pred_len)
            out = out + means.squeeze(1)[:, None].repeat(1, self.pred_len)

        return out  # [B,H]

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        # x_enc: [B, L, N]
        batch_mark = self._build_batch_mark(x_mark_enc, y_mark_dec, device=x_enc.device, dtype=x_enc.dtype)
        outs = []
        for i in range(x_enc.size(-1)):
            outs.append(self._forecast_single(x_enc[:, :, i], batch_mark).unsqueeze(-1))  # [B,H,1]
        y = torch.cat(outs, dim=-1)  # [B,H,N]
        return y, None

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, y_mark_dec=y_mark_dec, **kwargs)
        return y
```

---

## 2) factory / **init** 연결

### 2-1) `src/itransformer/models/__init__.py` 수정

현재 export에 새 모델 추가.

```python
# src/itransformer/models/__init__.py
from itransformer.models.itransformer import ITransformer
from itransformer.models.m0 import ITransformerM0
from itransformer.models.patch_transformer import PatchITransformer

# new baselines
from itransformer.models.patchtst import PatchTST
from itransformer.models.informer import Informer
from itransformer.models.dlinear import DLinear
from itransformer.models.tide import TiDE
```

---

### 2-2) `src/itransformer/models/factory.py` 수정

variant 분기만 추가하면 됨. (TSLIB 레거시 분기는 유지해도 무방) ([GitHub][6])

```python
# src/itransformer/models/factory.py
from itransformer.models.itransformer import ITransformer
from itransformer.models.m0 import ITransformerM0
from itransformer.models.patch_transformer import PatchITransformer
from itransformer.models.tslib_adapter import TSLibForecastAdapter  # legacy

from itransformer.models.patchtst import PatchTST
from itransformer.models.informer import Informer
from itransformer.models.dlinear import DLinear
from itransformer.models.tide import TiDE

PATCH_VARIANTS = {"P1", "P2", "P3", "P4"}

def build_model(cfg):
    v = str(getattr(cfg.model, "variant", "") or "")
    if v == "P0":
        return ITransformer(cfg)
    if v == "M0":
        return ITransformerM0(cfg)
    if v in PATCH_VARIANTS:
        return PatchITransformer(cfg)
    if v == "TSLIB":
        return TSLibForecastAdapter(cfg)

    if v == "PatchTST":
        return PatchTST(cfg)
    if v == "Informer":
        return Informer(cfg)
    if v == "DLinear":
        return DLinear(cfg)
    if v == "TiDE":
        return TiDE(cfg)

    raise ValueError(f"Unknown model.variant: {v}")
```

---

## 3) Hydra 모델 설정 추가

가이드가 요구하는 공통 필드 + meta/patch 블록 유지.

### 3-1) `conf/model/DLinear.yaml`

```yaml
variant: DLinear

# guide-required common fields (some are not used by DLinear, but kept for schema consistency)
d_model: 512
n_heads: 8
e_layers: 2
d_ff: 2048
dropout: 0.0
activation: gelu
use_norm: true
factor: 5

dlinear:
  kernel_size: 25
  individual: true

meta:
  enabled: false
  mode: none
  source: none
  proj: linear
  mlp_hidden: 512

patch:
  enabled: false
  patch_len: 0
  mode: overlap
  local_win: 24
  use_pos_emb: true
```

---

### 3-2) `conf/model/PatchTST.yaml`

```yaml
variant: PatchTST

d_model: 512
n_heads: 8
e_layers: 3
d_ff: 2048
activation: gelu
use_norm: true
factor: 5

patchtst:
  patch_len: 16
  stride: 8
  padding: 8
  head_dropout: 0.1

meta:
  enabled: false
  mode: none
  source: none
  proj: linear
  mlp_hidden: 512

patch:
  enabled: false
  patch_len: 0
  mode: overlap
  local_win: 24
  use_pos_emb: true
```

---

### 3-3) `conf/model/Informer.yaml`

```yaml
variant: Informer

d_model: 512
n_heads: 8
e_layers: 2
d_layers: 1
d_ff: 2048
dropout: 0.1
activation: gelu
use_norm: true
factor: 5

informer:
  attn: prob         # prob | full
  factor: 5
  embed: fixed       # fixed | learned | timeF
  freq: ${data.freq}
  distil: true
  mix: true
  output_attention: false

meta:
  enabled: false
  mode: none
  source: none
  proj: linear
  mlp_hidden: 512

patch:
  enabled: false
  patch_len: 0
  mode: overlap
  local_win: 24
  use_pos_emb: true
```

---

### 3-4) `conf/model/TiDE.yaml`

```yaml
variant: TiDE

d_model: 512
n_heads: 8
e_layers: 2
d_layers: 1
d_ff: 2048
dropout: 0.1
activation: relu
use_norm: true
factor: 5

tide:
  feature_encode_dim: 2
  bias: true
  decode_dim: ${data.c_out}
  feature_dim: null   # null이면 (timeenc,freq)로 추정, 필요하면 정수로 override

meta:
  enabled: false
  mode: none
  source: none
  proj: linear
  mlp_hidden: 512

patch:
  enabled: false
  patch_len: 0
  mode: overlap
  local_win: 24
  use_pos_emb: true
```

---

## 4) downstream/eval에 y_mark_dec 전달 최소 패치

### 4-1) `src/itransformer/downstream.py`

현재는 `use_tslib`일 때만 y_mark_dec를 넘김. ([GitHub][1])
Informer/TiDE는 `needs_y_mark_dec=True`를 갖고 있으니 그 플래그로 분기 확장.

#### (A) `_evaluate` 함수 내부 변경

기존:

```python
if use_tslib:
    out = model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
elif meta_emb is None:
    out = model(x_enc, x_mark)
else:
    out = model(x_enc, x_mark, meta_emb)
```

변경:

```python
needs_mark = bool(getattr(model, "needs_y_mark_dec", False))
if use_tslib or needs_mark:
    out = model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
elif meta_emb is None:
    out = model(x_enc, x_mark)
else:
    out = model(x_enc, x_mark, meta_emb)
```

#### (B) train loop 내부 `out = model(...)` 부분도 동일 변경

`optimizer.zero_grad()` 직후 모델 호출부를 위와 동일하게 바꿔주면 끝.

---

### 4-2) `src/itransformer/eval.py`

eval도 `_predict()`가 `use_tslib`만 분기하고 있음. ([GitHub][7])
여기도 동일하게 확장.

기존 `_predict`:

```python
def _predict(model, x_enc, x_mark, meta_emb, y_mark=None, *, use_tslib: bool = False):
    if use_tslib:
        return model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
    if meta_emb is None:
        return model(x_enc, x_mark)
    return model(x_enc, x_mark, meta_emb)
```

변경:

```python
def _predict(model, x_enc, x_mark, meta_emb, y_mark=None, *, use_tslib: bool = False):
    needs_mark = bool(getattr(model, "needs_y_mark_dec", False))
    if use_tslib or needs_mark:
        return model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
    if meta_emb is None:
        return model(x_enc, x_mark)
    return model(x_enc, x_mark, meta_emb)
```

---

## 5) Smoke test용 plan 템플릿

가이드가 권장한 대로, 처음엔 `train.mode=sl`로 1 epoch만 돌려서 shape/저장물 확인하는 게 안전.
(아래는 DLinear 예시. PatchTST/Informer/TiDE도 `model=`만 바꿔서 복제하면 됨)

`conf/plan/DLinear_smoke_e1.yaml`:

```yaml
plan_id: DLinear_smoke_e1
runs:
  - id_template: "SUP-{dataset}.DLinear.smoke.e1.pr{pred_len}.sd{seed}"
    entry: downstream
    overrides:
      - data={dataset}
      - data.pred_len={pred_len}
      - model=DLinear
      - train=downstream
      - train.mode=sl
      - train.epochs=1
      - train.batch_size=128
      - train.patience=0
      - optim.name=adam
      - optim.lr=0.0001
      - optim.scheduler=none
      - metadata.enabled=false
      - runtime.device=cuda
      - runtime.seed={seed}
      - runtime.deterministic=true
      - run.code=SUP-{dataset}
      - run.hparams_tag=dlinear-smoke-e1-pr{pred_len}
      - run.id={id}
sweep:
  dataset: [ETTh1]
  pred_len: [96]
  seed: [0]
ops: []
cmps: []
aggs: []
```

---

## 6) “하이퍼파라미터 누락” 체크리스트

네가 요구한 “모델 고유 하이퍼파라미터 하나도 누락하지 않기” 관점에서, 위 통합은 최소한 다음을 모두 hTST: `patch_len/stride/padding/head_dropout` + 공통(`d_model/n_heads/e_layers/d_ff/dropout/activation/use_norm`) ([GitHub][3])

* Informer: `attn(prob/full)/factor/embed/freq/distil/mix/output_attention` + 공통 + `d_layers` ([GitHub][4])
* DLinear: `kernel_size/individual` + 공통 ([GitHub][2])
* TiDE: `feature_encode_dim/bias/decode_dim/feature_dim` + 공통 + `d_layers` ([GitHub][5])



[1]: https://raw.githubusercontent.com/p0tat0-113/iTransformer_based_industrial_TS_encoder_research/new_model/src/itransformer/downstream.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/models/model.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/PatchTST.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/models/decoder.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TiDE.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/layers/PatchTST_backbone.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/p0tat0-113/iTransformer_based_industrial_TS_encoder_research/new_model/src/itransformer/eval.py "raw.githubusercontent.com"
