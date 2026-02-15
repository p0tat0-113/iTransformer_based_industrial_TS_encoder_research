from __future__ import annotations

import math

import torch
import torch.nn as nn

from itransformer.models.layers.attention import AttentionLayer, FullAttention
from itransformer.models.layers.transformer import Encoder, EncoderLayer


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int, *, contiguous: bool = False):
        super().__init__()
        self.dim0 = int(dim0)
        self.dim1 = int(dim1)
        self.contiguous = bool(contiguous)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(self.dim0, self.dim1)
        return y.contiguous() if self.contiguous else y


class _SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class _PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(padding)
        self.pad = nn.ReplicationPad1d((0, self.padding))
        self.proj = nn.Linear(self.patch_len, d_model, bias=False)
        self.pos = _SinusoidalPositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # x: [B, N, L]
        n_vars = int(x.size(1))
        x = self.pad(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, N, P, patch_len]
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3))  # [B*N, P, patch_len]
        x = self.proj(x) + self.pos(x)
        x = self.dropout(x)
        return x, n_vars, int(x.size(1))


class _FlattenHead(nn.Module):
    def __init__(self, in_dim: int, pred_len: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(in_dim, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D, P]
        y = self.flatten(x)  # [B, N, D*P]
        y = self.linear(y)  # [B, N, H]
        return self.dropout(y)


class PatchTST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.pred_len = int(cfg.data.pred_len)
        self.n_vars_cfg = int(cfg.data.enc_in)
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
        if patch_len <= 0 or stride <= 0:
            raise ValueError(f"PatchTST expects patch_len/stride > 0 (got {patch_len}/{stride})")

        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.patch_embedding = _PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        attn_layers = []
        for _ in range(e_layers):
            attn = AttentionLayer(
                FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
                d_model=d_model,
                n_heads=n_heads,
            )
            attn_layers.append(EncoderLayer(attn, d_model, d_ff=d_ff, dropout=dropout, activation=activation))
        norm = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))
        self.encoder = Encoder(attn_layers, norm_layer=norm)

        patch_num = (self.seq_len + self.padding - self.patch_len) // self.stride + 1
        self.head = _FlattenHead(d_model * int(patch_num), self.pred_len, dropout=head_dropout)

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc

        x = x.transpose(1, 2)  # [B, N, L]
        enc_in, n_vars, patch_num = self.patch_embedding(x)  # [B*N, P, D]
        enc_out, attns = self.encoder(enc_in)  # [B*N, P, D]

        bsz = x_enc.size(0)
        dim = enc_out.size(2)
        if n_vars != self.n_vars_cfg:
            raise ValueError(
                f"PatchTST input_vars mismatch: input={n_vars}, config.enc_in={self.n_vars_cfg}"
            )
        enc_out = enc_out.view(bsz, n_vars, patch_num, dim).permute(0, 1, 3, 2)  # [B, N, D, P]
        dec_out = self.head(enc_out).transpose(1, 2)  # [B, H, N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, **kwargs)
        return y
