from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _time_mark_dim_from_cfg(cfg, fallback: int = 4) -> int:
    tcfg = getattr(cfg.model, "tide", None)
    mark_dim_override = int(getattr(tcfg, "feature_dim", 0) or 0)
    if mark_dim_override > 0:
        return mark_dim_override
    timeenc = int(getattr(cfg.data, "timeenc", 0) or 0)
    freq = str(getattr(cfg.data, "freq", "h") or "h").lower()
    if timeenc == 0:
        return 5 if freq == "t" else 4
    freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
    return int(freq_map.get(freq, fallback))


def _align_mark_dim(x_mark: torch.Tensor, mark_dim: int) -> torch.Tensor:
    cur = int(x_mark.size(-1))
    if cur == mark_dim:
        return x_mark
    if cur > mark_dim:
        return x_mark[..., :mark_dim]
    pad = torch.zeros(
        x_mark.size(0),
        x_mark.size(1),
        mark_dim - cur,
        dtype=x_mark.dtype,
        device=x_mark.device,
    )
    return torch.cat([x_mark, pad], dim=-1)


class _BiasLayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _ResBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.skip = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = _BiasLayerNorm(out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y + self.skip(x)
        return self.norm(y)


class TiDE(nn.Module):
    needs_y_mark_dec = True

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.label_len = int(cfg.data.label_len)
        self.pred_len = int(cfg.data.pred_len)
        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        hidden_dim = int(cfg.model.d_model)
        res_hidden = int(cfg.model.d_model)
        encoder_num = int(getattr(cfg.model, "e_layers", 2) or 2)
        decoder_num = int(getattr(cfg.model, "d_layers", 1) or 1)
        temporal_hidden = int(cfg.model.d_ff)
        dropout = float(cfg.model.dropout)

        tcfg = getattr(cfg.model, "tide", None)
        self.feature_dim = _time_mark_dim_from_cfg(cfg)
        self.feature_encode_dim = int(getattr(tcfg, "feature_encode_dim", 2) or 2)
        self.decode_dim = int(getattr(tcfg, "decode_dim", int(cfg.data.c_out)) or int(cfg.data.c_out))
        bias = bool(getattr(tcfg, "bias", True))

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = _ResBlock(self.feature_dim, res_hidden, self.feature_encode_dim, dropout, bias)
        enc_blocks = [_ResBlock(flatten_dim, res_hidden, hidden_dim, dropout, bias)]
        for _ in range(max(0, encoder_num - 1)):
            enc_blocks.append(_ResBlock(hidden_dim, res_hidden, hidden_dim, dropout, bias))
        self.encoders = nn.Sequential(*enc_blocks)

        dec_blocks = []
        for _ in range(max(0, decoder_num - 1)):
            dec_blocks.append(_ResBlock(hidden_dim, res_hidden, hidden_dim, dropout, bias))
        dec_blocks.append(_ResBlock(hidden_dim, res_hidden, self.decode_dim * self.pred_len, dropout, bias))
        self.decoders = nn.Sequential(*dec_blocks)

        self.temporal_decoder = _ResBlock(
            self.decode_dim + self.feature_encode_dim,
            temporal_hidden,
            1,
            dropout,
            bias,
        )
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def _build_batch_mark(
        self,
        x_mark_enc: torch.Tensor | None,
        y_mark_dec: torch.Tensor | None,
        *,
        batch: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if x_mark_enc is None and y_mark_dec is None:
            return torch.zeros((batch, self.seq_len + self.pred_len, self.feature_dim), device=device, dtype=dtype)

        if x_mark_enc is None:
            future = y_mark_dec[:, -self.pred_len :, :]
            hist = torch.zeros((batch, self.seq_len, future.size(-1)), device=device, dtype=future.dtype)
            marks = torch.cat([hist, future], dim=1)
            return _align_mark_dim(marks, self.feature_dim)

        if y_mark_dec is None:
            hist = x_mark_enc
            future = x_mark_enc[:, -1:, :].repeat(1, self.pred_len, 1)
            marks = torch.cat([hist, future], dim=1)
            return _align_mark_dim(marks, self.feature_dim)

        hist = x_mark_enc
        future = y_mark_dec[:, -self.pred_len :, :]
        marks = torch.cat([hist, future], dim=1)
        return _align_mark_dim(marks, self.feature_dim)

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

        feat = self.feature_encoder(batch_mark)  # [B, L+H, k]
        flat = torch.cat([x, feat.reshape(x.size(0), -1)], dim=-1)  # [B, L + (L+H)*k]
        hid = self.encoders(flat)

        dec = self.decoders(hid).reshape(x.size(0), self.pred_len, self.decode_dim)  # [B, H, decode_dim]
        out = self.temporal_decoder(torch.cat([feat[:, self.seq_len :, :], dec], dim=-1)).squeeze(-1)  # [B, H]
        out = out + self.residual_proj(x)

        if self.use_norm:
            out = out * stdev.squeeze(1)[:, None].repeat(1, self.pred_len)
            out = out + means.squeeze(1)[:, None].repeat(1, self.pred_len)
        return out

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        # x_enc: [B, L, N]
        batch_mark = self._build_batch_mark(
            x_mark_enc,
            y_mark_dec,
            batch=x_enc.size(0),
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        outs = []
        n_vars = int(x_enc.size(-1))
        for i in range(n_vars):
            outs.append(self._forecast_single(x_enc[:, :, i], batch_mark).unsqueeze(-1))
        y = torch.cat(outs, dim=-1)  # [B, H, N]
        return y, None

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        y, _ = self.forecast(
            x_enc,
            x_mark_enc=x_mark_enc,
            meta_emb=meta_emb,
            y_mark_dec=y_mark_dec,
            **kwargs,
        )
        return y
