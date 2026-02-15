from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1:
            raise ValueError(f"dlinear.kernel_size must be >= 1 (got {self.kernel_size})")
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        if self.kernel_size == 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        tail = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, tail], dim=1)
        return self.avg(x_pad.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.mavg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.mavg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """LTSF-Linear baseline (seasonal + trend linear heads)."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.pred_len = int(cfg.data.pred_len)
        self.n_vars_cfg = int(cfg.data.enc_in)
        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        dcfg = getattr(cfg.model, "dlinear", None)
        self.kernel_size = int(getattr(dcfg, "kernel_size", 25) or 25)
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"dlinear.kernel_size must be odd for length-preserving moving average (got {self.kernel_size})"
            )
        self.individual = bool(getattr(dcfg, "individual", True))

        self.decomp = SeriesDecomposition(self.kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.n_vars_cfg)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.n_vars_cfg)]
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

        seasonal, trend = self.decomp(x)
        seasonal = seasonal.transpose(1, 2)  # [B, N, L]
        trend = trend.transpose(1, 2)  # [B, N, L]

        n_vars_in = int(seasonal.size(1))
        if self.individual:
            n_vars_head = len(self.linear_seasonal)
            if n_vars_in != n_vars_head:
                raise ValueError(
                    "DLinear individual head size mismatch: "
                    f"input_vars={n_vars_in}, configured_vars={n_vars_head}. "
                    "Set data.enc_in to the true variable count or use dlinear.individual=false."
                )
            out_s = []
            out_t = []
            for i in range(n_vars_in):
                out_s.append(self.linear_seasonal[i](seasonal[:, i, :]).unsqueeze(1))
                out_t.append(self.linear_trend[i](trend[:, i, :]).unsqueeze(1))
            out_s = torch.cat(out_s, dim=1)  # [B, N, H]
            out_t = torch.cat(out_t, dim=1)  # [B, N, H]
        else:
            out_s = self.linear_seasonal(seasonal)
            out_t = self.linear_trend(trend)

        y = (out_s + out_t).transpose(1, 2)  # [B, H, N]
        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return y, None

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, **kwargs):
        y, _ = self.forecast(x_enc, x_mark_enc=x_mark_enc, meta_emb=meta_emb, **kwargs)
        return y
