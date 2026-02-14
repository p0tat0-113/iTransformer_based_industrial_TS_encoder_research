import torch
import torch.nn as nn
import torch.nn.functional as F

from itransformer.models.layers.attention import FullAttention, AttentionLayer
from itransformer.models.layers.transformer import Encoder, EncoderLayer


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _moving_average_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Replicate-padded moving average over time axis.

    x: [B, L, N]
    """
    k = int(kernel_size)
    if k <= 1:
        return x
    pad_total = k - 1
    left = pad_total // 2
    right = pad_total - left
    xt = x.permute(0, 2, 1)  # [B, N, L]
    xt = F.pad(xt, (left, right), mode="replicate")
    trend = F.avg_pool1d(xt, kernel_size=k, stride=1)
    return trend.permute(0, 2, 1)


def _sigmoid01_torch(x: torch.Tensor, *, k: float, x0: float) -> torch.Tensor:
    """Sigmoid mapped to [0, 1] on x in [0, 1]."""
    s = torch.sigmoid(k * (x - x0))
    s0 = torch.sigmoid(torch.tensor(k * (0.0 - x0), dtype=x.dtype, device=x.device))
    s1 = torch.sigmoid(torch.tensor(k * (1.0 - x0), dtype=x.dtype, device=x.device))
    return (s - s0) / (s1 - s0 + 1e-12)


def _hgate_t0_from_pred_len(pred_len: int) -> int:
    """Map pred_len to sigmoid center index for conservative long-horizon bias init."""
    l_min, l_max = 96, 720
    t0_at_lmax = 350
    if pred_len <= l_min:
        return max(1, pred_len - 1)
    t0_lmin = l_min - 1
    alpha = (pred_len - l_min) / float(l_max - l_min)
    t0 = int(round((1.0 - alpha) * t0_lmin + alpha * t0_at_lmax))
    return int(min(max(t0, 1), pred_len - 1))


def _make_hgate_bias_schedule(
    *,
    pred_len: int,
    init_start: float,
    init_end: float,
    schedule: str,
) -> torch.Tensor:
    """Create horizon-gate bias initialization curve."""
    if pred_len < 2:
        return torch.tensor([init_start], dtype=torch.float32)
    if schedule == "linear":
        return torch.linspace(init_start, init_end, steps=pred_len)
    if schedule != "sigmix_v1":
        raise ValueError(f"Unsupported horizon_gate.init_schedule: {schedule}")

    # Fixed-hparam schedule proposed by experiment notes:
    # only pred_len + init_start are used; init_end is intentionally ignored.
    l_max = 720.0
    delta_max = 6.0
    q = 2.0
    a_min = 0.02
    a_max = 0.10
    r_a = 1.2
    p = 2.4
    k = 18.0

    frac_l = pred_len / l_max
    delta = delta_max * (frac_l**q)
    a = a_min + (a_max - a_min) * (frac_l**r_a)
    a = float(min(max(a, 0.0), 0.95))

    t0 = _hgate_t0_from_pred_len(pred_len)
    x0 = t0 / float(pred_len - 1)
    x = torch.linspace(0.0, 1.0, pred_len, dtype=torch.float32)
    soft = x.pow(p)
    steep = _sigmoid01_torch(x, k=k, x0=x0)

    d_raw = a * soft + (1.0 - a) * steep
    d = (d_raw - d_raw[0]) / (d_raw[-1] - d_raw[0] + 1e-12)
    return init_start - delta * d


class CausalConvBlock1D(nn.Module):
    """Conv over patch-index axis (m_p) with configurable padding mode.

    padding_mode:
    - causal: left-pad only (strictly past-to-current)
    - same: symmetric/asymmetric same padding (bidirectional context in lookback)

    conv_type:
    - standard: regular Conv1d(d->d, k)
    - dwsep: depthwise Conv1d(groups=d) + pointwise Conv1d(1x1)
    """

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        layers: int,
        dilation: int,
        dropout: float,
        padding_mode: str = "causal",
        conv_type: str = "standard",
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError("temporal_conv.layers must be >= 1")
        if kernel_size <= 0:
            raise ValueError("temporal_conv.kernel_size must be >= 1")
        if dilation <= 0:
            raise ValueError("temporal_conv.dilation must be >= 1")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_mode = str(padding_mode or "causal").lower()
        if self.padding_mode not in ("causal", "same"):
            raise ValueError(
                f"temporal_conv.padding_mode must be one of: causal, same (got {self.padding_mode})"
            )
        self.conv_type = str(conv_type or "standard").lower()
        if self.conv_type not in ("standard", "dwsep"):
            raise ValueError(
                f"temporal_conv.type must be one of: standard, dwsep (got {self.conv_type})"
            )
        self.ln = nn.LayerNorm(d_model)
        if self.conv_type == "standard":
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                    )
                    for _ in range(int(layers))
                ]
            )
            self.pointwise_convs = None
        else:
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        groups=d_model,
                    )
                    for _ in range(int(layers))
                ]
            )
            self.pointwise_convs = nn.ModuleList(
                [nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1) for _ in range(int(layers))]
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, m, d] (token-independent temporal encoding)
        bsz, tok, m, dim = x.shape
        if m <= 1:
            return x

        y = self.ln(x)
        y = y.reshape(bsz * tok, m, dim).transpose(1, 2)  # [B*T, d, m]

        total_pad = (self.kernel_size - 1) * self.dilation
        if self.padding_mode == "causal":
            left_pad, right_pad = total_pad, 0
        else:
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
        for i, conv in enumerate(self.convs):
            y = F.pad(y, (left_pad, right_pad))
            y = conv(y)
            if self.pointwise_convs is not None:
                y = self.pointwise_convs[i](y)
            if i != len(self.convs) - 1:
                y = F.gelu(y)
            y = self.dropout(y)

        y = y.transpose(1, 2).reshape(bsz, tok, m, dim)
        return x + y


class PMASlotizer(nn.Module):
    """Pooling by Multihead Attention (PMA) with optional FFN+residual."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_ffn: bool,
        kv_include_seeds: bool = False,
    ):
        super().__init__()
        self.kv_include_seeds = bool(kv_include_seeds)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(n_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, seeds: torch.Tensor, *, need_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [B, T, m, d]
        # seeds: [K, d]
        bsz, tok, m, dim = x.shape
        x_flat = x.reshape(bsz * tok, m, dim)
        q = seeds.unsqueeze(0).expand(x_flat.size(0), -1, -1)  # [B*T, K, d]
        if self.kv_include_seeds:
            # Flamingo Perceiver Resampler style: allow seed-to-seed interactions inside the
            # cross-attention by adding the latent seeds to K/V.
            kv = torch.cat([x_flat, q], dim=1)  # [B*T, m+K, d]
            out, attn_w = self.attn(q, kv, kv, need_weights=need_weights, average_attn_weights=True)
        else:
            out, attn_w = self.attn(q, x_flat, x_flat, need_weights=need_weights, average_attn_weights=True)
        out = self.dropout(out)
        if self.use_ffn:
            out = out + self.ffn(self.ffn_ln(out))
        return out.reshape(bsz, tok, out.size(1), dim), attn_w  # [B, T, K, d], [B*T, K, S] (or None)


class MeanSlotizer(nn.Module):
    """Mean pooling over patches; replicates pooled vector to K slots (no params)."""

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be >= 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, m, d]
        pooled = x.mean(dim=2)  # [B, T, d]
        return pooled.unsqueeze(2).expand(-1, -1, self.k, -1).contiguous()  # [B, T, K, d]


class SlotFuseMLP(nn.Module):
    def __init__(self, d_model: int, *, k_total: int, hidden: int, dropout: float):
        super().__init__()
        self.k_total = int(k_total)
        self.net = nn.Sequential(
            nn.Linear(self.k_total * d_model, int(hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, K, d]
        bsz, tok, k, dim = x.shape
        if k != self.k_total:
            raise ValueError(f"Expected K={self.k_total}, got K={k}")
        h = x.reshape(bsz, tok, k * dim)
        return self.net(h)


class LowRankLinear(nn.Module):
    """Low-rank factorized linear layer: W ~= W2 @ W1."""

    def __init__(self, in_features: int, out_features: int, *, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        self.w1 = nn.Linear(self.in_features, self.rank, bias=False)
        self.w2 = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


class ResidualGatedFuse(nn.Module):
    """Baseline (p=L) + gated residual corrections from other slots.

    correction-form + global gate:
        delta_i = extra_i - baseline
        alpha = softmax(score_i)  (over extras)
        g = sigmoid(Linear(baseline))  (scalar per token)
        fused = baseline + g * sum_i alpha_i * delta_i
    """

    def __init__(
        self,
        d_model: int,
        *,
        k_total: int,
        baseline_idx: int,
        hidden: int,
        dropout: float,
        extra_slot_dropout: float = 0.0,
        score_mode: str = "full",
        score_rank: int = 0,
        build_scorer: bool = True,
    ):
        super().__init__()
        self.k_total = int(k_total)
        self.baseline_idx = int(baseline_idx)
        if not (0 <= self.baseline_idx < self.k_total):
            raise ValueError(f"baseline_idx must be in [0,{self.k_total-1}], got {self.baseline_idx}")
        self.k_extra = self.k_total - 1
        if self.k_extra <= 0:
            raise ValueError("ResidualGatedFuse requires k_total >= 2")

        self.build_scorer = bool(build_scorer)
        hidden = int(hidden)
        dropout = float(dropout)
        self.extra_slot_dropout = float(extra_slot_dropout)
        if self.extra_slot_dropout < 0.0 or self.extra_slot_dropout >= 1.0:
            raise ValueError(
                f"slot_fuse.extra_slot_dropout must be in [0, 1), got {self.extra_slot_dropout}"
            )
        self.score_mode = "none"
        self.score_rank = 0
        self.score_mlp = None
        if self.build_scorer:
            if hidden <= 0:
                raise ValueError("hidden must be >= 1")
            self.score_mode = str(score_mode or "full").lower()
            if self.score_mode not in ("full", "lowrank"):
                raise ValueError(
                    f"slot_fuse.score_mode must be one of: full, lowrank (got {self.score_mode})"
                )
            self.score_rank = int(score_rank)
            if self.score_mode == "lowrank" and self.score_rank <= 0:
                raise ValueError(
                    "slot_fuse.score_rank must be >= 1 when slot_fuse.score_mode=lowrank "
                    f"(got {self.score_rank})"
                )

            # score_i = MLP([u_L ; extra_i ; (extra_i - u_L)]) -> scalar
            if self.score_mode == "lowrank":
                score_in = LowRankLinear(3 * int(d_model), hidden, rank=self.score_rank, bias=True)
            else:
                score_in = nn.Linear(3 * int(d_model), hidden)
            self.score_mlp = nn.Sequential(
                score_in,
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        self.gate = nn.Linear(int(d_model), 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def compute_alpha_g(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.score_mlp is None:
            raise RuntimeError(
                "ResidualGatedFuse scorer is disabled (build_scorer=false); "
                "compute_alpha_g() cannot be used in this configuration."
            )
        # x: [B, T, K, d]
        bsz, tok, k, dim = x.shape
        if k != self.k_total:
            raise ValueError(f"Expected K={self.k_total}, got K={k}")

        baseline = x[:, :, self.baseline_idx, :]  # [B, T, d]
        extras = torch.cat(
            [x[:, :, : self.baseline_idx, :], x[:, :, self.baseline_idx + 1 :, :]],
            dim=2,
        )  # [B, T, K-1, d]

        base = baseline.unsqueeze(2).expand_as(extras)  # [B, T, K-1, d]
        delta = extras - base  # [B, T, K-1, d]
        feat = torch.cat([base, extras, delta], dim=-1)  # [B, T, K-1, 3d]
        scores = self.score_mlp(feat).squeeze(-1)  # [B, T, K-1]
        if self.training and self.extra_slot_dropout > 0.0:
            # Drop only extra slots and renormalize among remaining extras.
            keep = torch.rand_like(scores) > self.extra_slot_dropout  # [B, T, K-1]
            all_dropped = ~keep.any(dim=2, keepdim=True)  # [B, T, 1]
            if all_dropped.any():
                # Ensure at least one extra stays active per token.
                max_idx = scores.argmax(dim=2, keepdim=True)  # [B, T, 1]
                rescue = torch.zeros_like(keep)
                rescue.scatter_(2, max_idx, True)
                keep = keep | (all_dropped.expand_as(keep) & rescue)
            masked_scores = scores.masked_fill(~keep, -1e9)
            alpha = torch.softmax(masked_scores, dim=2).unsqueeze(-1)  # [B, T, K-1, 1]
        else:
            alpha = torch.softmax(scores, dim=2).unsqueeze(-1)  # [B, T, K-1, 1]
        g = torch.sigmoid(self.gate(baseline))  # [B, T, 1]
        return alpha, g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, K, d]
        bsz, tok, k, dim = x.shape
        if k != self.k_total:
            raise ValueError(f"Expected K={self.k_total}, got K={k}")

        baseline = x[:, :, self.baseline_idx, :]  # [B, T, d]
        extras = torch.cat(
            [x[:, :, : self.baseline_idx, :], x[:, :, self.baseline_idx + 1 :, :]],
            dim=2,
        )  # [B, T, K-1, d]

        base = baseline.unsqueeze(2).expand_as(extras)  # [B, T, K-1, d]
        delta = extras - base  # [B, T, K-1, d]
        alpha, g = self.compute_alpha_g(x)
        corr = (alpha * delta).sum(dim=2)  # [B, T, d]
        return baseline + g * corr


class OutputAttentionScorer(nn.Module):
    """Compute slot alpha in output space via low-rank attention-style scoring.

    Inputs:
      - baseline: [B, T, H]
      - extras:   [B, T, K-1, H]
    Output:
      - alpha:    [B, T, K-1, 1]
    """

    def __init__(
        self,
        *,
        pred_len: int,
        k_extra: int,
        rank: int,
        use_delta_key: bool = True,
        norm: str = "layernorm",
        tau: float = 1.0,
        extra_slot_dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.k_extra = int(k_extra)
        self.rank = int(rank)
        self.use_delta_key = bool(use_delta_key)
        self.norm_mode = str(norm or "layernorm").lower()
        self.tau = float(tau)
        self.extra_slot_dropout = float(extra_slot_dropout)

        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be >= 1, got {self.pred_len}")
        if self.k_extra <= 0:
            raise ValueError(f"k_extra must be >= 1, got {self.k_extra}")
        if self.rank <= 0:
            raise ValueError(f"output_attn_rank must be >= 1, got {self.rank}")
        if self.norm_mode not in ("layernorm", "l2", "none"):
            raise ValueError(
                "slot_fuse.output_attn_norm must be one of: layernorm, l2, none "
                f"(got {self.norm_mode})"
            )
        if self.tau <= 0.0:
            raise ValueError(f"slot_fuse.output_attn_tau must be > 0, got {self.tau}")
        if self.extra_slot_dropout < 0.0 or self.extra_slot_dropout >= 1.0:
            raise ValueError(
                "slot_fuse.extra_slot_dropout must be in [0, 1), "
                f"got {self.extra_slot_dropout}"
            )

        self.norm = nn.LayerNorm(self.pred_len) if self.norm_mode == "layernorm" else None
        self.q_proj = nn.Linear(self.pred_len, self.rank, bias=True)
        k_in = self.pred_len * 2 if self.use_delta_key else self.pred_len
        self.k_proj = nn.Linear(k_in, self.rank, bias=True)
        self.score_bias = nn.Parameter(torch.zeros(self.k_extra))

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "none":
            return x
        if self.norm_mode == "l2":
            return F.normalize(x, p=2.0, dim=-1, eps=1e-8)
        return self.norm(x)

    def _slot_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, T, K-1]
        if self.training and self.extra_slot_dropout > 0.0:
            keep = torch.rand_like(scores) > self.extra_slot_dropout
            all_dropped = ~keep.any(dim=2, keepdim=True)
            if all_dropped.any():
                max_idx = scores.argmax(dim=2, keepdim=True)
                rescue = torch.zeros_like(keep)
                rescue.scatter_(2, max_idx, True)
                keep = keep | (all_dropped.expand_as(keep) & rescue)
            scores = scores.masked_fill(~keep, -1e9)
        return torch.softmax(scores, dim=2)

    def forward(self, baseline: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # baseline: [B, T, H], extras: [B, T, K-1, H]
        base = baseline.unsqueeze(2).expand_as(extras)
        delta = extras - base

        base_n = self._apply_norm(base)
        extras_n = self._apply_norm(extras)
        delta_n = self._apply_norm(delta)

        q = self.q_proj(self._apply_norm(baseline))  # [B, T, r]
        if self.use_delta_key:
            k_in = torch.cat([extras_n, delta_n], dim=-1)  # [B, T, K-1, 2H]
        else:
            k_in = extras_n  # [B, T, K-1, H]
        k = self.k_proj(k_in)  # [B, T, K-1, r]

        denom = (float(self.rank) ** 0.5) * self.tau
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / denom  # [B, T, K-1]
        scores = scores + self.score_bias.view(1, 1, -1)
        alpha = self._slot_softmax(scores).unsqueeze(-1)  # [B, T, K-1, 1]
        return alpha


class HorizonLowRankScorer(nn.Module):
    """Horizon-aware Low-Rank Slot Gating (HLSG) scorer in representation space.

    Computes horizon-wise slot weights alpha_{i,h} using a low-rank factorization:

        a_i = W([u_L ; u_i ; (u_i - u_L)])   in R^r
        score_{i,h} = <a_i, e_h> / (sqrt(r) * tau)
        alpha_{i,h} = softmax_i(score_{i,h})

    Inputs:
      - baseline_u: [B, T, d]
      - extras_u:   [B, T, K-1, d]
    Output:
      - alpha:      [B, T, K-1, H]  (H = pred_len)
    """

    def __init__(
        self,
        *,
        d_model: int,
        pred_len: int,
        k_extra: int,
        rank: int,
        dropout: float,
        norm: str = "layernorm",
        tau: float = 1.0,
        extra_slot_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.k_extra = int(k_extra)
        self.rank = int(rank)
        self.dropout_p = float(dropout)
        self.norm_mode = str(norm or "layernorm").lower()
        self.tau = float(tau)
        self.extra_slot_dropout = float(extra_slot_dropout)

        if self.d_model <= 0:
            raise ValueError(f"d_model must be >= 1 (got {self.d_model})")
        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be >= 1 (got {self.pred_len})")
        if self.k_extra <= 0:
            raise ValueError(f"k_extra must be >= 1 (got {self.k_extra})")
        if self.rank <= 0:
            raise ValueError(f"rank must be >= 1 (got {self.rank})")
        if self.norm_mode not in ("layernorm", "l2", "none"):
            raise ValueError(
                "slot_fuse.hls_norm must be one of: layernorm, l2, none "
                f"(got {self.norm_mode})"
            )
        if self.tau <= 0.0:
            raise ValueError(f"slot_fuse.hls_tau must be > 0 (got {self.tau})")
        if self.extra_slot_dropout < 0.0 or self.extra_slot_dropout >= 1.0:
            raise ValueError(
                "slot_fuse.extra_slot_dropout must be in [0, 1), "
                f"got {self.extra_slot_dropout}"
            )

        self.u_norm = nn.LayerNorm(self.d_model) if self.norm_mode == "layernorm" else None
        self.in_proj = nn.Linear(3 * self.d_model, self.rank, bias=True)
        # Start from near-uniform alpha (scores≈0) to avoid random slot mixing at init.
        # Gradients still flow into in_proj via h_emb.
        nn.init.zeros_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        self.dropout = nn.Dropout(self.dropout_p)

        # Horizon basis vectors e_h.
        self.h_emb = nn.Parameter(torch.zeros(self.pred_len, self.rank))
        nn.init.normal_(self.h_emb, std=0.02)

        # Slot prior bias (helps encode global preference / scale preference with minimal params).
        self.score_bias = nn.Parameter(torch.zeros(self.k_extra))

    def _norm_u(self, u: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "none":
            return u
        if self.norm_mode == "l2":
            return F.normalize(u, p=2.0, dim=-1, eps=1e-8)
        return self.u_norm(u)

    def _slot_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, T, K-1, H]
        if self.training and self.extra_slot_dropout > 0.0:
            keep = torch.rand_like(scores[..., 0]) > self.extra_slot_dropout  # [B, T, K-1]
            all_dropped = ~keep.any(dim=2, keepdim=True)  # [B, T, 1]
            if all_dropped.any():
                max_idx = scores.mean(dim=-1).argmax(dim=2, keepdim=True)  # [B, T, 1]
                rescue = torch.zeros_like(keep)
                rescue.scatter_(2, max_idx, True)
                keep = keep | (all_dropped.expand_as(keep) & rescue)
            scores = scores.masked_fill(~keep.unsqueeze(-1), -1e9)
        return torch.softmax(scores, dim=2)

    def forward(self, baseline_u: torch.Tensor, extras_u: torch.Tensor) -> torch.Tensor:
        # baseline_u: [B, T, d], extras_u: [B, T, K-1, d]
        if baseline_u.dim() != 3:
            raise ValueError(f"baseline_u must be [B,T,d] (got shape {tuple(baseline_u.shape)})")
        if extras_u.dim() != 4:
            raise ValueError(f"extras_u must be [B,T,K-1,d] (got shape {tuple(extras_u.shape)})")
        bsz, tok, k_extra, dim = extras_u.shape
        if k_extra != self.k_extra:
            raise ValueError(f"Expected k_extra={self.k_extra}, got {k_extra}")
        if dim != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {dim}")

        base = baseline_u.unsqueeze(2).expand_as(extras_u)  # [B, T, K-1, d]
        delta = extras_u - base

        base_n = self._norm_u(base)
        extras_n = self._norm_u(extras_u)
        delta_n = self._norm_u(delta)
        feat = torch.cat([base_n, extras_n, delta_n], dim=-1)  # [B, T, K-1, 3d]

        a = self.in_proj(feat)  # [B, T, K-1, r]
        a = F.gelu(a)
        a = self.dropout(a)

        denom = (float(self.rank) ** 0.5) * self.tau
        # scores: [B, T, K-1, H]
        scores = torch.einsum("btkr,hr->btkh", a, self.h_emb) / denom
        scores = scores + self.score_bias.view(1, 1, -1, 1)

        alpha = self._slot_softmax(scores)  # [B, T, K-1, H]
        return alpha


class SlotSelfAttention(nn.Module):
    """Self-attention over the slot axis K for each variable token independently."""

    def __init__(self, d_model: int, *, n_heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(n_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, K, d]
        bsz, n, k, dim = x.shape
        y = self.ln(x).reshape(bsz * n, k, dim)
        out, _ = self.attn(y, y, y, need_weights=False)
        out = self.dropout(out).reshape(bsz, n, k, dim)
        return x + out


class SlotCondLayerNorm(nn.Module):
    """LayerNorm with slot/scale-conditioned FiLM (gamma, beta).

    y = LN(x)
    y' = (1 + gamma[id]) * y + beta[id]

    Initialization: gamma/beta are zero so the module is identical to the base LayerNorm at start.
    """

    def __init__(self, d_model: int, *, n_ids: int, apply_to: str = "all"):
        super().__init__()
        self.ln = nn.LayerNorm(int(d_model))
        self.gamma = nn.Embedding(int(n_ids), int(d_model))
        self.beta = nn.Embedding(int(n_ids), int(d_model))
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

        apply_to = str(apply_to or "all").lower()
        if apply_to not in ("all", "x_only"):
            raise ValueError(f"apply_to must be one of: all, x_only (got {apply_to})")
        self.apply_to = apply_to

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor | None = None, *, x_tokens: int | None = None) -> torch.Tensor:
        # x: [B*, T, d], cond_idx: [B*]
        y = self.ln(x)
        if cond_idx is None:
            return y

        gamma = self.gamma(cond_idx)  # [B*, d]
        beta = self.beta(cond_idx)  # [B*, d]

        if self.apply_to == "x_only" and x_tokens is not None and x_tokens < y.size(1):
            y_x = y[:, :x_tokens, :] * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            return torch.cat([y_x, y[:, x_tokens:, :]], dim=1)
        return y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class CondEncoderLayer(nn.Module):
    """EncoderLayer variant that supports slot/scale-conditioned LayerNorm."""

    def __init__(
        self,
        attention,
        d_model: int,
        *,
        n_ids: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        cond_apply_to: str = "all",
    ):
        super().__init__()
        d_model = int(d_model)
        d_ff = int(d_ff or 4 * d_model)

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = SlotCondLayerNorm(d_model, n_ids=n_ids, apply_to=cond_apply_to)
        self.norm2 = SlotCondLayerNorm(d_model, n_ids=n_ids, apply_to=cond_apply_to)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask=None,
        *,
        cond_idx: torch.Tensor | None = None,
        x_tokens: int | None = None,
    ):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x, cond_idx, x_tokens=x_tokens)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y, cond_idx, x_tokens=x_tokens), attn


class MultiScaleSlotEmbedding(nn.Module):
    """Multi-scale patchify -> temporal encoder -> PMA slotize (token-wise)."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.model_d_model = int(cfg.model.d_model)
        self.dropout = float(cfg.model.dropout)

        ms = cfg.model.multislot
        # Internal embedding dimension for the slotizer/temporal-conv path.
        self.d_model = int(getattr(ms, "d_model", self.model_d_model))
        if self.d_model <= 0:
            raise ValueError(f"model.multislot.d_model must be >= 1, got {self.d_model}")

        # If slotizer/conv uses a different dimension, project slots to model.d_model for the encoder.
        self.out_proj = None
        if self.d_model != self.model_d_model:
            self.out_proj = nn.Linear(self.d_model, self.model_d_model)

        scales = list(ms.scales)
        k_per_scale = list(ms.k_per_scale)
        if len(scales) != len(k_per_scale):
            raise ValueError("model.multislot.scales and k_per_scale must have the same length")
        if len(scales) == 0:
            raise ValueError("model.multislot.scales must be non-empty")

        self.scales = [int(p) for p in scales]
        self.k_per_scale = [int(k) for k in k_per_scale]
        if any(p <= 0 for p in self.scales):
            raise ValueError(f"Invalid scale in model.multislot.scales: {self.scales}")
        if any(k <= 0 for k in self.k_per_scale):
            raise ValueError(f"Invalid k in model.multislot.k_per_scale: {self.k_per_scale}")

        self.k_total = int(sum(self.k_per_scale))
        self.pad_value = float(getattr(ms, "pad_value", 0.0))

        slotizer_cfg = getattr(ms, "slotizer", None)
        self.slotizer_mode = str(getattr(slotizer_cfg, "mode", "pma") or "pma").lower()
        if self.slotizer_mode not in ("pma", "mean"):
            raise ValueError(f"Unsupported model.multislot.slotizer.mode: {self.slotizer_mode}")
        self.slotizer_share_across_scales = bool(getattr(slotizer_cfg, "share_across_scales", False))
        if self.slotizer_share_across_scales and self.slotizer_mode != "pma":
            raise ValueError("model.multislot.slotizer.share_across_scales=true requires slotizer.mode=pma")
        post_slot_attn_cfg = getattr(slotizer_cfg, "post_slot_attn", None)
        self.post_slot_attn_enabled = bool(getattr(post_slot_attn_cfg, "enabled", False))
        self.post_slot_attn_share_across_scales = bool(
            getattr(post_slot_attn_cfg, "share_across_scales", True)
        )

        div_cfg = getattr(ms, "diversity", None)
        self.div_enabled = bool(getattr(div_cfg, "enabled", False))
        self.div_apply_to = str(getattr(div_cfg, "apply_to", "x_only") or "x_only").lower()
        if self.div_apply_to not in ("all", "x_only"):
            raise ValueError(f"Unsupported model.multislot.diversity.apply_to: {self.div_apply_to}")
        self.div_min_patches = int(getattr(div_cfg, "min_patches", 2) or 2)
        self.div_patch_renorm = bool(getattr(div_cfg, "patch_renorm", True))
        self.div_eps = float(getattr(div_cfg, "eps", 1e-8) or 1e-8)
        self.last_div_loss: torch.Tensor | None = None

        conv_cfg = ms.temporal_conv
        ksize = int(conv_cfg.kernel_size)
        layers = int(conv_cfg.layers)
        dilation = int(conv_cfg.dilation)
        padding_mode = str(getattr(conv_cfg, "padding_mode", "causal") or "causal")
        conv_type = str(getattr(conv_cfg, "type", "standard") or "standard")

        pma_cfg = ms.pma
        pma_heads = int(pma_cfg.n_heads)
        use_ffn = bool(getattr(pma_cfg, "ffn", True))
        kv_include_seeds = bool(getattr(pma_cfg, "kv_include_seeds", False))
        post_slot_attn_heads = int(getattr(post_slot_attn_cfg, "n_heads", pma_heads) or pma_heads)
        if self.post_slot_attn_enabled and self.slotizer_mode != "pma":
            raise ValueError("model.multislot.slotizer.post_slot_attn.enabled=true requires slotizer.mode=pma")

        self.patch_proj = nn.ModuleDict()
        self.pos_emb = nn.ParameterDict()
        self.scale_emb = nn.ParameterDict()
        self.seeds = nn.ParameterDict()
        self.temporal_enc = nn.ModuleDict()
        self.slotizer = nn.ModuleDict()
        self.slotizer_shared = None
        self.post_slot_attn = nn.ModuleDict()
        self.post_slot_attn_shared = None

        if self.slotizer_mode == "pma" and self.slotizer_share_across_scales:
            self.slotizer_shared = PMASlotizer(
                self.d_model,
                n_heads=pma_heads,
                d_ff=int(cfg.model.d_ff),
                dropout=self.dropout,
                use_ffn=use_ffn,
                kv_include_seeds=kv_include_seeds,
            )
        if self.post_slot_attn_enabled and self.post_slot_attn_share_across_scales:
            self.post_slot_attn_shared = SlotSelfAttention(
                self.d_model,
                n_heads=post_slot_attn_heads,
                dropout=self.dropout,
            )

        for p, k in zip(self.scales, self.k_per_scale):
            key = str(p)
            m_p = _ceil_div(self.seq_len, p)
            self.patch_proj[key] = nn.Linear(p, self.d_model)

            pos = nn.Parameter(torch.zeros(m_p, self.d_model))
            nn.init.normal_(pos, std=0.02)
            self.pos_emb[key] = pos

            scale = nn.Parameter(torch.zeros(self.d_model))
            nn.init.normal_(scale, std=0.02)
            self.scale_emb[key] = scale

            if self.slotizer_mode == "pma":
                seeds = nn.Parameter(torch.zeros(k, self.d_model))
                nn.init.normal_(seeds, std=0.02)
                self.seeds[key] = seeds

            self.temporal_enc[key] = CausalConvBlock1D(
                self.d_model,
                kernel_size=ksize,
                layers=layers,
                dilation=dilation,
                dropout=self.dropout,
                padding_mode=padding_mode,
                conv_type=conv_type,
            )
            if self.slotizer_mode == "pma":
                if not self.slotizer_share_across_scales:
                    self.slotizer[key] = PMASlotizer(
                        self.d_model,
                        n_heads=pma_heads,
                        d_ff=int(cfg.model.d_ff),
                        dropout=self.dropout,
                        use_ffn=use_ffn,
                        kv_include_seeds=kv_include_seeds,
                    )
                if self.post_slot_attn_enabled and k > 1 and not self.post_slot_attn_share_across_scales:
                    self.post_slot_attn[key] = SlotSelfAttention(
                        self.d_model,
                        n_heads=post_slot_attn_heads,
                        dropout=self.dropout,
                    )
            else:
                self.slotizer[key] = MeanSlotizer(k)

        self.out_dropout = nn.Dropout(self.dropout)

    def forward(self, x_enc: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        # x_enc: [B, L, N], x_mark: [B, L, C] (optional)
        n_vars = int(x_enc.size(-1))
        x = x_enc.permute(0, 2, 1)  # [B, N, L]
        if x_mark is not None:
            mark = x_mark.permute(0, 2, 1)  # [B, C, L]
            x = torch.cat([x, mark], dim=1)  # [B, T, L]

        bsz, tok, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        div_on = self.div_enabled and self.training and self.slotizer_mode == "pma"
        div_loss_total = x_enc.new_tensor(0.0) if div_on else None
        div_terms = 0

        slots_all = []
        for p, k in zip(self.scales, self.k_per_scale):
            key = str(p)
            l_pad = _ceil_div(seq_len, p) * p
            pad_len = l_pad - seq_len
            if pad_len > 0:
                x_p = F.pad(x, (0, pad_len), value=self.pad_value)
            else:
                x_p = x
            m_p = l_pad // p

            patches = x_p.reshape(bsz, tok, m_p, p)  # [B, T, m, p]
            emb = self.patch_proj[key](patches)  # [B, T, m, d]
            pos = self.pos_emb[key][:m_p].view(1, 1, m_p, self.d_model)
            emb = emb + pos

            emb = self.temporal_enc[key](emb)
            if self.slotizer_mode == "pma":
                slotizer = self.slotizer_shared if self.slotizer_share_across_scales else self.slotizer[key]
                slots, attn_w = slotizer(emb, self.seeds[key], need_weights=div_on)  # [B, T, K_p, d]
                if self.post_slot_attn_enabled and k > 1:
                    if self.post_slot_attn_share_across_scales:
                        slots = self.post_slot_attn_shared(slots)
                    else:
                        slots = self.post_slot_attn[key](slots)
            else:
                slots = self.slotizer[key](emb)  # [B, T, K_p, d]
                attn_w = None

            # Diversity regularizer: encourage different slots to attend to different patches.
            # Only computed during training to avoid overhead in eval.
            if div_on and attn_w is not None and k > 1 and m_p >= self.div_min_patches:
                # attn_w: [B*T, K, S], where S=m (or m+K if kv_include_seeds).
                s_len = int(attn_w.size(-1))
                a = attn_w.view(bsz, tok, k, s_len)
                if self.div_apply_to == "x_only" and n_vars < tok:
                    a = a[:, :n_vars, :, :]
                a = a.reshape(-1, k, s_len)

                # Patch-only attention slice (avoid seed-in-KV "cheating") + optional renorm.
                a = a[:, :, :m_p]
                if self.div_patch_renorm:
                    a = a / (a.sum(dim=-1, keepdim=True) + self.div_eps)

                # Row-wise L2 normalize, then penalize off-diagonal overlap of A A^T.
                a = a / (a.norm(p=2, dim=-1, keepdim=True) + self.div_eps)
                gram = torch.bmm(a, a.transpose(1, 2))  # [B', K, K]
                eye = torch.eye(k, device=gram.device, dtype=gram.dtype).unsqueeze(0)
                off = gram - eye
                denom = float(k * (k - 1))
                div_loss_total = div_loss_total + (off.square().sum(dim=(1, 2)) / denom).mean()
                div_terms += 1

            slots = slots + self.scale_emb[key].view(1, 1, 1, self.d_model)
            slots_all.append(slots)

        z = torch.cat(slots_all, dim=2)  # [B, T, K_total, d]
        if z.size(2) != self.k_total:
            raise ValueError(f"Expected K_total={self.k_total}, got {z.size(2)}")
        z = self.out_dropout(z)
        if self.out_proj is not None:
            z = self.out_proj(z)  # [B, T, K_total, model_d]

        if div_on:
            if div_terms > 0:
                self.last_div_loss = div_loss_total / float(div_terms)
            else:
                self.last_div_loss = x_enc.new_tensor(0.0)
        else:
            self.last_div_loss = None
        return z


class ITransformerM0(nn.Module):
    """iTransformer variant with Multi-scale Latent Slots (M0)."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.pred_len = int(cfg.data.pred_len)
        self.output_attention = False
        self.use_norm = bool(cfg.model.use_norm)

        # Keep the attribute name for downstream freeze rules.
        self.enc_embedding = MultiScaleSlotEmbedding(cfg)
        self.k_total = int(self.enc_embedding.k_total)
        self.last_div_loss: torch.Tensor | None = None

        cov_cfg = getattr(cfg.model.multislot, "covariates", None)
        self.cov_mode = str(getattr(cov_cfg, "mode", "slotize") or "slotize").lower()
        self.cov_embed = None
        if self.cov_mode == "mlp_global":
            hidden = int(getattr(cov_cfg, "mlp_hidden", cfg.model.d_model))
            self.cov_embed = nn.Sequential(
                nn.Linear(self.seq_len, hidden),
                nn.GELU(),
                nn.Linear(hidden, int(cfg.model.d_model)),
                nn.Dropout(float(cfg.model.dropout)),
            )
        elif self.cov_mode != "slotize":
            raise ValueError(f"Unsupported model.multislot.covariates.mode: {self.cov_mode}")

        slot_attn_cfg = getattr(cfg.model.multislot, "slot_attn", None)
        self.slot_attn = None
        self.slot_attn_mode = "post"
        if self.k_total > 1 and bool(getattr(slot_attn_cfg, "enabled", False)):
            mode = str(getattr(slot_attn_cfg, "mode", "post") or "post").lower()
            if mode not in ("post", "interleave", "between"):
                raise ValueError(f"Unsupported model.multislot.slot_attn.mode: {mode}")
            self.slot_attn_mode = mode
            n_heads = int(getattr(slot_attn_cfg, "n_heads", cfg.model.n_heads))
            self.slot_attn = SlotSelfAttention(
                int(cfg.model.d_model),
                n_heads=n_heads,
                dropout=float(cfg.model.dropout),
            )

        cond_ln_cfg = getattr(cfg.model.multislot, "cond_ln", None)
        self.cond_ln_enabled = bool(getattr(cond_ln_cfg, "enabled", False)) and self.k_total > 1
        self.cond_ln_apply_to = str(getattr(cond_ln_cfg, "apply_to", "all") or "all").lower()
        cond_id_mode = str(getattr(cond_ln_cfg, "id", "slot") or "slot").lower()
        if cond_id_mode not in ("slot", "scale"):
            raise ValueError(f"Unsupported model.multislot.cond_ln.id: {cond_id_mode}")
        self.cond_ln_id = cond_id_mode

        self.cond_n_ids = 0
        if self.cond_ln_enabled:
            if self.cond_ln_id == "slot":
                self.cond_n_ids = int(self.k_total)
                cond_ids = torch.arange(self.k_total, dtype=torch.long)
            else:
                n_scales = len(self.enc_embedding.scales)
                self.cond_n_ids = int(n_scales)
                ids = []
                for scale_idx, kk in enumerate(self.enc_embedding.k_per_scale):
                    ids.extend([scale_idx] * int(kk))
                cond_ids = torch.tensor(ids, dtype=torch.long)
                if cond_ids.numel() != self.k_total:
                    raise RuntimeError("scale-id mapping length mismatch")
            self.register_buffer("cond_ids", cond_ids, persistent=False)

        layers = []
        for _ in range(cfg.model.e_layers):
            attn = AttentionLayer(
                FullAttention(
                    mask_flag=False,
                    attention_dropout=cfg.model.dropout,
                    output_attention=False,
                ),
                cfg.model.d_model,
                cfg.model.n_heads,
            )
            if self.cond_ln_enabled:
                layers.append(
                    CondEncoderLayer(
                        attn,
                        cfg.model.d_model,
                        n_ids=self.cond_n_ids,
                        d_ff=cfg.model.d_ff,
                        dropout=cfg.model.dropout,
                        activation=cfg.model.activation,
                        cond_apply_to=self.cond_ln_apply_to,
                    )
                )
            else:
                layers.append(
                    EncoderLayer(
                        attn,
                        cfg.model.d_model,
                        cfg.model.d_ff,
                        dropout=cfg.model.dropout,
                        activation=cfg.model.activation,
                    )
                )
        self.encoder = Encoder(layers, norm_layer=nn.LayerNorm(cfg.model.d_model))

        # Slot fuse strategies:
        # - K_total==1: identity (take slot 0)
        # - mlp: concat + MLP
        # - residual_gated: baseline(p=L) + gated residual corrections in representation space
        # - residual_gated_output: per-slot projectors, then gated residual corrections in output space
        # - scale_flatten_mean: per-scale flatten(K_p*d)->pred heads, then mean over scales
        # - select_global: use p=L slot only (no fuse)
        fuse_cfg = cfg.model.multislot.slot_fuse
        fuse_mode = str(getattr(fuse_cfg, "mode", "mlp") or "mlp").lower()
        self.fuse_mode = fuse_mode
        self.global_slot_idx = None
        self.slot_fuse = None
        self.slot_projectors = None
        self.scale_projectors = None
        self.global_ma_enabled = False
        self.global_ma_kernel = 0
        self.global_ma_combine = "gated_add"
        self.global_ma_tr_proj = None
        self.global_ma_rs_proj = None
        self.global_ma_tr_head = None
        self.global_ma_rs_head = None
        self.global_ma_gate = None
        self.output_hgate_enabled = False
        self.output_hgate_mode = "linear"
        self.output_hgate_init_schedule = "linear"
        self.output_hgate_linear = None
        self.output_hgate_bias = None
        self.output_hgate_bias_per_var = None
        self.output_halpha_enabled = False
        self.output_halpha_hidden = 0
        self.output_halpha_mlp = None
        self.output_alpha_source = "repr"
        self.output_attn_rank = 64
        self.output_attn_use_delta_key = True
        self.output_attn_norm = "layernorm"
        self.output_attn_tau = 1.0
        self.output_attn_scorer = None
        self.hls_rank = 32
        self.hls_norm = "layernorm"
        self.hls_tau = 1.0
        self.output_hls_scorer = None
        fuse_extra_dropout = float(getattr(fuse_cfg, "extra_slot_dropout", 0.0) or 0.0)
        fuse_score_mode = str(getattr(fuse_cfg, "score_mode", "full") or "full").lower()
        fuse_score_rank = int(getattr(fuse_cfg, "score_rank", 0) or 0)
        if fuse_score_mode not in ("full", "lowrank"):
            raise ValueError(
                "model.multislot.slot_fuse.score_mode must be one of: full, lowrank "
                f"(got {fuse_score_mode})"
            )
        self.output_alpha_source = str(getattr(fuse_cfg, "alpha_source", "repr") or "repr").lower()
        if self.output_alpha_source not in ("repr", "output_attn", "hls_lr"):
            raise ValueError(
                "model.multislot.slot_fuse.alpha_source must be one of: repr, output_attn, hls_lr "
                f"(got {self.output_alpha_source})"
            )
        self.output_attn_rank = int(getattr(fuse_cfg, "output_attn_rank", 64) or 64)
        self.output_attn_use_delta_key = bool(getattr(fuse_cfg, "output_attn_use_delta_key", True))
        self.output_attn_norm = str(getattr(fuse_cfg, "output_attn_norm", "layernorm") or "layernorm").lower()
        self.output_attn_tau = float(getattr(fuse_cfg, "output_attn_tau", 1.0) or 1.0)
        self.hls_rank = int(getattr(fuse_cfg, "hls_rank", 32) or 32)
        self.hls_norm = str(getattr(fuse_cfg, "hls_norm", "layernorm") or "layernorm").lower()
        self.hls_tau = float(getattr(fuse_cfg, "hls_tau", 1.0) or 1.0)
        hgate_cfg = getattr(fuse_cfg, "horizon_gate", None)
        self.output_hgate_enabled = bool(getattr(hgate_cfg, "enabled", False))
        self.output_hgate_mode = str(getattr(hgate_cfg, "mode", "linear") or "linear").lower()
        if self.output_hgate_mode not in ("linear", "bias", "bias_per_var"):
            raise ValueError(
                "model.multislot.slot_fuse.horizon_gate.mode must be one of: linear, bias, bias_per_var "
                f"(got {self.output_hgate_mode})"
        )
        self.output_hgate_init_schedule = str(getattr(hgate_cfg, "init_schedule", "linear") or "linear").lower()
        if self.output_hgate_init_schedule not in ("linear", "sigmix_v1"):
            raise ValueError(
                "model.multislot.slot_fuse.horizon_gate.init_schedule must be one of: linear, sigmix_v1 "
                f"(got {self.output_hgate_init_schedule})"
            )
        output_hgate_init_start = float(getattr(hgate_cfg, "init_start", -2.0))
        output_hgate_init_end = float(getattr(hgate_cfg, "init_end", -4.0))
        halpha_cfg = getattr(fuse_cfg, "horizon_alpha", None)
        self.output_halpha_enabled = bool(getattr(halpha_cfg, "enabled", False))
        self.output_halpha_hidden = int(getattr(halpha_cfg, "hidden", 16) or 16)
        if self.output_halpha_enabled and self.output_halpha_hidden <= 0:
            raise ValueError(
                "model.multislot.slot_fuse.horizon_alpha.hidden must be >= 1 "
                f"(got {self.output_halpha_hidden})"
            )
        global_ma_cfg = getattr(fuse_cfg, "global_ma", None)
        self.global_ma_enabled = bool(getattr(global_ma_cfg, "enabled", False))
        self.global_ma_kernel = int(getattr(global_ma_cfg, "kernel_size", 25) or 25)
        self.global_ma_combine = str(getattr(global_ma_cfg, "combine", "gated_add") or "gated_add").lower()
        if self.global_ma_combine not in ("add", "gated_add"):
            raise ValueError(
                f"model.multislot.slot_fuse.global_ma.combine must be one of: add, gated_add (got {self.global_ma_combine})"
            )
        if self.global_ma_kernel <= 0:
            raise ValueError(
                f"model.multislot.slot_fuse.global_ma.kernel_size must be >= 1 (got {self.global_ma_kernel})"
            )
        if self.global_ma_enabled and fuse_mode != "residual_gated_output":
            raise ValueError(
                "model.multislot.slot_fuse.global_ma.enabled=true requires slot_fuse.mode=residual_gated_output"
            )
        if self.output_hgate_enabled and fuse_mode != "residual_gated_output":
            raise ValueError(
                "model.multislot.slot_fuse.horizon_gate.enabled=true requires slot_fuse.mode=residual_gated_output"
            )
        if self.output_halpha_enabled and fuse_mode != "residual_gated_output":
            raise ValueError(
                "model.multislot.slot_fuse.horizon_alpha.enabled=true requires slot_fuse.mode=residual_gated_output"
            )
        if self.output_alpha_source == "output_attn" and fuse_mode != "residual_gated_output":
            raise ValueError(
                "model.multislot.slot_fuse.alpha_source=output_attn requires "
                "slot_fuse.mode=residual_gated_output"
            )
        if self.output_alpha_source == "hls_lr" and fuse_mode != "residual_gated_output":
            raise ValueError(
                "model.multislot.slot_fuse.alpha_source=hls_lr requires "
                "slot_fuse.mode=residual_gated_output"
            )
        if self.output_halpha_enabled and self.output_alpha_source != "repr":
            raise ValueError(
                "model.multislot.slot_fuse.horizon_alpha.enabled=true currently requires "
                "slot_fuse.alpha_source=repr"
            )
        if self.output_alpha_source == "hls_lr" and self.output_halpha_enabled:
            raise ValueError(
                "model.multislot.slot_fuse.alpha_source=hls_lr cannot be combined with "
                "slot_fuse.horizon_alpha.enabled=true (both define horizon-wise alpha)"
            )

        def _find_global_slot_idx() -> int | None:
            offset = 0
            for p, kk in zip(self.enc_embedding.scales, self.enc_embedding.k_per_scale):
                if int(p) == int(self.seq_len):
                    return int(offset)  # first slot of the global (p=L) scale
                offset += int(kk)
            return None

        if fuse_mode == "scale_flatten_mean":
            self.scale_projectors = nn.ModuleList(
                [
                    nn.Linear(int(cfg.model.d_model) * int(kk), int(cfg.data.pred_len), bias=True)
                    for kk in self.enc_embedding.k_per_scale
                ]
            )
        elif self.k_total > 1:
            if fuse_mode == "mlp":
                self.slot_fuse = SlotFuseMLP(
                    int(cfg.model.d_model),
                    k_total=self.k_total,
                    hidden=int(fuse_cfg.hidden),
                    dropout=float(cfg.model.dropout),
                )
            elif fuse_mode == "residual_gated":
                baseline_idx = _find_global_slot_idx()
                if baseline_idx is None:
                    baseline_idx = self.k_total - 1
                self.slot_fuse = ResidualGatedFuse(
                    int(cfg.model.d_model),
                    k_total=self.k_total,
                    baseline_idx=baseline_idx,
                    hidden=int(fuse_cfg.hidden),
                    dropout=float(cfg.model.dropout),
                    extra_slot_dropout=fuse_extra_dropout,
                    score_mode=fuse_score_mode,
                    score_rank=fuse_score_rank,
                )
            elif fuse_mode == "residual_gated_output":
                baseline_idx = _find_global_slot_idx()
                if baseline_idx is None:
                    if self.global_ma_enabled:
                        raise ValueError(
                            "global_ma decomposition requires a global scale p=seq_len in model.multislot.scales"
                        )
                    baseline_idx = self.k_total - 1
                # Only build the heavy repr-space scorer when it is actually used.
                build_repr_scorer = (self.output_alpha_source == "repr") and (not self.output_halpha_enabled)
                self.slot_fuse = ResidualGatedFuse(
                    int(cfg.model.d_model),
                    k_total=self.k_total,
                    baseline_idx=baseline_idx,
                    hidden=int(fuse_cfg.hidden),
                    dropout=float(cfg.model.dropout),
                    extra_slot_dropout=fuse_extra_dropout,
                    score_mode=fuse_score_mode,
                    score_rank=fuse_score_rank,
                    build_scorer=build_repr_scorer,
                )
                self.slot_projectors = nn.ModuleList(
                    [nn.Linear(int(cfg.model.d_model), int(cfg.data.pred_len), bias=True) for _ in range(self.k_total)]
                )
                if self.output_halpha_enabled:
                    self.output_halpha_mlp = nn.Sequential(
                        nn.Linear(3, self.output_halpha_hidden),
                        nn.GELU(),
                        nn.Dropout(float(cfg.model.dropout)),
                        nn.Linear(self.output_halpha_hidden, 1),
                    )
                    # Start from uniform alpha (all scores=0) to avoid random mixing at init.
                    for mod in self.output_halpha_mlp.modules():
                        if isinstance(mod, nn.Linear):
                            nn.init.zeros_(mod.weight)
                            if mod.bias is not None:
                                nn.init.zeros_(mod.bias)
                if self.output_alpha_source == "output_attn":
                    self.output_attn_scorer = OutputAttentionScorer(
                        pred_len=int(cfg.data.pred_len),
                        k_extra=self.k_total - 1,
                        rank=self.output_attn_rank,
                        use_delta_key=self.output_attn_use_delta_key,
                        norm=self.output_attn_norm,
                        tau=self.output_attn_tau,
                        extra_slot_dropout=fuse_extra_dropout,
                    )
                if self.output_alpha_source == "hls_lr":
                    self.output_hls_scorer = HorizonLowRankScorer(
                        d_model=int(cfg.model.d_model),
                        pred_len=int(cfg.data.pred_len),
                        k_extra=self.k_total - 1,
                        rank=self.hls_rank,
                        dropout=float(cfg.model.dropout),
                        norm=self.hls_norm,
                        tau=self.hls_tau,
                        extra_slot_dropout=fuse_extra_dropout,
                    )
                if self.output_hgate_enabled:
                    pred_len = int(cfg.data.pred_len)
                    hbias = _make_hgate_bias_schedule(
                        pred_len=pred_len,
                        init_start=output_hgate_init_start,
                        init_end=output_hgate_init_end,
                        schedule=self.output_hgate_init_schedule,
                    )
                    if self.output_hgate_mode == "linear":
                        self.output_hgate_linear = nn.Linear(int(cfg.model.d_model), pred_len)
                        nn.init.zeros_(self.output_hgate_linear.weight)
                        with torch.no_grad():
                            self.output_hgate_linear.bias.copy_(hbias)
                    elif self.output_hgate_mode == "bias":
                        self.output_hgate_bias = nn.Parameter(hbias)
                    else:
                        n_gate_vars = int(getattr(cfg.data, "enc_in", 0) or 0)
                        if n_gate_vars <= 0:
                            raise ValueError(
                                "model.multislot.slot_fuse.horizon_gate.mode=bias_per_var requires "
                                "data.enc_in >= 1"
                            )
                        self.output_hgate_bias_per_var = nn.Parameter(
                            hbias.unsqueeze(0).repeat(n_gate_vars, 1)
                        )
                if self.global_ma_enabled:
                    d_model = int(cfg.model.d_model)
                    pred_len = int(cfg.data.pred_len)
                    self.global_ma_tr_proj = nn.Linear(self.seq_len, d_model)
                    self.global_ma_rs_proj = nn.Linear(self.seq_len, d_model)
                    self.global_ma_tr_head = nn.Linear(d_model, pred_len)
                    self.global_ma_rs_head = nn.Linear(d_model, pred_len)
                    if self.global_ma_combine == "gated_add":
                        self.global_ma_gate = nn.Linear(d_model, 1)
                        nn.init.constant_(self.global_ma_gate.bias, -2.0)
            elif fuse_mode == "select_global":
                self.global_slot_idx = _find_global_slot_idx()
                if self.global_slot_idx is None:
                    raise ValueError(
                        "slot_fuse.mode=select_global requires a global scale p=seq_len in model.multislot.scales"
                    )
            else:
                raise ValueError(f"Unsupported model.multislot.slot_fuse.mode: {fuse_mode}")
        self.projector = nn.Linear(int(cfg.model.d_model), int(cfg.data.pred_len), bias=True)

    def get_diversity_loss(self) -> torch.Tensor | None:
        """Return the most recent diversity regularizer term (if enabled)."""
        return self.last_div_loss

    def _apply_interleave_slot_attn(
        self,
        tokens_flat: torch.Tensor,
        *,
        bsz: int,
        tok_total: int,
        k_total: int,
        x_tokens: int,
    ) -> torch.Tensor:
        # tokens_flat: [B*K, T_total, d]
        # Apply slot-attn only to the x-variable tokens to avoid turning global covariate
        # tokens into slot-specific ones (e.g., when covariates.mode=mlp_global).
        if self.slot_attn is None:
            return tokens_flat

        dim = tokens_flat.size(-1)
        z = tokens_flat.reshape(bsz, k_total, tok_total, dim).permute(0, 2, 1, 3)  # [B, T_total, K, d]
        z_x = self.slot_attn(z[:, :x_tokens, :, :])  # [B, N, K, d]
        if x_tokens < tok_total:
            z = torch.cat([z_x, z[:, x_tokens:, :, :]], dim=1)
        else:
            z = z_x
        return z.permute(0, 2, 1, 3).reshape(bsz * k_total, tok_total, dim)  # [B*K, T_total, d]

    def _encode_tokens(
        self,
        tokens_flat: torch.Tensor,
        *,
        bsz: int,
        tok_total: int,
        k_total: int,
        x_tokens: int,
        cond_idx: torch.Tensor | None = None,
    ):
        if not self.cond_ln_enabled and (self.slot_attn is None or self.slot_attn_mode == "post"):
            return self.encoder(tokens_flat, attn_mask=None)

        x = tokens_flat
        attns = []
        last_idx = len(self.encoder.attn_layers) - 1
        for i, attn_layer in enumerate(self.encoder.attn_layers):
            if self.cond_ln_enabled:
                # CondEncoderLayer supports slot/scale-conditioned LayerNorm.
                x, attn = attn_layer(
                    x,
                    attn_mask=None,
                    cond_idx=cond_idx,
                    x_tokens=x_tokens if self.cond_ln_apply_to == "x_only" else None,
                )
            else:
                x, attn = attn_layer(x, attn_mask=None)
            attns.append(attn)
            if self.slot_attn_mode == "interleave" or (self.slot_attn_mode == "between" and i != last_idx):
                x = self._apply_interleave_slot_attn(
                    x,
                    bsz=bsz,
                    tok_total=tok_total,
                    k_total=k_total,
                    x_tokens=x_tokens,
                )
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x, attns

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape
        if self.cov_mode == "slotize":
            z = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, K, d]
            self.last_div_loss = getattr(self.enc_embedding, "last_div_loss", None)
            bsz, tok, k_total, dim = z.shape
            z_flat = z.permute(0, 2, 1, 3).reshape(bsz * k_total, tok, dim)
            cond_idx = self.cond_ids.repeat(bsz) if self.cond_ln_enabled else None
            enc_flat, attns = self._encode_tokens(
                z_flat,
                bsz=bsz,
                tok_total=tok,
                k_total=k_total,
                x_tokens=n_vars,
                cond_idx=cond_idx,
            )  # [B*K, T, d]
            enc_out = enc_flat.reshape(bsz, k_total, tok, dim).permute(0, 2, 1, 3)  # [B, T, K, d]
            enc_x = enc_out[:, :n_vars, :, :]  # [B, N, K, d]
        else:
            # Covariates are embedded with a dedicated MLP (no patchify/conv/PMA),
            # then mixed with each k-th slot group via encoder self-attention.
            z = self.enc_embedding(x_enc, None)  # [B, N, K, d]
            self.last_div_loss = getattr(self.enc_embedding, "last_div_loss", None)
            bsz, tok, k_total, dim = z.shape
            x_flat = z.permute(0, 2, 1, 3).reshape(bsz * k_total, tok, dim)  # [B*K, N, d]

            if x_mark_enc is None:
                tokens_flat = x_flat
                cov_vars = 0
            else:
                mark = x_mark_enc.permute(0, 2, 1)  # [B, C, L]
                cov = self.cov_embed(mark.reshape(-1, self.seq_len)).reshape(bsz, -1, dim)  # [B, C, d]
                cov_vars = cov.size(1)
                cov_rep = cov.unsqueeze(1).expand(bsz, k_total, cov_vars, dim).reshape(
                    bsz * k_total, cov_vars, dim
                )
                tokens_flat = torch.cat([x_flat, cov_rep], dim=1)  # [B*K, N+C, d]

            cond_idx = self.cond_ids.repeat(bsz) if self.cond_ln_enabled else None
            enc_flat, attns = self._encode_tokens(
                tokens_flat,
                bsz=bsz,
                tok_total=tokens_flat.size(1),
                k_total=k_total,
                x_tokens=tok,
                cond_idx=cond_idx,
            )  # [B*K, N(+C), d]
            enc_flat_x = enc_flat[:, :tok, :]  # keep x vars only
            enc_x = enc_flat_x.reshape(bsz, k_total, tok, dim).permute(0, 2, 1, 3)  # [B, N, K, d]

        if self.slot_attn is not None and self.slot_attn_mode == "post":
            enc_x = self.slot_attn(enc_x)  # [B, N, K, d]

        if self.fuse_mode == "select_global":
            if self.global_slot_idx is None:
                raise RuntimeError("global_slot_idx is not set for select_global mode")
            fused = enc_x[:, :, self.global_slot_idx, :]  # [B, N, d]
            dec_out = self.projector(fused).permute(0, 2, 1)[:, :, :n_vars]
        elif self.fuse_mode == "scale_flatten_mean":
            if self.scale_projectors is None:
                raise RuntimeError("scale_projectors are not initialized for scale_flatten_mean")

            preds = []
            start = 0
            for kk, head in zip(self.enc_embedding.k_per_scale, self.scale_projectors):
                end = start + int(kk)
                slots_s = enc_x[:, :, start:end, :]  # [B, N, K_p, d]
                flat = slots_s.contiguous().reshape(
                    slots_s.size(0), slots_s.size(1), int(kk) * slots_s.size(-1)
                )  # [B, N, K_p*d]
                preds.append(head(flat))  # [B, N, pred_len]
                start = end
            pred_mean = torch.stack(preds, dim=2).mean(dim=2)  # [B, N, pred_len]
            dec_out = pred_mean.permute(0, 2, 1)[:, :, :n_vars]
        elif self.fuse_mode == "residual_gated_output":
            if self.slot_projectors is None or self.slot_fuse is None:
                raise RuntimeError("slot projectors / slot_fuse are not initialized for residual_gated_output")

            preds = [head(enc_x[:, :, i, :]) for i, head in enumerate(self.slot_projectors)]
            pred_slots = torch.stack(preds, dim=2)  # [B, N, K, pred_len]
            baseline_idx = int(self.slot_fuse.baseline_idx)
            if self.global_ma_enabled:
                if (
                    self.global_ma_tr_proj is None
                    or self.global_ma_rs_proj is None
                    or self.global_ma_tr_head is None
                    or self.global_ma_rs_head is None
                ):
                    raise RuntimeError("global MA modules are not initialized")
                trend = _moving_average_1d(x_enc, self.global_ma_kernel)  # [B, L, N]
                resid = x_enc - trend  # [B, L, N]
                u_tr = self.global_ma_tr_proj(trend.permute(0, 2, 1))  # [B, N, d]
                u_rs = self.global_ma_rs_proj(resid.permute(0, 2, 1))  # [B, N, d]
                y_tr = self.global_ma_tr_head(u_tr)  # [B, N, pred_len]
                y_rs = self.global_ma_rs_head(u_rs)  # [B, N, pred_len]
                if self.global_ma_combine == "gated_add":
                    if self.global_ma_gate is None:
                        raise RuntimeError("global MA gate is not initialized")
                    g0 = torch.sigmoid(self.global_ma_gate(enc_x[:, :, baseline_idx, :]))  # [B, N, 1]
                    baseline = y_tr + g0 * y_rs
                else:
                    baseline = y_tr + y_rs
            else:
                baseline = pred_slots[:, :, baseline_idx, :]  # [B, N, pred_len]
            extras = torch.cat(
                [pred_slots[:, :, :baseline_idx, :], pred_slots[:, :, baseline_idx + 1 :, :]],
                dim=2,
            )  # [B, N, K-1, pred_len]
            delta = extras - baseline.unsqueeze(2)  # [B, N, K-1, pred_len]

            if self.output_halpha_enabled:
                if self.output_halpha_mlp is None:
                    raise RuntimeError("output horizon alpha MLP is not initialized")
                base = baseline.unsqueeze(2).expand_as(extras)  # [B, N, K-1, pred_len]
                dlt = extras - base  # [B, N, K-1, pred_len]
                feat = torch.stack([base, extras, dlt], dim=-1)  # [B, N, K-1, pred_len, 3]
                scores = self.output_halpha_mlp(feat).squeeze(-1)  # [B, N, K-1, pred_len]
                extra_p = float(getattr(self.slot_fuse, "extra_slot_dropout", 0.0))
                if self.training and extra_p > 0.0:
                    keep = torch.rand_like(scores[..., 0]) > extra_p  # [B, N, K-1]
                    all_dropped = ~keep.any(dim=2, keepdim=True)  # [B, N, 1]
                    if all_dropped.any():
                        max_idx = scores.mean(dim=-1).argmax(dim=2, keepdim=True)  # [B, N, 1]
                        rescue = torch.zeros_like(keep)
                        rescue.scatter_(2, max_idx, True)
                        keep = keep | (all_dropped.expand_as(keep) & rescue)
                    masked_scores = scores.masked_fill(~keep.unsqueeze(-1), -1e9)
                    alpha = torch.softmax(masked_scores, dim=2)  # [B, N, K-1, pred_len]
                else:
                    alpha = torch.softmax(scores, dim=2)  # [B, N, K-1, pred_len]
            else:
                if self.output_alpha_source == "output_attn":
                    if self.output_attn_scorer is None:
                        raise RuntimeError("output attention scorer is not initialized")
                    alpha = self.output_attn_scorer(baseline, extras)  # [B, N, K-1, 1]
                    g = torch.sigmoid(self.slot_fuse.gate(enc_x[:, :, baseline_idx, :]))  # [B, N, 1]
                elif self.output_alpha_source == "hls_lr":
                    if self.output_hls_scorer is None:
                        raise RuntimeError("HLSG scorer is not initialized")
                    base_u = enc_x[:, :, baseline_idx, :]  # [B, N, d]
                    extras_u = torch.cat(
                        [enc_x[:, :, :baseline_idx, :], enc_x[:, :, baseline_idx + 1 :, :]],
                        dim=2,
                    )  # [B, N, K-1, d]
                    alpha = self.output_hls_scorer(base_u, extras_u)  # [B, N, K-1, pred_len]
                    g = torch.sigmoid(self.slot_fuse.gate(base_u))  # [B, N, 1]
                else:
                    alpha, g = self.slot_fuse.compute_alpha_g(enc_x)  # [B, N, K-1, 1], [B, N, 1]

            corr = (alpha * delta).sum(dim=2)  # [B, N, pred_len]
            if self.output_hgate_enabled:
                if self.output_hgate_mode == "linear":
                    if self.output_hgate_linear is None:
                        raise RuntimeError("horizon gate linear is not initialized")
                    g_h = torch.sigmoid(
                        self.output_hgate_linear(enc_x[:, :, baseline_idx, :])
                    )  # [B, N, pred_len]
                elif self.output_hgate_mode == "bias":
                    if self.output_hgate_bias is None:
                        raise RuntimeError("horizon gate bias is not initialized")
                    g_h = torch.sigmoid(self.output_hgate_bias).view(1, 1, -1)  # [1, 1, pred_len]
                else:
                    if self.output_hgate_bias_per_var is None:
                        raise RuntimeError("horizon gate per-variable bias is not initialized")
                    if n_vars > self.output_hgate_bias_per_var.size(0):
                        raise RuntimeError(
                            "horizon gate per-variable bias has fewer variables than input "
                            f"(got gate_vars={self.output_hgate_bias_per_var.size(0)}, input_vars={n_vars})"
                        )
                    g_h = torch.sigmoid(self.output_hgate_bias_per_var[:n_vars, :]).unsqueeze(0)
                fused_pred = baseline + g_h * corr  # [B, N, pred_len]
            else:
                if self.output_halpha_enabled:
                    g = torch.sigmoid(self.slot_fuse.gate(enc_x[:, :, baseline_idx, :]))  # [B, N, 1]
                fused_pred = baseline + g * corr  # [B, N, pred_len]
            dec_out = fused_pred.permute(0, 2, 1)[:, :, :n_vars]
        elif self.slot_fuse is None:
            fused = enc_x[:, :, 0, :]  # [B, N, d]
            dec_out = self.projector(fused).permute(0, 2, 1)[:, :, :n_vars]
        else:
            fused = self.slot_fuse(enc_x)  # [B, N, d]
            dec_out = self.projector(fused).permute(0, 2, 1)[:, :, :n_vars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None):
        dec_out, _ = self.forecast(x_enc, x_mark_enc, meta_emb=meta_emb)
        return dec_out[:, -self.pred_len :, :]
