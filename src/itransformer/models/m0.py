import torch
import torch.nn as nn
import torch.nn.functional as F

from itransformer.models.layers.attention import FullAttention, AttentionLayer
from itransformer.models.layers.transformer import Encoder, EncoderLayer


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class CausalConvBlock1D(nn.Module):
    """Causal conv over the patch-index axis (m_p) without changing length."""

    def __init__(self, d_model: int, *, kernel_size: int, layers: int, dilation: int, dropout: float):
        super().__init__()
        if layers <= 0:
            raise ValueError("temporal_conv.layers must be >= 1")
        if kernel_size <= 0:
            raise ValueError("temporal_conv.kernel_size must be >= 1")
        if dilation <= 0:
            raise ValueError("temporal_conv.dilation must be >= 1")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.ln = nn.LayerNorm(d_model)
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, m, d] (token-independent temporal encoding)
        bsz, tok, m, dim = x.shape
        if m <= 1:
            return x

        y = self.ln(x)
        y = y.reshape(bsz * tok, m, dim).transpose(1, 2)  # [B*T, d, m]

        pad = (self.kernel_size - 1) * self.dilation
        for i, conv in enumerate(self.convs):
            y = F.pad(y, (pad, 0))  # left pad only (causal)
            y = conv(y)
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


class ResidualGatedFuse(nn.Module):
    """Baseline (p=L) + gated residual corrections from other slots.

    correction-form + global gate:
        delta_i = extra_i - baseline
        alpha = softmax(score_i)  (over extras)
        g = sigmoid(Linear(baseline))  (scalar per token)
        fused = baseline + g * sum_i alpha_i * delta_i
    """

    def __init__(self, d_model: int, *, k_total: int, baseline_idx: int, hidden: int, dropout: float):
        super().__init__()
        self.k_total = int(k_total)
        self.baseline_idx = int(baseline_idx)
        if not (0 <= self.baseline_idx < self.k_total):
            raise ValueError(f"baseline_idx must be in [0,{self.k_total-1}], got {self.baseline_idx}")
        self.k_extra = self.k_total - 1
        if self.k_extra <= 0:
            raise ValueError("ResidualGatedFuse requires k_total >= 2")

        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError("hidden must be >= 1")
        dropout = float(dropout)

        # score_i = MLP([u_L ; extra_i ; (extra_i - u_L)]) -> scalar
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * int(d_model), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.gate = nn.Linear(int(d_model), 1)
        nn.init.constant_(self.gate.bias, -2.0)

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
        feat = torch.cat([base, extras, delta], dim=-1)  # [B, T, K-1, 3d]
        scores = self.score_mlp(feat).squeeze(-1)  # [B, T, K-1]
        alpha = torch.softmax(scores, dim=2).unsqueeze(-1)  # [B, T, K-1, 1]
        corr = (alpha * delta).sum(dim=2)  # [B, T, d]
        g = torch.sigmoid(self.gate(baseline))  # [B, T, 1]
        return baseline + g * corr


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

        pma_cfg = ms.pma
        pma_heads = int(pma_cfg.n_heads)
        use_ffn = bool(getattr(pma_cfg, "ffn", True))
        kv_include_seeds = bool(getattr(pma_cfg, "kv_include_seeds", False))

        self.patch_proj = nn.ModuleDict()
        self.pos_emb = nn.ParameterDict()
        self.scale_emb = nn.ParameterDict()
        self.seeds = nn.ParameterDict()
        self.temporal_enc = nn.ModuleDict()
        self.slotizer = nn.ModuleDict()

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
            )
            if self.slotizer_mode == "pma":
                self.slotizer[key] = PMASlotizer(
                    self.d_model,
                    n_heads=pma_heads,
                    d_ff=int(cfg.model.d_ff),
                    dropout=self.dropout,
                    use_ffn=use_ffn,
                    kv_include_seeds=kv_include_seeds,
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
                slots, attn_w = self.slotizer[key](emb, self.seeds[key], need_weights=div_on)  # [B, T, K_p, d]
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
        # - residual_gated: baseline(p=L) + gated residual corrections from other slots
        # - select_global: use p=L slot only (no fuse)
        fuse_cfg = cfg.model.multislot.slot_fuse
        fuse_mode = str(getattr(fuse_cfg, "mode", "mlp") or "mlp").lower()
        self.fuse_mode = fuse_mode
        self.global_slot_idx = None
        self.slot_fuse = None

        def _find_global_slot_idx() -> int | None:
            offset = 0
            for p, kk in zip(self.enc_embedding.scales, self.enc_embedding.k_per_scale):
                if int(p) == int(self.seq_len):
                    return int(offset)  # first slot of the global (p=L) scale
                offset += int(kk)
            return None

        if self.k_total > 1:
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
                )
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
        elif self.slot_fuse is None:
            fused = enc_x[:, :, 0, :]  # [B, N, d]
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
