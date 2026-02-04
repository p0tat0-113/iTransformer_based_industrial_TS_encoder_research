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

    def __init__(self, d_model: int, *, n_heads: int, d_ff: int, dropout: float, use_ffn: bool):
        super().__init__()
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

    def forward(self, x: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        # x: [B, T, m, d]
        # seeds: [K, d]
        bsz, tok, m, dim = x.shape
        x_flat = x.reshape(bsz * tok, m, dim)
        q = seeds.unsqueeze(0).expand(x_flat.size(0), -1, -1)  # [B*T, K, d]
        out, _ = self.attn(q, x_flat, x_flat, need_weights=False)
        out = self.dropout(out)
        if self.use_ffn:
            out = out + self.ffn(self.ffn_ln(out))
        return out.reshape(bsz, tok, out.size(1), dim)  # [B, T, K, d]


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


class MultiScaleSlotEmbedding(nn.Module):
    """Multi-scale patchify -> temporal encoder -> PMA slotize (token-wise)."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.d_model = int(cfg.model.d_model)
        self.dropout = float(cfg.model.dropout)

        ms = cfg.model.multislot
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

        conv_cfg = ms.temporal_conv
        ksize = int(conv_cfg.kernel_size)
        layers = int(conv_cfg.layers)
        dilation = int(conv_cfg.dilation)

        pma_cfg = ms.pma
        pma_heads = int(pma_cfg.n_heads)
        use_ffn = bool(getattr(pma_cfg, "ffn", True))

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
            self.slotizer[key] = PMASlotizer(
                self.d_model,
                n_heads=pma_heads,
                d_ff=int(cfg.model.d_ff),
                dropout=self.dropout,
                use_ffn=use_ffn,
            )

        self.out_dropout = nn.Dropout(self.dropout)

    def forward(self, x_enc: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        # x_enc: [B, L, N], x_mark: [B, L, C] (optional)
        x = x_enc.permute(0, 2, 1)  # [B, N, L]
        if x_mark is not None:
            mark = x_mark.permute(0, 2, 1)  # [B, C, L]
            x = torch.cat([x, mark], dim=1)  # [B, T, L]

        bsz, tok, seq_len = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

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
            slots = self.slotizer[key](emb, self.seeds[key])  # [B, T, K_p, d]
            slots = slots + self.scale_emb[key].view(1, 1, 1, self.d_model)
            slots_all.append(slots)

        z = torch.cat(slots_all, dim=2)  # [B, T, K_total, d]
        if z.size(2) != self.k_total:
            raise ValueError(f"Expected K_total={self.k_total}, got {z.size(2)}")
        return self.out_dropout(z)


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

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=cfg.model.dropout,
                            output_attention=False,
                        ),
                        cfg.model.d_model,
                        cfg.model.n_heads,
                    ),
                    cfg.model.d_model,
                    cfg.model.d_ff,
                    dropout=cfg.model.dropout,
                    activation=cfg.model.activation,
                )
                for _ in range(cfg.model.e_layers)
            ],
            norm_layer=nn.LayerNorm(cfg.model.d_model),
        )

        self.slot_fuse = SlotFuseMLP(
            int(cfg.model.d_model),
            k_total=self.enc_embedding.k_total,
            hidden=int(cfg.model.multislot.slot_fuse.hidden),
            dropout=float(cfg.model.dropout),
        )
        self.projector = nn.Linear(int(cfg.model.d_model), int(cfg.data.pred_len), bias=True)

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape
        z = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, K, d]
        bsz, tok, k_total, dim = z.shape

        z_flat = z.permute(0, 2, 1, 3).reshape(bsz * k_total, tok, dim)
        enc_out, attns = self.encoder(z_flat, attn_mask=None)
        enc_out = enc_out.reshape(bsz, k_total, tok, dim).permute(0, 2, 1, 3)  # [B, T, K, d]

        fused = self.slot_fuse(enc_out)  # [B, T, d]
        dec_out = self.projector(fused).permute(0, 2, 1)[:, :, :n_vars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None):
        dec_out, _ = self.forecast(x_enc, x_mark_enc, meta_emb=meta_emb)
        return dec_out[:, -self.pred_len :, :]
