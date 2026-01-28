import torch
import torch.nn as nn

from itransformer.models.layers.attention import FullAttention, AttentionLayer
from itransformer.models.layers.transformer import Encoder, EncoderLayer
from itransformer.models.layers.embed import DataEmbeddingInverted


class ITransformer(nn.Module):
    """Encoder-only iTransformer with optional metadata fusion."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.data.seq_len
        self.pred_len = cfg.data.pred_len
        self.output_attention = False
        self.use_norm = cfg.model.use_norm

        self.meta_enabled = bool(getattr(cfg.model.meta, "enabled", False))
        self.meta_mode = getattr(cfg.model.meta, "mode", "none")

        self.enc_embedding = DataEmbeddingInverted(
            cfg.data.seq_len,
            cfg.model.d_model,
            dropout=cfg.model.dropout,
        )

        if self.meta_enabled:
            meta_dim = cfg.metadata.embedding.dim
            proj = getattr(cfg.model.meta, "proj", "linear")
            if proj == "mlp":
                hidden = getattr(cfg.model.meta, "mlp_hidden", cfg.model.d_model)
                self.meta_proj = nn.Sequential(
                    nn.Linear(meta_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, cfg.model.d_model),
                )
            else:
                self.meta_proj = nn.Linear(meta_dim, cfg.model.d_model)

            if self.meta_mode == "concat":
                self.meta_fuse = nn.Linear(cfg.model.d_model * 2, cfg.model.d_model)
            elif self.meta_mode == "fusion":
                hidden = getattr(cfg.model.meta, "mlp_hidden", cfg.model.d_model)
                self.meta_fuse = nn.Sequential(
                    nn.Linear(cfg.model.d_model * 2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, cfg.model.d_model),
                )
            else:
                self.meta_fuse = None

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
        self.projector = nn.Linear(cfg.model.d_model, cfg.data.pred_len, bias=True)

    def _apply_meta(self, enc_out, meta_emb):
        if meta_emb is None or not self.meta_enabled or self.meta_mode == "none":
            return enc_out

        # meta_emb: [N, M] or [B, N, M]
        if meta_emb.dim() == 2:
            meta_emb = meta_emb.unsqueeze(0).expand(enc_out.size(0), -1, -1)

        meta_emb = self.meta_proj(meta_emb)

        if self.meta_mode == "add":
            return enc_out + meta_emb
        if self.meta_mode in ("concat", "fusion"):
            fused = torch.cat([enc_out, meta_emb], dim=-1)
            return self.meta_fuse(fused)
        return enc_out

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_meta(enc_out, meta_emb)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :n_vars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None):
        dec_out, _ = self.forecast(x_enc, x_mark_enc, meta_emb=meta_emb)
        return dec_out[:, -self.pred_len:, :]
