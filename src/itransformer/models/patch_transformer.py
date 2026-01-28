import torch
import torch.nn as nn

from itransformer.models.layers.attention import AttentionLayer, FullAttention
from itransformer.models.layers.transformer import Encoder, EncoderLayer
from itransformer.models.patch_utils import build_patch_attn_mask


class PatchITransformer(nn.Module):
    """Patch-based downstream forecaster (patch tokens)."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.data.seq_len
        self.pred_len = cfg.data.pred_len
        self.use_norm = cfg.model.use_norm

        self.patch_mode = cfg.model.patch.mode
        self.local_win = cfg.model.patch.local_win
        patch_len = int(cfg.model.patch.patch_len)
        if self.patch_mode == "mean_pool":
            patch_len = self.seq_len
        if patch_len <= 0:
            raise ValueError("model.patch.patch_len must be set for patch modes")
        self.patch_len = patch_len
        self.patch_count = self.seq_len // self.patch_len
        if self.patch_count <= 0:
            raise ValueError("patch_len must be <= seq_len")

        self.value_proj = nn.Linear(1, cfg.model.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=self.patch_mode in ("same_time", "local"),
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

    def forward(self, x_enc, x_mark=None, meta_emb=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        bsz, _, n_vars = x_enc.shape
        usable_len = self.patch_count * self.patch_len
        x_enc = x_enc[:, :usable_len, :]
        patches = x_enc.reshape(bsz, self.patch_count, self.patch_len, n_vars)
        patch_mean = patches.mean(dim=2)  # [B, P, N]

        tokens = patch_mean.reshape(bsz, self.patch_count * n_vars, 1)
        enc_out = self.value_proj(tokens)

        attn_mask = build_patch_attn_mask(
            self.patch_count,
            n_vars,
            self.patch_mode,
            self.local_win,
            x_enc.device,
        )
        enc_out, _ = self.encoder(enc_out, attn_mask=attn_mask)

        enc_out = enc_out.reshape(bsz, self.patch_count, n_vars, -1)
        pooled = enc_out.mean(dim=1)  # [B, N, E]
        dec_out = self.projector(pooled).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
