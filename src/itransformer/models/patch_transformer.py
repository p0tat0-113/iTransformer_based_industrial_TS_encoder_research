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
        if patch_len <= 0:
            raise ValueError("model.patch.patch_len must be set for patch modes")
        self.patch_len = patch_len
        self.patch_count = self.seq_len // self.patch_len
        if self.patch_count <= 0:
            raise ValueError("patch_len must be <= seq_len")

        self.patch_embed = nn.Linear(self.patch_len, cfg.model.d_model)
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
        patches = patches.permute(0, 1, 3, 2)  # [B, P, N, patch_len]
        patch_emb = self.patch_embed(patches)  # [B, P, N, E]

        if self.patch_mode == "mean_pool":
            tokens = patch_emb.mean(dim=1)  # [B, N, E]
            attn_mask = None
        else:
            tokens = patch_emb.reshape(bsz, self.patch_count * n_vars, -1)
            attn_mask = build_patch_attn_mask(
                self.patch_count,
                n_vars,
                self.patch_mode,
                self.local_win,
                x_enc.device,
            )

        enc_out, _ = self.encoder(tokens, attn_mask=attn_mask)
        if self.patch_mode != "mean_pool":
            enc_out = enc_out.reshape(bsz, self.patch_count, n_vars, -1).mean(dim=1)
        pooled = enc_out  # [B, N, E]
        dec_out = self.projector(pooled).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
