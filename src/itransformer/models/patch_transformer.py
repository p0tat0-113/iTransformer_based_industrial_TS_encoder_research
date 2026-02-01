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
        self.meta_enabled = bool(getattr(cfg.model.meta, "enabled", False))
        self.meta_mode = getattr(cfg.model.meta, "mode", "none")

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
        self.projector_patch = nn.Linear(
            self.patch_count * cfg.model.d_model,
            cfg.data.pred_len,
            bias=True,
        )

    def _apply_meta(self, enc_out, meta_emb, *, repeat_patches: bool):
        if meta_emb is None or not self.meta_enabled or self.meta_mode == "none":
            return enc_out
        if meta_emb.dim() == 2:
            meta_emb = meta_emb.unsqueeze(0).expand(enc_out.size(0), -1, -1)
        if repeat_patches:
            meta_emb = meta_emb.repeat_interleave(self.patch_count, dim=1)
        meta_emb = self.meta_proj(meta_emb)
        if self.meta_mode == "add":
            return enc_out + meta_emb
        if self.meta_mode in ("concat", "fusion"):
            fused = torch.cat([enc_out, meta_emb], dim=-1)
            return self.meta_fuse(fused)
        return enc_out

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
        time_emb = None
        time_vars = 0
        if x_mark is not None:
            mark_trim = x_mark[:, :usable_len, :]
            mark_patches = mark_trim.reshape(bsz, self.patch_count, self.patch_len, -1)
            mark_patches = mark_patches.permute(0, 1, 3, 2)  # [B, P, T, patch_len]
            time_emb = self.patch_embed(mark_patches)  # [B, P, T, E]
            time_vars = time_emb.size(2)

        if self.patch_mode == "mean_pool":
            tokens = patch_emb.mean(dim=1)  # [B, N, E]
            tokens = self._apply_meta(tokens, meta_emb, repeat_patches=False)
            if time_emb is not None:
                time_tokens = time_emb.mean(dim=1)  # [B, T, E]
                tokens = torch.cat([tokens, time_tokens], dim=1)
            attn_mask = None
        else:
            total_vars = n_vars + time_vars
            data_tokens = patch_emb.reshape(bsz, self.patch_count * n_vars, -1)
            data_tokens = self._apply_meta(data_tokens, meta_emb, repeat_patches=True)
            data_tokens = data_tokens.reshape(bsz, self.patch_count, n_vars, -1)
            if time_emb is not None:
                combined = torch.cat([data_tokens, time_emb], dim=2)  # [B, P, N+T, E]
                tokens = combined.reshape(bsz, self.patch_count * total_vars, -1)
            else:
                tokens = data_tokens.reshape(bsz, self.patch_count * n_vars, -1)
            attn_mask = build_patch_attn_mask(
                self.patch_count,
                total_vars,
                self.patch_mode,
                self.local_win,
                x_enc.device,
            )

        enc_out, _ = self.encoder(tokens, attn_mask=attn_mask)
        if self.patch_mode == "mean_pool":
            enc_out = enc_out[:, :n_vars, :]
            pooled = enc_out  # [B, N, E]
            dec_out = self.projector(pooled).permute(0, 2, 1)
        else:
            if time_emb is not None:
                enc_out = enc_out.reshape(bsz, self.patch_count, total_vars, -1)[:, :, :n_vars, :]
            else:
                enc_out = enc_out.reshape(bsz, self.patch_count, n_vars, -1)
            # concatenate patches in temporal order for each variable
            enc_out = enc_out.permute(0, 2, 1, 3).reshape(
                bsz, n_vars, self.patch_count * enc_out.size(-1)
            )
            dec_out = self.projector_patch(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
