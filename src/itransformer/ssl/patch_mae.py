import torch
import torch.nn as nn

from itransformer.models.layers.attention import AttentionLayer, FullAttention
from itransformer.models.layers.transformer import Encoder, EncoderLayer


class PatchMAE(nn.Module):
    """Minimal patch-masking autoencoder for pretraining."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.data.seq_len
        patch_len = int(getattr(cfg.model.patch, "patch_len", 0) or cfg.ssl.patch_len)
        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        self.patch_len = patch_len
        self.mask_ratio = cfg.ssl.mask_ratio
        self.patch_mode = getattr(cfg.model.patch, "mode", None) or getattr(cfg.ssl, "patch_mode", "all")
        self.local_win = int(getattr(cfg.model.patch, "local_win", 0) or getattr(cfg.ssl, "local_win", 0))
        self.use_norm = cfg.model.use_norm
        self.meta_enabled = bool(getattr(cfg.model.meta, "enabled", False))
        self.meta_mode = getattr(cfg.model.meta, "mode", "none")

        self.patch_count = self.seq_len // self.patch_len
        if self.patch_count <= 0:
            raise ValueError("patch_len must be <= seq_len")

        self.patch_embed = nn.Linear(self.patch_len, cfg.model.d_model)

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
        self.projector = nn.Linear(cfg.model.d_model, self.patch_len, bias=True)

    def _apply_meta(self, enc_out, meta_emb):
        if meta_emb is None or not self.meta_enabled or self.meta_mode == "none":
            return enc_out
        if meta_emb.dim() == 2:
            # meta is per-sensor; repeat across patch_count
            meta_emb = meta_emb.unsqueeze(0).expand(enc_out.size(0), -1, -1)
            meta_emb = meta_emb.repeat_interleave(self.patch_count, dim=1)
        meta_emb = self.meta_proj(meta_emb)
        if self.meta_mode == "add":
            return enc_out + meta_emb
        if self.meta_mode in ("concat", "fusion"):
            fused = torch.cat([enc_out, meta_emb], dim=-1)
            return self.meta_fuse(fused)
        return enc_out

    def forward(self, x_enc, x_mark=None, meta_emb=None, return_details: bool = False):
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

        time_emb = None
        time_vars = 0
        if x_mark is not None:
            mark_trim = x_mark[:, :usable_len, :]
            mark_patches = mark_trim.reshape(bsz, self.patch_count, self.patch_len, -1)
            mark_patches = mark_patches.permute(0, 1, 3, 2)  # [B, P, T, patch_len]
            time_emb = self.patch_embed(mark_patches)  # [B, P, T, E]
            time_vars = time_emb.size(2)

        mask = torch.rand(bsz, self.patch_count, n_vars, device=x_enc.device) < self.mask_ratio
        patch_emb = self.patch_embed(patches).masked_fill(mask.unsqueeze(-1), 0.0)

        data_tokens = patch_emb.reshape(bsz, self.patch_count * n_vars, -1)
        data_tokens = self._apply_meta(data_tokens, meta_emb)
        data_tokens = data_tokens.reshape(bsz, self.patch_count, n_vars, -1)

        total_vars = n_vars + time_vars
        if time_emb is not None:
            combined = torch.cat([data_tokens, time_emb], dim=2)  # [B, P, N+T, E]
        else:
            combined = data_tokens
        enc_out = combined.reshape(bsz, self.patch_count * total_vars, -1)
        attn_mask = None
        if self.patch_mode in ("same_time", "local"):
            device = x_enc.device
            token_count = self.patch_count * total_vars
            idx = torch.arange(token_count, device=device)
            patch_idx = idx // total_vars
            if self.patch_mode == "same_time":
                mask_mat = patch_idx[:, None] != patch_idx[None, :]
            else:
                win = max(0, self.local_win)
                mask_mat = (patch_idx[:, None] - patch_idx[None, :]).abs() > win
            attn_mask = mask_mat.unsqueeze(0).unsqueeze(0)
        enc_out, _ = self.encoder(enc_out, attn_mask=attn_mask)
        if time_emb is not None:
            enc_out = enc_out.reshape(bsz, self.patch_count, total_vars, -1)[:, :, :n_vars, :]
        else:
            enc_out = enc_out.reshape(bsz, self.patch_count, n_vars, -1)
        recon = self.projector(enc_out)  # [B, P, N, patch_len]

        target = patches
        if self.use_norm:
            scale = stdev[:, 0, :].unsqueeze(1).unsqueeze(-1)
            bias = means[:, 0, :].unsqueeze(1).unsqueeze(-1)
            recon = recon * scale + bias
            target = target * scale + bias

        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum() * self.patch_len
        if denom.item() == 0:
            denom = torch.tensor(1.0, device=x_enc.device)
        mse = ((recon - target) ** 2 * mask_f).sum() / denom
        mae = (torch.abs(recon - target) * mask_f).sum() / denom
        loss = mse
        if return_details:
            return loss, mse, mae
        return loss
