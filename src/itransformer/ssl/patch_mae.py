import torch
import torch.nn as nn

from itransformer.models.layers.attention import AttentionLayer, FullAttention
from itransformer.models.layers.embed import DataEmbeddingInverted
from itransformer.models.layers.transformer import Encoder, EncoderLayer


class PatchMAE(nn.Module):
    """Minimal patch-masking autoencoder for pretraining."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.data.seq_len
        self.patch_len = max(1, int(cfg.ssl.patch_len))
        self.mask_ratio = cfg.ssl.mask_ratio
        self.patch_mode = getattr(cfg.model.patch, "mode", None) or getattr(cfg.ssl, "patch_mode", "all")
        self.local_win = int(getattr(cfg.model.patch, "local_win", 0) or getattr(cfg.ssl, "local_win", 0))
        self.use_norm = cfg.model.use_norm
        self.meta_enabled = bool(getattr(cfg.model.meta, "enabled", False))
        self.meta_mode = getattr(cfg.model.meta, "mode", "none")

        self.patch_count = self.seq_len // self.patch_len
        if self.patch_count <= 0:
            raise ValueError("patch_len must be <= seq_len")

        self.value_proj = nn.Linear(1, cfg.model.d_model)

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
        self.projector = nn.Linear(cfg.model.d_model, 1, bias=True)

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
        patch_mean = patches.mean(dim=2)  # [B, P, N]

        mask = torch.rand(bsz, self.patch_count, device=x_enc.device) < self.mask_ratio
        patch_masked = patch_mean.masked_fill(mask.unsqueeze(-1), 0.0)

        # tokenization: order by patch then variate
        tokens = patch_masked.permute(0, 1, 2).reshape(bsz, self.patch_count * n_vars, 1)
        enc_out = self.value_proj(tokens)
        enc_out = self._apply_meta(enc_out, meta_emb)
        attn_mask = None
        if self.patch_mode in ("same_time", "local"):
            device = x_enc.device
            token_count = self.patch_count * n_vars
            idx = torch.arange(token_count, device=device)
            patch_idx = idx // n_vars
            if self.patch_mode == "same_time":
                mask_mat = patch_idx[:, None] != patch_idx[None, :]
            else:
                win = max(0, self.local_win)
                mask_mat = (patch_idx[:, None] - patch_idx[None, :]).abs() > win
            attn_mask = mask_mat.unsqueeze(0).unsqueeze(0)
        enc_out, _ = self.encoder(enc_out, attn_mask=attn_mask)
        recon_tokens = self.projector(enc_out).reshape(bsz, self.patch_count, n_vars)

        if self.use_norm:
            recon_tokens = recon_tokens * stdev[:, 0, :].unsqueeze(1).repeat(1, self.patch_count, 1)
            recon_tokens = recon_tokens + means[:, 0, :].unsqueeze(1).repeat(1, self.patch_count, 1)

        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum() * n_vars
        if denom.item() == 0:
            denom = torch.tensor(1.0, device=x_enc.device)
        mse = ((recon_tokens - patch_mean) ** 2 * mask_f).sum() / denom
        mae = (torch.abs(recon_tokens - patch_mean) * mask_f).sum() / denom
        loss = mse
        if return_details:
            return loss, mse, mae
        return loss
