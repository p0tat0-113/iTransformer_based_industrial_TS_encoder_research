import torch
import torch.nn as nn

from itransformer.models.layers.attention import AttentionLayer, FullAttention
from itransformer.models.layers.embed import DataEmbeddingInverted
from itransformer.models.layers.transformer import Encoder, EncoderLayer
from itransformer.models.patch_utils import build_patch_attn_mask


class VarMAE(nn.Module):
    """Minimal variate-masking autoencoder for pretraining."""

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.data.seq_len
        self.mask_ratio = cfg.ssl.mask_ratio
        self.use_norm = cfg.model.use_norm
        self.meta_enabled = bool(getattr(cfg.model.meta, "enabled", False))
        self.meta_mode = getattr(cfg.model.meta, "mode", "none")
        self.tokenizer = getattr(cfg.ssl, "tokenizer", "auto")
        self.mask_axis = getattr(cfg.ssl, "mask_axis", "var")

        patch_enabled = bool(getattr(cfg.model.patch, "enabled", False)) or cfg.model.variant in {
            "P1",
            "P2",
            "P3",
            "P4",
        }
        if self.tokenizer == "auto":
            self.use_patch = patch_enabled
        elif self.tokenizer == "patch":
            self.use_patch = True
        elif self.tokenizer in ("var", "variate"):
            self.use_patch = False
        else:
            raise ValueError(f"Unsupported ssl.tokenizer: {self.tokenizer}")

        if self.use_patch:
            self.patch_mode = getattr(cfg.model.patch, "mode", None) or getattr(cfg.ssl, "patch_mode", "all")
            self.local_win = int(getattr(cfg.model.patch, "local_win", 0) or getattr(cfg.ssl, "local_win", 0))
            patch_len = int(getattr(cfg.model.patch, "patch_len", 0) or 0)
            if patch_len <= 0:
                patch_len = int(getattr(cfg.ssl, "patch_len", 0) or 0)
            if self.patch_mode == "mean_pool":
                patch_len = self.seq_len
            if patch_len <= 0:
                raise ValueError("patch_len must be set for patch tokenizer")
            self.patch_len = patch_len
            self.patch_count = self.seq_len // self.patch_len
            if self.patch_count <= 0:
                raise ValueError("patch_len must be <= seq_len")
            self.value_proj = nn.Linear(1, cfg.model.d_model)
        else:
            self.patch_mode = "none"
            self.local_win = 0
            self.patch_len = 0
            self.patch_count = 0
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
        self.projector = nn.Linear(cfg.model.d_model, cfg.data.seq_len, bias=True)

    def _apply_meta(self, enc_out, meta_emb, n_vars: int | None = None):
        if meta_emb is None or not self.meta_enabled or self.meta_mode == "none":
            return enc_out
        if meta_emb.dim() == 2:
            meta_emb = meta_emb.unsqueeze(0).expand(enc_out.size(0), -1, -1)

        if self.use_patch:
            if n_vars is None:
                n_vars = enc_out.size(1)
            if meta_emb.size(1) == n_vars:
                meta_emb = meta_emb.unsqueeze(1).expand(-1, self.patch_count, -1, -1)
                meta_emb = meta_emb.reshape(enc_out.size(0), self.patch_count * n_vars, -1)
            elif meta_emb.size(1) != enc_out.size(1):
                if meta_emb.size(1) < enc_out.size(1):
                    pad_len = enc_out.size(1) - meta_emb.size(1)
                    pad = torch.zeros(
                        meta_emb.size(0),
                        pad_len,
                        meta_emb.size(2),
                        device=meta_emb.device,
                        dtype=meta_emb.dtype,
                    )
                    meta_emb = torch.cat([meta_emb, pad], dim=1)
                else:
                    meta_emb = meta_emb[:, : enc_out.size(1), :]
        else:
            if meta_emb.size(1) != enc_out.size(1):
                if meta_emb.size(1) < enc_out.size(1):
                    pad_len = enc_out.size(1) - meta_emb.size(1)
                    pad = torch.zeros(
                        meta_emb.size(0),
                        pad_len,
                        meta_emb.size(2),
                        device=meta_emb.device,
                        dtype=meta_emb.dtype,
                    )
                    meta_emb = torch.cat([meta_emb, pad], dim=1)
                else:
                    meta_emb = meta_emb[:, : enc_out.size(1), :]
        meta_emb = self.meta_proj(meta_emb)
        if self.meta_mode == "add":
            return enc_out + meta_emb
        if self.meta_mode in ("concat", "fusion"):
            fused = torch.cat([enc_out, meta_emb], dim=-1)
            return self.meta_fuse(fused)
        return enc_out

    def forward(self, x_enc, x_mark=None, meta_emb=None, return_details: bool = False):
        # x_enc: [B, L, N]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        bsz, _, n_vars = x_enc.shape

        if not self.use_patch:
            mask_var = torch.rand(bsz, n_vars, device=x_enc.device) < self.mask_ratio
            x_masked = x_enc.masked_fill(mask_var.unsqueeze(1), 0.0)

            enc_out = self.enc_embedding(x_masked, x_mark)
            enc_out = self._apply_meta(enc_out, meta_emb)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            recon = self.projector(enc_out).permute(0, 2, 1)[:, :, :n_vars]
            usable_len = x_enc.size(1)
        else:
            usable_len = self.patch_count * self.patch_len
            x_trim = x_enc[:, :usable_len, :]
            patches = x_trim.reshape(bsz, self.patch_count, self.patch_len, n_vars)
            patch_mean = patches.mean(dim=2)  # [B, P, N]

            if self.mask_axis == "token":
                mask_tokens = torch.rand(bsz, self.patch_count, n_vars, device=x_enc.device) < self.mask_ratio
                patch_masked = patch_mean.masked_fill(mask_tokens, 0.0)
                mask_var = mask_tokens.any(dim=1)
            else:
                mask_var = torch.rand(bsz, n_vars, device=x_enc.device) < self.mask_ratio
                patch_masked = patch_mean.masked_fill(mask_var.unsqueeze(1), 0.0)

            tokens = patch_masked.reshape(bsz, self.patch_count * n_vars, 1)
            enc_out = self.value_proj(tokens)
            enc_out = self._apply_meta(enc_out, meta_emb, n_vars=n_vars)
            attn_mask = build_patch_attn_mask(
                self.patch_count,
                n_vars,
                self.patch_mode,
                self.local_win,
                x_enc.device,
            )
            enc_out, _ = self.encoder(enc_out, attn_mask=attn_mask)
            enc_out = enc_out.reshape(bsz, self.patch_count, n_vars, -1).mean(dim=1)
            recon = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            recon = recon * stdev[:, 0, :].unsqueeze(1).repeat(1, recon.size(1), 1)
            recon = recon + means[:, 0, :].unsqueeze(1).repeat(1, recon.size(1), 1)

        # MSE on masked variates only
        mask_f = mask_var.unsqueeze(1).float()
        denom = mask_f.sum() * usable_len
        if denom.item() == 0:
            denom = torch.tensor(1.0, device=x_enc.device)
        target = x_enc[:, :usable_len, :]
        pred = recon[:, :usable_len, :]
        mse = ((pred - target) ** 2 * mask_f).sum() / denom
        mae = (torch.abs(pred - target) * mask_f).sum() / denom
        loss = mse
        if return_details:
            return loss, mse, mae
        return loss
