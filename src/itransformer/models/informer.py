from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from itransformer.models.layers.attention import AttentionLayer, FullAttention


def _time_mark_dim_from_cfg(cfg, fallback: int = 4) -> int:
    icfg = getattr(cfg.model, "informer", None)
    mark_dim_override = int(getattr(icfg, "mark_dim", 0) or 0)
    if mark_dim_override > 0:
        return mark_dim_override
    timeenc = int(getattr(cfg.data, "timeenc", 0) or 0)
    freq = str(getattr(cfg.data, "freq", "h") or "h").lower()
    if timeenc == 0:
        return 5 if freq == "t" else 4
    freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
    return int(freq_map.get(freq, fallback))


def _align_mark_dim(x_mark: torch.Tensor, mark_dim: int) -> torch.Tensor:
    cur = int(x_mark.size(-1))
    if cur == mark_dim:
        return x_mark
    if cur > mark_dim:
        return x_mark[..., :mark_dim]
    pad = torch.zeros(
        x_mark.size(0),
        x_mark.size(1),
        mark_dim - cur,
        dtype=x_mark.dtype,
        device=x_mark.device,
    )
    return torch.cat([x_mark, pad], dim=-1)


class TriangularCausalMask:
    def __init__(self, batch: int, length: int, device):
        self.mask = torch.triu(
            torch.ones(batch, 1, length, length, dtype=torch.bool, device=device), diagonal=1
        )


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, L, D]
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class TimeLinearEmbedding(nn.Module):
    def __init__(self, mark_dim: int, d_model: int):
        super().__init__()
        self.mark_dim = int(mark_dim)
        self.proj = nn.Linear(self.mark_dim, d_model, bias=False)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        x = _align_mark_dim(x_mark, self.mark_dim)
        return self.proj(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, mark_dim: int, dropout: float):
        super().__init__()
        self.value = TokenEmbedding(c_in, d_model)
        self.pos = PositionalEmbedding(d_model)
        self.temporal = TimeLinearEmbedding(mark_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        out = self.value(x) + self.pos(x)
        if x_mark is not None:
            out = out + self.temporal(x_mark)
        return self.dropout(out)


class ConvDistillLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        y = self.norm(y)
        y = self.act(y)
        y = self.pool(y)
        return y.transpose(1, 2)


class ProbAttention(nn.Module):
    """
    ProbSparse attention used by Informer.
    Interface is compatible with AttentionLayer / FullAttention:
    queries/keys/values: [B, L, H, E] -> out: [B, L, H, E]
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = int(factor)
        self.scale = scale
        self.mask_flag = bool(mask_flag)
        self.output_attention = bool(output_attention)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        bsz, l_q, n_heads, dim = queries.shape
        _, l_k, _, _ = keys.shape

        q = queries.transpose(1, 2)  # [B, H, Lq, D]
        k = keys.transpose(1, 2)  # [B, H, Lk, D]
        v = values.transpose(1, 2)  # [B, H, Lk, D]

        sample_k = min(l_k, max(1, int(self.factor * math.ceil(math.log(l_k + 1)))))
        n_top = min(l_q, max(1, int(self.factor * math.ceil(math.log(l_q + 1)))))

        scores_top, top_idx = self._prob_qk(q, k, sample_k=sample_k, n_top=n_top)
        scale = self.scale or (1.0 / math.sqrt(dim))
        scores_top = scores_top * scale

        context = self._initial_context(v, l_q=l_q)
        context, attn = self._update_context(
            context,
            v,
            scores_top=scores_top,
            top_idx=top_idx,
            l_q=l_q,
            attn_mask=attn_mask,
        )

        out = context.transpose(1, 2).contiguous()  # [B, Lq, H, D]
        return out, attn

    def _prob_qk(self, q: torch.Tensor, k: torch.Tensor, *, sample_k: int, n_top: int):
        # q: [B, H, Lq, D], k: [B, H, Lk, D]
        bsz, n_heads, l_q, dim = q.shape
        _, _, l_k, _ = k.shape

        k_expand = k.unsqueeze(-3).expand(bsz, n_heads, l_q, l_k, dim)
        index_sample = torch.randint(l_k, (l_q, sample_k), device=q.device)
        q_pos = torch.arange(l_q, device=q.device).unsqueeze(1)
        k_sample = k_expand[:, :, q_pos, index_sample, :]  # [B,H,Lq,sample_k,D]

        q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)  # [B,H,Lq,sample_k]
        sparsity = q_k_sample.max(dim=-1).values - q_k_sample.mean(dim=-1)  # [B,H,Lq]
        top_idx = sparsity.topk(n_top, dim=-1, sorted=False).indices  # [B,H,n_top]

        q_top = q.gather(dim=2, index=top_idx.unsqueeze(-1).expand(-1, -1, -1, dim))  # [B,H,n_top,D]
        q_k = torch.matmul(q_top, k.transpose(-2, -1))  # [B,H,n_top,Lk]
        return q_k, top_idx

    def _initial_context(self, v: torch.Tensor, *, l_q: int) -> torch.Tensor:
        # v: [B, H, Lv, D]
        bsz, n_heads, l_v, dim = v.shape
        if not self.mask_flag:
            v_mean = v.mean(dim=-2)
            return v_mean.unsqueeze(-2).expand(bsz, n_heads, l_q, dim).clone()

        if l_q != l_v:
            raise ValueError(
                f"ProbAttention(masked) expects Lq==Lk for self-attention, got Lq={l_q}, Lk={l_v}"
            )
        return v.cumsum(dim=-2)

    def _update_context(
        self,
        context: torch.Tensor,
        v: torch.Tensor,
        *,
        scores_top: torch.Tensor,
        top_idx: torch.Tensor,
        l_q: int,
        attn_mask: torch.Tensor | None,
    ):
        # context/v: [B,H,L,D], scores_top: [B,H,n_top,Lk], top_idx: [B,H,n_top]
        bsz, n_heads, l_v, _ = v.shape
        scores = scores_top

        if self.mask_flag:
            mask_top = self._build_top_mask(
                bsz=bsz,
                n_heads=n_heads,
                l_q=l_q,
                l_k=l_v,
                top_idx=top_idx,
                scores_shape=scores.shape,
                device=v.device,
                attn_mask=attn_mask,
            )
            scores = scores.masked_fill(mask_top, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))  # [B,H,n_top,Lk]
        updates = torch.matmul(attn, v).type_as(context)  # [B,H,n_top,D]

        context[
            torch.arange(bsz, device=v.device)[:, None, None],
            torch.arange(n_heads, device=v.device)[None, :, None],
            top_idx,
            :,
        ] = updates

        if not self.output_attention:
            return context, None

        dense_attn = torch.full(
            (bsz, n_heads, l_q, l_v),
            fill_value=(1.0 / float(l_v)),
            dtype=attn.dtype,
            device=attn.device,
        )
        dense_attn[
            torch.arange(bsz, device=v.device)[:, None, None],
            torch.arange(n_heads, device=v.device)[None, :, None],
            top_idx,
            :,
        ] = attn
        return context, dense_attn

    @staticmethod
    def _build_top_mask(
        *,
        bsz: int,
        n_heads: int,
        l_q: int,
        l_k: int,
        top_idx: torch.Tensor,
        scores_shape: torch.Size,
        device,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attn_mask is None:
            key_pos = torch.arange(l_k, device=device).view(1, 1, 1, l_k)
            q_pos = top_idx.unsqueeze(-1)
            return key_pos > q_pos

        mask = getattr(attn_mask, "mask", attn_mask)
        if mask.dim() != 4:
            raise ValueError(f"attn_mask must be rank-4 [B,1/H,Lq,Lk], got {tuple(mask.shape)}")
        if mask.size(0) != bsz or mask.size(-2) != l_q or mask.size(-1) != l_k:
            raise ValueError(
                "attn_mask shape mismatch: "
                f"expected [{bsz},1|{n_heads},{l_q},{l_k}], got {tuple(mask.shape)}"
            )

        if mask.size(1) == 1 and n_heads > 1:
            mask = mask.expand(bsz, n_heads, l_q, l_k)
        elif mask.size(1) != n_heads:
            raise ValueError(
                "attn_mask head dimension mismatch: "
                f"expected 1 or {n_heads}, got {mask.size(1)}"
            )

        gathered = mask[
            torch.arange(bsz, device=device)[:, None, None],
            torch.arange(n_heads, device=device)[None, :, None],
            top_idx,
            :,
        ]
        return gathered.view(scores_shape)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if str(activation).lower() == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None and len(self.conv_layers) > 0:
            for layer, conv in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = layer(x, attn_mask=attn_mask)
                x = conv(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for layer in self.attn_layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if str(activation).lower() == "gelu" else F.relu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_sa, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = self.norm1(x + self.dropout(x_sa))

        x_ca, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        y = self.norm2(x + self.dropout(x_ca))

        y_ff = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(1, 2))
        return self.norm3(y + y_ff)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class Informer(nn.Module):
    needs_y_mark_dec = True

    def __init__(self, cfg):
        super().__init__()
        self.seq_len = int(cfg.data.seq_len)
        self.label_len = int(cfg.data.label_len)
        self.pred_len = int(cfg.data.pred_len)
        self.enc_in = int(cfg.data.enc_in)
        self.c_out = int(cfg.data.c_out)
        self.use_norm = bool(getattr(cfg.model, "use_norm", True))

        d_model = int(cfg.model.d_model)
        n_heads = int(cfg.model.n_heads)
        e_layers = int(cfg.model.e_layers)
        d_layers = int(getattr(cfg.model, "d_layers", 1) or 1)
        d_ff = int(cfg.model.d_ff)
        dropout = float(cfg.model.dropout)
        activation = str(getattr(cfg.model, "activation", "gelu"))

        icfg = getattr(cfg.model, "informer", None)
        attn_type = str(getattr(icfg, "attn", "prob") or "prob").lower()
        factor = int(getattr(icfg, "factor", getattr(cfg.model, "factor", 5)) or 5)
        distil = bool(getattr(icfg, "distil", True))
        output_attention = bool(getattr(icfg, "output_attention", False))
        mark_dim = _time_mark_dim_from_cfg(cfg)

        Attn = ProbAttention if attn_type == "prob" else FullAttention
        if attn_type not in ("prob", "full"):
            raise ValueError(f"informer.attn must be one of: prob, full (got {attn_type})")

        self.enc_embedding = DataEmbedding(self.enc_in, d_model, mark_dim=mark_dim, dropout=dropout)
        self.dec_embedding = DataEmbedding(self.enc_in, d_model, mark_dim=mark_dim, dropout=dropout)

        enc_layers = []
        for _ in range(e_layers):
            inner = (
                Attn(mask_flag=False, factor=factor, attention_dropout=dropout, output_attention=output_attention)
                if Attn is ProbAttention
                else Attn(mask_flag=False, attention_dropout=dropout, output_attention=output_attention)
            )
            enc_layers.append(
                EncoderLayer(
                    AttentionLayer(inner, d_model=d_model, n_heads=n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            )
        conv_layers = [ConvDistillLayer(d_model) for _ in range(e_layers - 1)] if distil and e_layers > 1 else None
        self.encoder = Encoder(enc_layers, conv_layers=conv_layers, norm_layer=nn.LayerNorm(d_model))

        dec_layers = []
        for _ in range(d_layers):
            self_inner = (
                Attn(mask_flag=True, factor=factor, attention_dropout=dropout, output_attention=False)
                if Attn is ProbAttention
                else Attn(mask_flag=True, attention_dropout=dropout, output_attention=False)
            )
            cross_inner = FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False)
            dec_layers.append(
                DecoderLayer(
                    AttentionLayer(self_inner, d_model=d_model, n_heads=n_heads),
                    AttentionLayer(cross_inner, d_model=d_model, n_heads=n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.decoder = Decoder(
            dec_layers,
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True),
        )

    def _build_dec_inp(self, x_enc: torch.Tensor) -> torch.Tensor:
        # [B, label_len + pred_len, N]
        zeros = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(2), dtype=x_enc.dtype, device=x_enc.device)
        return torch.cat([x_enc[:, -self.label_len :, :], zeros], dim=1)

    def _build_mark_dec(self, x_mark_enc: torch.Tensor | None, y_mark_dec: torch.Tensor | None):
        if y_mark_dec is not None:
            needed = self.label_len + self.pred_len
            if y_mark_dec.size(1) >= needed:
                return y_mark_dec[:, -needed:, :]
            pad = torch.zeros(
                y_mark_dec.size(0),
                needed - y_mark_dec.size(1),
                y_mark_dec.size(2),
                dtype=y_mark_dec.dtype,
                device=y_mark_dec.device,
            )
            return torch.cat([pad, y_mark_dec], dim=1)

        if x_mark_enc is None:
            return None
        hist = x_mark_enc[:, -self.label_len :, :]
        future = hist[:, -1:, :].repeat(1, self.pred_len, 1)
        return torch.cat([hist, future], dim=1)

    def forecast(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc

        x_dec = self._build_dec_inp(x)
        x_mark_dec = self._build_mark_dec(x_mark_enc, y_mark_dec)

        enc_out = self.enc_embedding(x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_mask = TriangularCausalMask(x.size(0), dec_out.size(1), device=x.device).mask
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_mask, cross_mask=None)
        out = dec_out[:, -self.pred_len :, :]

        if self.use_norm:
            if out.size(-1) != stdev.size(-1):
                raise ValueError(
                    "Informer denorm channel mismatch: "
                    f"output_channels={out.size(-1)}, norm_channels={stdev.size(-1)}. "
                    "Set data.c_out==data.enc_in or disable model.use_norm."
                )
            out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out, attns

    def forward(self, x_enc, x_mark_enc=None, meta_emb=None, y_mark_dec=None, **kwargs):
        y, _ = self.forecast(
            x_enc,
            x_mark_enc=x_mark_enc,
            meta_emb=meta_emb,
            y_mark_dec=y_mark_dec,
            **kwargs,
        )
        return y
