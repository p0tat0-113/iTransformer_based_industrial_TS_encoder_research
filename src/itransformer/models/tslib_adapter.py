from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


def _repo_root_from_cfg(cfg) -> str:
    raw = str(getattr(cfg.model.tslib, "repo_root", "") or "").strip()
    if not raw:
        raise ValueError("model.tslib.repo_root is required")

    if os.path.isabs(raw):
        return raw

    # Resolve workspace-relative paths robustly from src/itransformer/models.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return os.path.abspath(os.path.join(project_root, raw))


def _ensure_tslib_import_path(repo_root: str) -> None:
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"TSLib repo root not found: {repo_root}")
    models_dir = os.path.join(repo_root, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"TSLib models dir not found: {models_dir}")

    # TSLib modules expect top-level imports like `from layers...`, `from models...`.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _build_tslib_args(cfg) -> SimpleNamespace:
    tcfg = cfg.model.tslib

    def _int(name: str, default: int) -> int:
        return int(getattr(tcfg, name, default) or default)

    def _str(name: str, default: str) -> str:
        return str(getattr(tcfg, name, default) or default)

    def _bool(name: str, default: bool) -> bool:
        return bool(getattr(tcfg, name, default))

    args = SimpleNamespace()

    # Core task/data/model fields expected by TSLib models.
    args.task_name = _str("task_name", "long_term_forecast")
    args.model = _str("model_name", "iTransformer")
    args.seq_len = int(cfg.data.seq_len)
    args.label_len = int(cfg.data.label_len)
    args.pred_len = int(cfg.data.pred_len)
    args.features = str(cfg.data.features)
    args.target = str(cfg.data.target)
    args.freq = str(cfg.data.freq)

    enc_in = int(cfg.data.enc_in)
    args.enc_in = enc_in
    args.dec_in = int(getattr(cfg.data, "dec_in", enc_in) or enc_in)
    args.c_out = int(getattr(cfg.data, "c_out", enc_in) or enc_in)

    args.d_model = int(cfg.model.d_model)
    args.n_heads = int(cfg.model.n_heads)
    args.e_layers = int(cfg.model.e_layers)
    args.d_layers = _int("d_layers", 1)
    args.d_ff = int(cfg.model.d_ff)
    args.dropout = float(cfg.model.dropout)
    args.activation = str(cfg.model.activation)
    args.use_norm = 1 if bool(cfg.model.use_norm) else 0

    # Frequently required extras from TSLib run defaults.
    args.factor = _int("factor", 1)
    args.embed = _str("embed", "timeF")
    args.output_attention = False
    args.distil = True
    args.num_class = _int("num_class", 1)
    args.moving_avg = _int("moving_avg", 25)
    args.top_k = _int("top_k", 5)
    args.num_kernels = _int("num_kernels", 6)
    args.down_sampling_layers = _int("down_sampling_layers", 1)
    args.down_sampling_window = _int("down_sampling_window", 1)
    args.down_sampling_method = _str("down_sampling_method", "avg")
    args.channel_independence = _int("channel_independence", 1)
    args.decomp_method = _str("decomp_method", "moving_avg")
    args.expand = _int("expand", 2)
    args.d_conv = _int("d_conv", 4)
    args.individual = _bool("individual", False)

    return args


class TSLibForecastAdapter(nn.Module):
    """Adapter that runs a TSLib long-term forecasting model in this codebase."""

    def __init__(self, cfg):
        super().__init__()
        self.pred_len = int(cfg.data.pred_len)
        self.label_len = int(cfg.data.label_len)
        self.model_name = str(cfg.model.tslib.model_name)

        repo_root = _repo_root_from_cfg(cfg)
        _ensure_tslib_import_path(repo_root)

        args = _build_tslib_args(cfg)
        module = importlib.import_module(f"models.{self.model_name}")
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            raise AttributeError(
                f"TSLib module models.{self.model_name} does not define class Model"
            )

        if self.model_name == "PatchTST":
            patch_len = int(getattr(cfg.model.tslib, "patch_len", 16) or 16)
            stride = int(getattr(cfg.model.tslib, "stride", 8) or 8)
            self.model = model_cls(args, patch_len=patch_len, stride=stride).float()
        elif self.model_name == "DLinear":
            individual = bool(getattr(cfg.model.tslib, "individual", False))
            self.model = model_cls(args, individual=individual).float()
        else:
            self.model = model_cls(args).float()

    def _build_decoder_inputs(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, n_vars = x_enc.shape
        label_len = min(self.label_len, seq_len)
        if label_len < self.label_len:
            pad = torch.zeros(
                bsz,
                self.label_len - label_len,
                n_vars,
                dtype=x_enc.dtype,
                device=x_enc.device,
            )
            context = torch.cat([pad, x_enc[:, -label_len:, :]], dim=1)
        else:
            context = x_enc[:, -label_len:, :]

        zeros = torch.zeros(
            bsz,
            self.pred_len,
            n_vars,
            dtype=x_enc.dtype,
            device=x_enc.device,
        )
        x_dec = torch.cat([context, zeros], dim=1)

        if x_mark_enc is None:
            return x_dec, None

        mark_dim = x_mark_enc.size(-1)
        if label_len < self.label_len:
            mark_pad = torch.zeros(
                bsz,
                self.label_len - label_len,
                mark_dim,
                dtype=x_mark_enc.dtype,
                device=x_mark_enc.device,
            )
            mark_context = torch.cat([mark_pad, x_mark_enc[:, -label_len:, :]], dim=1)
        else:
            mark_context = x_mark_enc[:, -label_len:, :]

        if mark_context.size(1) == 0:
            future_mark = torch.zeros(
                bsz,
                self.pred_len,
                mark_dim,
                dtype=x_mark_enc.dtype,
                device=x_mark_enc.device,
            )
        else:
            future_mark = mark_context[:, -1:, :].repeat(1, self.pred_len, 1)
        x_mark_dec = torch.cat([mark_context, future_mark], dim=1)
        return x_dec, x_mark_dec

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        meta_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # meta_emb is intentionally ignored for TSLib baselines.
        del meta_emb

        x_dec, x_mark_dec = self._build_decoder_inputs(x_enc, x_mark_enc)
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() != 3:
            raise ValueError(
                f"TSLib model output must be rank-3 [B,L,N], got shape={tuple(out.shape)}"
            )

        if out.size(1) < self.pred_len:
            raise ValueError(
                f"TSLib model output length {out.size(1)} < pred_len {self.pred_len}"
            )
        out = out[:, -self.pred_len :, :]

        n_vars = x_enc.size(-1)
        if out.size(-1) != n_vars:
            raise ValueError(
                "TSLib model output channel mismatch: "
                f"got {out.size(-1)}, expected {n_vars}. "
                "Set data.enc_in/data.c_out and model.tslib.model_name compatibly."
            )
        return out
