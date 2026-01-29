import time
from typing import Dict

import torch
import torch.nn.functional as F


def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((pred - true) ** 2).item()
    mae = torch.mean(torch.abs(pred - true)).item()
    return {"mse": mse, "mae": mae}


def parse_level(tag: str, default: float = 0.1) -> float:
    if tag is None or tag == "":
        return default
    if isinstance(tag, (int, float)):
        return float(tag)
    if not isinstance(tag, str):
        tag = str(tag)
    t = tag.lower().strip()
    if t in ("l1", "level1"):
        return 0.05
    if t in ("l2", "level2"):
        return 0.1
    if t in ("l3", "level3"):
        return 0.2
    try:
        return float(t)
    except Exception:
        return default


def apply_noise(x: torch.Tensor, level: float) -> torch.Tensor:
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(x) * (level * std)
    return x + noise


def apply_bias_scale(x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return x * (1.0 + scale) + bias * std


def apply_missing_values(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    mask = torch.rand_like(x) < rate
    return x.masked_fill(mask, 0.0)


def apply_missing_channels(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    bsz, _, n_vars = x.shape
    mask = torch.rand(bsz, n_vars, device=x.device) < rate
    return x.masked_fill(mask.unsqueeze(1), 0.0)


def apply_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return x
    bsz, seq_len, n_vars = x.shape
    usable_len = (seq_len // factor) * factor
    x_trim = x[:, :usable_len, :]
    x_ds = x_trim[:, ::factor, :]
    # linear interpolation back to original length
    x_ds = x_ds.permute(0, 2, 1).reshape(bsz * n_vars, 1, -1)
    x_up = F.interpolate(x_ds, size=usable_len, mode="linear", align_corners=False)
    x_up = x_up.reshape(bsz, n_vars, usable_len).permute(0, 2, 1)
    if usable_len < seq_len:
        pad = torch.zeros(bsz, seq_len - usable_len, n_vars, device=x.device)
        x_up = torch.cat([x_up, pad], dim=1)
    return x_up


def measure_inference_time(model, x_enc, x_mark, meta_emb=None, repeats: int = 10) -> float:
    # returns avg time per forward (ms)
    if x_mark is None:
        args = (x_enc,)
    else:
        args = (x_enc, x_mark)
    if meta_emb is not None:
        args = args + (meta_emb,)

    if x_enc.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(*args)
    if x_enc.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / max(1, repeats)
