from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.models.factory import build_model


def _load_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _load_model(run_dir: str, device: torch.device):
    cfg = _load_cfg(run_dir)
    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(run_dir, "downstream_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"downstream checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return cfg, model


def _get_sample(cfg, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    _, test_loader = data_provider(cfg, flag="test")
    if sample_idx < 0:
        raise ValueError("sample_idx must be >= 0")
    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            batch_x, batch_y, batch_x_mark, _ = batch
            x = torch.as_tensor(batch_x, dtype=torch.float32)
            y = torch.as_tensor(batch_y, dtype=torch.float32)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32)
            return x, y, x_mark
    raise IndexError(f"sample_idx={sample_idx} out of range (test loader length={len(test_loader)})")


def _predict(model, x: torch.Tensor, x_mark: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        x_dev = x.to(device)
        x_mark_dev = x_mark.to(device) if x_mark is not None else None
        pred = model(x_dev, x_mark_dev)
    return pred.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay GT vs M0 vs P0 prediction for one test sample."
    )
    parser.add_argument("--m0-run-id", required=True, help="Run id for M0 model")
    parser.add_argument("--p0-run-id", required=True, help="Run id for P0 model")
    parser.add_argument("--runs-dir", default="./artifacts/runs", help="Runs directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Test sample index")
    parser.add_argument("--var-idx", type=int, default=0, help="Variable/channel index to plot")
    parser.add_argument("--out", default="./artifacts/analysis/m0_p0_overlay.png", help="Output image path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument(
        "--title",
        default="GT vs M0(sharedpma/outfuse) vs P0",
        help="Plot title",
    )
    args = parser.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    m0_dir = os.path.join(args.runs_dir, args.m0_run_id)
    p0_dir = os.path.join(args.runs_dir, args.p0_run_id)
    if not os.path.isdir(m0_dir):
        raise FileNotFoundError(f"M0 run dir not found: {m0_dir}")
    if not os.path.isdir(p0_dir):
        raise FileNotFoundError(f"P0 run dir not found: {p0_dir}")

    m0_cfg, m0_model = _load_model(m0_dir, device)
    p0_cfg, p0_model = _load_model(p0_dir, device)

    if int(m0_cfg.data.pred_len) != int(p0_cfg.data.pred_len):
        raise ValueError(
            f"pred_len mismatch: M0={m0_cfg.data.pred_len}, P0={p0_cfg.data.pred_len}"
        )
    if int(m0_cfg.data.seq_len) != int(p0_cfg.data.seq_len):
        raise ValueError(
            f"seq_len mismatch: M0={m0_cfg.data.seq_len}, P0={p0_cfg.data.seq_len}"
        )

    # Use one cfg to fetch the shared test sample (must match data split settings).
    x, y, x_mark = _get_sample(m0_cfg, args.sample_idx)
    pred_len = int(m0_cfg.data.pred_len)
    if args.var_idx < 0 or args.var_idx >= int(x.shape[-1]):
        raise IndexError(f"var_idx={args.var_idx} out of range (n_vars={int(x.shape[-1])})")

    y_true = y[:, -pred_len:, :]  # [1, pred_len, N]
    y_m0 = _predict(m0_model, x, x_mark, device)[:, -pred_len:, :]
    y_p0 = _predict(p0_model, x, x_mark, device)[:, -pred_len:, :]

    gt_series = y_true[0, :, args.var_idx].numpy()
    m0_series = y_m0[0, :, args.var_idx].numpy()
    p0_series = y_p0[0, :, args.var_idx].numpy()
    hist_series = x[0, :, args.var_idx].numpy()

    horizon_x = list(range(len(gt_series)))
    hist_x = list(range(-len(hist_series), 0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(hist_x, hist_series, label="History", color="gray", alpha=0.55, linewidth=1.0)
    plt.plot(horizon_x, gt_series, label="Ground Truth", color="black", linewidth=2.0)
    plt.plot(horizon_x, m0_series, label="M0", color="tab:blue", linewidth=1.7)
    plt.plot(horizon_x, p0_series, label="P0", color="tab:orange", linewidth=1.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title(
        f"{args.title}\n"
        f"sample={args.sample_idx}, var={args.var_idx}, seq_len={m0_cfg.data.seq_len}, pred_len={pred_len}"
    )
    plt.xlabel("Time step (0 = forecast start)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()

    print(f"[plot] saved: {args.out}")
    print(f"[plot] m0_run_id={args.m0_run_id}")
    print(f"[plot] p0_run_id={args.p0_run_id}")


if __name__ == "__main__":
    main()
