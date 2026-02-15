#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from itransformer.analysis.utils import load_run_checkpoint, load_state, resolve_run_dir
from itransformer.data import data_provider
from itransformer.models.factory import build_model
from itransformer.utils.metadata import load_or_build_embeddings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Informer attention modes (prob/full) on the same config/checkpoint "
            "and report latency + MSE/MAE."
        )
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id under artifacts/runs (optional).")
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs", help="Base runs directory.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path (optional).")
    parser.add_argument("--config", type=str, default="", help="Resolved config.yaml path (optional).")
    parser.add_argument("--data", type=str, default="ETTh1", help="Hydra data config name (used if no run/config).")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Evaluation split.")
    parser.add_argument("--modes", type=str, default="prob,full", help="Comma-separated attn modes.")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches to evaluate per mode.")
    parser.add_argument("--warmup-batches", type=int, default=10, help="Warmup batches excluded from timing.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--pred-len", type=int, default=0, help="Override pred_len (0 keeps config value).")
    parser.add_argument("--output-json", type=str, default="", help="Optional output json path.")
    return parser.parse_args()


def _load_cfg_from_file(path: str):
    loaded = OmegaConf.load(path)
    if loaded is None:
        raise ValueError(f"Failed to load config: {path}")
    return loaded


def _resolve_base_cfg(args: argparse.Namespace):
    ckpt_path = args.ckpt.strip() if args.ckpt else ""

    if args.run_id:
        run_dir = resolve_run_dir(args.run_id, args.runs_dir)
        cfg_path = os.path.join(run_dir, "config.yaml")
        cfg = _load_cfg_from_file(cfg_path)
        if not ckpt_path:
            ckpt_path = load_run_checkpoint(args.run_id, args.runs_dir)
        return cfg, ckpt_path

    if args.config:
        cfg = _load_cfg_from_file(args.config)
        return cfg, ckpt_path

    if ckpt_path:
        cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        if os.path.exists(cfg_path):
            cfg = _load_cfg_from_file(cfg_path)
            return cfg, ckpt_path

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[f"model=Informer", f"data={args.data}"])
    return cfg, ckpt_path


def _prepare_cfg(base_cfg, args: argparse.Namespace, mode: str):
    cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(base_cfg, resolve=False)))
    cfg.model.variant = "Informer"
    if not hasattr(cfg.model, "informer") or cfg.model.informer is None:
        cfg.model.informer = OmegaConf.create({})
    cfg.model.informer.attn = mode
    cfg.runtime.device = args.device
    cfg.train.num_workers = int(args.num_workers)
    if args.pred_len > 0:
        cfg.data.pred_len = int(args.pred_len)
    return cfg


def _maybe_meta(cfg, dataset, device: torch.device):
    if not getattr(cfg.metadata, "enabled", False):
        return None
    sensor_ids = getattr(dataset, "sensor_ids", None)
    if not sensor_ids:
        raise ValueError("metadata.enabled=true but dataset has no sensor_ids.")
    return load_or_build_embeddings(cfg, sensor_ids).to(device)


def _run_one_mode(cfg, ckpt_path: str, split: str, max_batches: int, warmup_batches: int):
    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")
    _, loader = data_provider(cfg, flag=split)
    dataset = loader.dataset

    model = build_model(cfg).to(device)
    if ckpt_path:
        load_state(model, ckpt_path)
    model.eval()

    meta_emb = _maybe_meta(cfg, dataset, device)
    pred_len = int(cfg.data.pred_len)

    elapsed = []
    mse_sum = 0.0
    mae_sum = 0.0
    n_elem = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device) if batch_x_mark is not None else None
            y_mark = torch.as_tensor(batch_y_mark, dtype=torch.float32, device=device) if batch_y_mark is not None else None
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)[:, -pred_len:, :]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= warmup_batches:
                elapsed.append(t1 - t0)

            diff = pred - true
            mse_sum += float(torch.sum(diff * diff).item())
            mae_sum += float(torch.sum(torch.abs(diff)).item())
            n_elem += int(diff.numel())

    if n_elem == 0:
        raise RuntimeError("No evaluation elements were processed.")

    avg_latency_ms = (sum(elapsed) / max(1, len(elapsed))) * 1000.0
    return {
        "mse": mse_sum / n_elem,
        "mae": mae_sum / n_elem,
        "avg_latency_ms": avg_latency_ms,
        "timed_batches": len(elapsed),
        "processed_batches": min(max_batches, len(loader)),
        "pred_len": pred_len,
        "split": split,
    }


def main() -> None:
    args = _parse_args()
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("No modes provided.")
    invalid = [m for m in modes if m not in ("prob", "full")]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid} (supported: prob, full)")

    base_cfg, ckpt_path = _resolve_base_cfg(args)
    results = []
    for mode in modes:
        cfg = _prepare_cfg(base_cfg, args, mode)
        out = _run_one_mode(
            cfg,
            ckpt_path=ckpt_path,
            split=args.split,
            max_batches=int(args.max_batches),
            warmup_batches=int(args.warmup_batches),
        )
        out["mode"] = mode
        results.append(out)

    print("| mode | MSE | MAE | avg_latency_ms | timed_batches |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['mode']} | {r['mse']:.6f} | {r['mae']:.6f} | "
            f"{r['avg_latency_ms']:.3f} | {r['timed_batches']} |"
        )

    if len(results) == 2:
        a, b = results[0], results[1]
        print(
            f"\nDelta ({a['mode']} - {b['mode']}): "
            f"ΔMSE={a['mse'] - b['mse']:+.6f}, "
            f"ΔMAE={a['mae'] - b['mae']:+.6f}, "
            f"Δlatency_ms={a['avg_latency_ms'] - b['avg_latency_ms']:+.3f}"
        )

    if args.output_json:
        payload = {
            "modes": modes,
            "ckpt": ckpt_path,
            "run_id": args.run_id,
            "split": args.split,
            "max_batches": int(args.max_batches),
            "warmup_batches": int(args.warmup_batches),
            "results": results,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
