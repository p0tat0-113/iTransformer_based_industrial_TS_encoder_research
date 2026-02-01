from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.ssl import PatchMAE, VarMAE
from itransformer.utils.ids import build_run_id
from itransformer.utils.metadata import load_or_build_embeddings


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _set_seed(cfg) -> None:
    if getattr(cfg.runtime, "seed", None) is None:
        return
    seed = int(cfg.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(cfg.runtime, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def _eval_val_loss(model, loader, device, meta_emb):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            loss, _, _ = model(x_enc, x_mark, meta_emb=meta_emb, return_details=True)
            losses.append(loss.detach().item())
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    run_id = cfg.run.id
    if not run_id or "{" in str(run_id):
        run_id = build_run_id(
            cfg.ids.run_id,
            code=cfg.run.code,
            dataset=cfg.data.name,
            variant=cfg.model.variant,
            hparams_tag=cfg.run.hparams_tag,
            seed=cfg.runtime.seed,
        )

    cfg.run.id = run_id
    run_dir = os.path.join(cfg.paths.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")
    _set_seed(cfg)

    train_data, train_loader = data_provider(cfg, flag="train")
    _, val_loader = data_provider(cfg, flag="val")
    meta_emb = None
    if cfg.metadata.enabled:
        sensor_ids = getattr(train_data, "sensor_ids", None)
        if not sensor_ids:
            raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
        meta_emb = load_or_build_embeddings(cfg, sensor_ids)
        meta_emb = meta_emb.to(device)

    if cfg.ssl.type == "var_mae" or getattr(cfg.model.patch, "mode", "none") == "none":
        model = VarMAE(cfg)
    elif cfg.ssl.type == "patch_mae":
        model = PatchMAE(cfg)
    else:
        raise ValueError(f"Unsupported ssl.type: {cfg.ssl.type}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    patience = int(getattr(cfg.train, "patience", 0) or 0)
    best_val = float("inf")
    best_epoch = 0
    best_train = {"loss": None, "mse": None, "mae": None}
    best_val_metrics = {"loss": None}
    wait = 0
    early_stopped = False
    stopped_epoch = None

    train_loss_curve = []
    val_loss_curve = []
    grad_norm_curve = []
    lr_curve = []

    epoch_times = []
    total_step_time = 0.0
    total_steps = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wall_start = time.perf_counter()
    for epoch in range(cfg.ssl.pretrain_epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_losses = []
        epoch_mses = []
        epoch_maes = []
        epoch_grad_norms = []
        for batch in train_loader:
            step_start = time.perf_counter()
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss, mse, mae = model(x_enc, x_mark, meta_emb=meta_emb, return_details=True)
            loss.backward()
            grad_norm = _grad_norm(model.parameters())
            optimizer.step()
            epoch_losses.append(loss.detach().item())
            epoch_mses.append(mse.detach().item())
            epoch_maes.append(mae.detach().item())
            epoch_grad_norms.append(grad_norm)
            step_time = time.perf_counter() - step_start
            total_step_time += step_time
            total_steps += 1

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_mse = sum(epoch_mses) / max(1, len(epoch_mses))
        train_mae = sum(epoch_maes) / max(1, len(epoch_maes))
        val_loss = _eval_val_loss(model, val_loader, device, meta_emb)

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        grad_norm_curve.append(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms)))
        lr_curve.append(float(optimizer.param_groups[0]["lr"]))

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[pretrain] epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_train = {"loss": train_loss, "mse": train_mse, "mae": train_mae}
            best_val_metrics = {"loss": val_loss}
            wait = 0
            ckpt_path = os.path.join(run_dir, "pretrain_checkpoint.pt")
            torch.save({"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, ckpt_path)
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                early_stopped = True
                stopped_epoch = epoch + 1
                break

    wall_time = time.perf_counter() - wall_start

    metrics_path = os.path.join(run_dir, "pretrain_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "best_epoch": best_epoch,
                    "best_train": best_train,
                    "best_val": best_val_metrics,
                    "early_stopped": early_stopped,
                    "patience": patience,
                    "stopped_epoch": stopped_epoch,
                },
                "curves": {
                    "train_loss": train_loss_curve,
                    "val_loss": val_loss_curve,
                    "grad_norm": grad_norm_curve,
                    "lr": lr_curve,
                },
                "cost": {
                    "wall_time_sec_total": wall_time,
                    "time_sec_per_epoch_mean": sum(epoch_times) / max(1, len(epoch_times)),
                    "time_sec_per_step_mean": total_step_time / max(1, total_steps),
                    "gpu_mem_peak_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
                    ),
                    "params_count": _param_count(model),
                },
                "notes": {"val_metric_basis": "masked_only"},
            },
            f,
            indent=2,
        )

    status = {
        "state": "completed",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[pretrain] run_id={run_id}")
    print(f"[pretrain] run_dir={run_dir}")


if __name__ == "__main__":
    main()
