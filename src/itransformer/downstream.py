from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.models.factory import build_model
from itransformer.utils.ids import build_run_id
from itransformer.utils.metadata import load_or_build_embeddings
from itransformer.utils.metrics import mae, mse


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


def _freeze_modules(modules, freeze: bool) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad = not freeze


def _load_ssl_checkpoint(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    filtered = {k: v for k, v in state.items() if not k.startswith("projector.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[downstream] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[downstream] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")


def _maybe_inject_patch_len(cfg, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return
    ckpt_cfg = ckpt.get("cfg") or {}
    if not isinstance(ckpt_cfg, dict):
        return
    patch_cfg = ckpt_cfg.get("model", {}).get("patch", {}) if isinstance(ckpt_cfg, dict) else {}
    ckpt_patch_len = patch_cfg.get("patch_len") or ckpt_cfg.get("ssl", {}).get("patch_len")
    ckpt_patch_mode = patch_cfg.get("mode") or ckpt_cfg.get("ssl", {}).get("patch_mode")

    if not hasattr(cfg.model, "patch"):
        return
    if getattr(cfg.model.patch, "mode", "") == "mean_pool":
        cfg.model.patch.patch_len = cfg.data.seq_len
        return
    if getattr(cfg.model.patch, "patch_len", 0) in (0, None) and ckpt_patch_len:
        cfg.model.patch.patch_len = ckpt_patch_len


def _evaluate(model, loader, device, pred_len, meta_emb=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            if meta_emb is None:
                out = model(x_enc, x_mark)
            else:
                out = model(x_enc, x_mark, meta_emb)
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            true = true[:, -pred_len:, :]
            preds.append(out)
            trues.append(true)
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    return {
        "mse": mse(pred, true).item(),
        "mae": mae(pred, true).item(),
    }


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

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_data, train_loader = data_provider(cfg, flag="train")
    val_data, val_loader = data_provider(cfg, flag="val")
    test_data, test_loader = data_provider(cfg, flag="test")

    if cfg.train.mode in ("ft", "lp") and cfg.train.ssl_ckpt_path:
        _maybe_inject_patch_len(cfg, cfg.train.ssl_ckpt_path)

    model = build_model(cfg)
    model = model.to(device)

    meta_emb = None
    if getattr(cfg.metadata, "enabled", False):
        sensor_ids = getattr(train_data, "sensor_ids", None)
        if not sensor_ids:
            raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
        meta_emb = load_or_build_embeddings(cfg, sensor_ids).to(device)

    if cfg.train.mode in ("ft", "lp"):
        if not cfg.train.ssl_ckpt_path:
            raise ValueError("train.ssl_ckpt_path is required for ft/lp")
        _load_ssl_checkpoint(model, cfg.train.ssl_ckpt_path)

    # Freeze rules
    if hasattr(model, "enc_embedding"):
        freeze_modules = [model.enc_embedding, model.encoder]
    else:
        # patch model
        freeze_modules = [model.value_proj, model.encoder]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr)
    criterion = torch.nn.MSELoss()

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
    for epoch in range(cfg.train.epochs):
        epoch_start = time.perf_counter()
        if cfg.train.mode == "lp":
            _freeze_modules(freeze_modules, True)
        elif cfg.train.mode == "ft" and cfg.train.freeze_epochs > 0:
            _freeze_modules(freeze_modules, epoch < cfg.train.freeze_epochs)
        else:
            _freeze_modules(freeze_modules, False)

        model.train()
        epoch_losses = []
        epoch_mses = []
        epoch_maes = []
        epoch_grad_norms = []
        for batch in train_loader:
            step_start = time.perf_counter()
            batch_x, batch_y, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            true = true[:, -cfg.data.pred_len :, :]

            optimizer.zero_grad()
            if meta_emb is None:
                out = model(x_enc, x_mark)
            else:
                out = model(x_enc, x_mark, meta_emb)
            loss = criterion(out, true)
            loss.backward()
            grad_norm = _grad_norm(model.parameters())
            optimizer.step()
            epoch_losses.append(loss.detach().item())
            epoch_mses.append(torch.mean((out - true) ** 2).detach().item())
            epoch_maes.append(torch.mean(torch.abs(out - true)).detach().item())
            epoch_grad_norms.append(grad_norm)
            step_time = time.perf_counter() - step_start
            total_step_time += step_time
            total_steps += 1

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_mse = sum(epoch_mses) / max(1, len(epoch_mses))
        train_mae = sum(epoch_maes) / max(1, len(epoch_maes))
        val_metrics = _evaluate(model, val_loader, device, cfg.data.pred_len, meta_emb=meta_emb)
        val_loss = val_metrics["mse"]

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        grad_norm_curve.append(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms)))
        lr_curve.append(float(optimizer.param_groups[0]["lr"]))

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[downstream] epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_train = {"loss": train_loss, "mse": train_mse, "mae": train_mae}
            best_val_metrics = {"loss": val_loss}
            wait = 0
            torch.save(
                {"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)},
                os.path.join(run_dir, "downstream_checkpoint.pt"),
            )
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                early_stopped = True
                stopped_epoch = epoch + 1
                break

    wall_time = time.perf_counter() - wall_start

    test_metrics = _evaluate(model, test_loader, device, cfg.data.pred_len, meta_emb=meta_emb)
    metrics = {
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
        "eval": {"test": test_metrics},
    }

    with open(os.path.join(run_dir, "downstream_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "state": "completed",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"[downstream] run_id={run_id}")
    print(f"[downstream] run_dir={run_dir}")


if __name__ == "__main__":
    main()
