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


def _freeze_modules(modules, freeze: bool) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad = not freeze


def _remap_ssl_state(model, state: dict) -> tuple[dict, dict]:
    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    remapped = {}
    skipped = {
        "shape_mismatch": [],
        "missing_keys": [],
        "skipped_projector": 0,
    }
    for key, value in state.items():
        if key.startswith("projector"):
            skipped["skipped_projector"] += 1
            continue
        new_key = key
        if key.startswith("encoder_layers."):
            new_key = "encoder.attn_layers." + key[len("encoder_layers.") :]
        elif key.startswith("encoder_norm."):
            new_key = "encoder.norm." + key[len("encoder_norm.") :]
        elif key.startswith("value_proj.") and hasattr(model, "patch_embed"):
            new_key = "patch_embed." + key[len("value_proj.") :]
        if new_key in target_keys:
            if target_state[new_key].shape == value.shape:
                remapped[new_key] = value
            else:
                skipped["shape_mismatch"].append(
                    (new_key, tuple(value.shape), tuple(target_state[new_key].shape))
                )
        else:
            skipped["missing_keys"].append(new_key)
    return remapped, skipped


def _load_ssl_checkpoint(model, path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    filtered, skipped = _remap_ssl_state(model, state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[downstream] ssl_ckpt_path: {path}")
    print(f"[downstream] remapped keys: {len(filtered)} / raw keys: {len(state)}")
    if missing:
        print(f"[downstream] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[downstream] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if skipped["shape_mismatch"]:
        preview = skipped["shape_mismatch"][:3]
        print(
            "[downstream] shape mismatches (first 3): "
            + ", ".join([f"{k} {s}!= {t}" for k, s, t in preview])
        )
        if len(skipped["shape_mismatch"]) > 3:
            print(f"[downstream] shape mismatches total: {len(skipped['shape_mismatch'])}")
    patch_embed_mismatch = [
        (k, s, t)
        for k, s, t in skipped["shape_mismatch"]
        if k.startswith("patch_embed.")
    ]
    if patch_embed_mismatch:
        preview = ", ".join([f"{k} {s}!= {t}" for k, s, t in patch_embed_mismatch[:3]])
        raise ValueError(
            "[downstream] patch_embed shape mismatch; aborting. "
            f"mismatches (first 3): {preview}"
        )
    def _is_allowed_mismatch(key: str) -> bool:
        return key.startswith("meta_")

    critical_shape = [
        (k, s, t)
        for k, s, t in skipped["shape_mismatch"]
        if not _is_allowed_mismatch(k)
    ]
    critical_missing = [
        k for k in skipped["missing_keys"] if not _is_allowed_mismatch(k)
    ]
    if critical_shape or critical_missing:
        parts = []
        if critical_shape:
            preview = ", ".join([f"{k} {s}!= {t}" for k, s, t in critical_shape[:3]])
            parts.append(f"shape_mismatch: {preview}")
        if critical_missing:
            parts.append(f"unmapped_keys: {critical_missing[:5]}")
        raise ValueError(
            "[downstream] SSL checkpoint keys mismatch; aborting. "
            + " | ".join(parts)
        )
    if skipped["missing_keys"]:
        print(
            f"[downstream] unmapped keys (first 5): {skipped['missing_keys'][:5]}"
        )
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        weight = patch_embed.weight.detach().float()
        print(
            f"[downstream] patch_embed stats: mean={weight.mean().item():.6f} "
            f"std={weight.std().item():.6f}"
        )
    return {
        "missing": missing,
        "unexpected": unexpected,
        "skipped": skipped,
        "patch_embed_loaded": any(k.startswith("patch_embed.") for k in filtered),
    }


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
    _set_seed(cfg)

    train_data, train_loader = data_provider(cfg, flag="train")
    val_flag = str(getattr(cfg.train, "val_flag", "val") or "val").lower()
    if val_flag not in ("val", "test"):
        raise ValueError(f"Unsupported train.val_flag: {val_flag} (expected 'val' or 'test')")

    test_data, test_loader = data_provider(cfg, flag="test")
    if val_flag == "test":
        val_data, val_loader = test_data, test_loader
        print("[downstream] WARNING: Using test split for validation (train.val_flag=test).")
    else:
        val_data, val_loader = data_provider(cfg, flag="val")

    train_mode = cfg.train.mode
    if train_mode in ("ft", "lp") and not cfg.train.ssl_ckpt_path:
        print("[downstream] ssl_ckpt_path is empty; running from scratch (no load, no freeze).")
        train_mode = "scratch"

    if train_mode in ("ft", "lp") and cfg.train.ssl_ckpt_path:
        _maybe_inject_patch_len(cfg, cfg.train.ssl_ckpt_path)

    model = build_model(cfg)
    model = model.to(device)

    meta_emb = None
    if getattr(cfg.metadata, "enabled", False):
        sensor_ids = getattr(train_data, "sensor_ids", None)
        if not sensor_ids:
            raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
        meta_emb = load_or_build_embeddings(cfg, sensor_ids).to(device)

    load_info = None
    if train_mode in ("ft", "lp"):
        load_info = _load_ssl_checkpoint(model, cfg.train.ssl_ckpt_path)

    # Freeze rules
    if hasattr(model, "enc_embedding"):
        freeze_modules = [model.enc_embedding, model.encoder]
    else:
        # patch model
        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            if load_info is not None and not load_info.get("patch_embed_loaded", True):
                print("[downstream] patch_embed not loaded; keeping it trainable in LP/FT.")
                freeze_modules = [model.encoder]
            else:
                freeze_modules = [patch_embed, model.encoder]
        else:
            freeze_modules = [model.value_proj, model.encoder]

    base_lr = float(cfg.optim.lr)
    cln_lr_mult = float(getattr(cfg.optim, "cln_lr_mult", 1.0) or 1.0)
    if cln_lr_mult <= 0:
        raise ValueError(f"optim.cln_lr_mult must be > 0 (got {cln_lr_mult})")

    # Optional: separate param group for slot-conditioned LayerNorm (CLN) gamma/beta embeddings.
    # This lets us keep the main model stable while allowing CLN to adapt faster.
    if cln_lr_mult != 1.0:
        cln_params = []
        base_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".gamma.weight") or name.endswith(".beta.weight"):
                cln_params.append(p)
            else:
                base_params.append(p)
        if cln_params:
            optimizer = torch.optim.Adam(
                [
                    {"params": base_params, "lr": base_lr},
                    {"params": cln_params, "lr": base_lr * cln_lr_mult},
                ]
            )
            print(
                "[downstream] Using CLN param group: "
                f"base_lr={base_lr:g}, cln_lr={base_lr * cln_lr_mult:g} (mult={cln_lr_mult:g}), "
                f"n_cln_params={len(cln_params)}"
            )
        else:
            print("[downstream] optim.cln_lr_mult is set but no CLN params found; ignoring.")
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    scheduler = None
    scheduler_name = str(getattr(cfg.optim, "scheduler", "none") or "none").lower()
    if scheduler_name != "none":
        if scheduler_name == "cosine":
            t_max = int(getattr(cfg.optim, "t_max", 0) or 0)
            if t_max <= 0:
                t_max = int(cfg.train.epochs)
            min_lr = float(getattr(cfg.optim, "min_lr", 0.0) or 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=min_lr,
            )
        else:
            raise ValueError(f"Unsupported optim.scheduler: {scheduler_name}")
    criterion = torch.nn.MSELoss()

    ms_cfg = getattr(cfg.model, "multislot", None)
    div_cfg = getattr(ms_cfg, "diversity", None) if ms_cfg is not None else None
    div_enabled = bool(getattr(div_cfg, "enabled", False))
    div_lambda = float(getattr(div_cfg, "lambda", 0.0) or 0.0)
    if div_lambda < 0:
        raise ValueError(f"model.multislot.diversity.lambda must be >= 0 (got {div_lambda})")
    if div_enabled and div_lambda > 0:
        print(f"[downstream] diversity loss enabled: lambda={div_lambda:g}")

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
    train_div_loss_curve = []

    epoch_times = []
    total_step_time = 0.0
    total_steps = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wall_start = time.perf_counter()
    for epoch in range(cfg.train.epochs):
        epoch_start = time.perf_counter()
        if train_mode == "lp":
            _freeze_modules(freeze_modules, True)
        elif train_mode == "ft" and cfg.train.freeze_epochs > 0:
            _freeze_modules(freeze_modules, epoch < cfg.train.freeze_epochs)
        else:
            _freeze_modules(freeze_modules, False)

        model.train()
        epoch_losses = []
        epoch_mses = []
        epoch_maes = []
        epoch_grad_norms = []
        epoch_div_losses = []
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
            mse_loss = criterion(out, true)
            loss = mse_loss

            div_loss = None
            if div_enabled and div_lambda > 0:
                get_div = getattr(model, "get_diversity_loss", None)
                if callable(get_div):
                    div_loss = get_div()
                if div_loss is not None:
                    loss = loss + div_lambda * div_loss
                    epoch_div_losses.append(float(div_loss.detach().item()))

            loss.backward()
            grad_norm = _grad_norm(model.parameters())
            optimizer.step()
            epoch_losses.append(mse_loss.detach().item())
            epoch_mses.append(torch.mean((out - true) ** 2).detach().item())
            epoch_maes.append(torch.mean(torch.abs(out - true)).detach().item())
            epoch_grad_norms.append(grad_norm)
            step_time = time.perf_counter() - step_start
            total_step_time += step_time
            total_steps += 1

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_mse = sum(epoch_mses) / max(1, len(epoch_mses))
        train_mae = sum(epoch_maes) / max(1, len(epoch_maes))
        train_div = sum(epoch_div_losses) / max(1, len(epoch_div_losses)) if epoch_div_losses else 0.0
        val_metrics = _evaluate(model, val_loader, device, cfg.data.pred_len, meta_emb=meta_emb)
        val_loss = val_metrics["mse"]

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        train_div_loss_curve.append(train_div)
        grad_norm_curve.append(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms)))
        lr_curve.append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"[downstream] epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} div_loss={train_div:.6f}"
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

    best_ckpt_path = os.path.join(run_dir, "downstream_checkpoint.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

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
            "train_div_loss": train_div_loss_curve,
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
        "notes": {"val_split": val_flag},
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
