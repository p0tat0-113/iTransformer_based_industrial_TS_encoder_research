from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.analysis.utils import load_run_checkpoint, load_state, resolve_run_dir
from itransformer.data import data_provider
from itransformer.evals.utils import (
    apply_bias_scale,
    apply_time_half_swap,
    apply_time_shuffle_full,
    apply_missing_values,
    apply_noise,
    apply_downsample,
    compute_metrics,
    parse_level,
)
from itransformer.models.factory import build_model
from itransformer.utils.ids import build_op_id
from itransformer.utils.metadata import load_or_build_embeddings


def _resolve_ckpt(cfg) -> str:
    if cfg.eval.on_run_id:
        return load_run_checkpoint(cfg.eval.on_run_id, cfg.paths.runs_dir)
    if getattr(cfg.eval, "ckpt_path", ""):
        return cfg.eval.ckpt_path
    raise ValueError("eval.on_run_id or eval.ckpt_path is required")


def _maybe_meta(cfg, dataset, device):
    if not getattr(cfg.metadata, "enabled", False):
        return None
    sensor_ids = getattr(dataset, "sensor_ids", None)
    if not sensor_ids:
        raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
    meta_emb = load_or_build_embeddings(cfg, sensor_ids)
    return meta_emb.to(device)


def _predict(model, x_enc, x_mark, meta_emb, y_mark=None, *, use_tslib: bool = False):
    needs_mark = bool(getattr(model, "needs_y_mark_dec", False))
    if use_tslib or needs_mark:
        return model(x_enc, x_mark, meta_emb, y_mark_dec=y_mark)
    if meta_emb is None:
        return model(x_enc, x_mark)
    return model(x_enc, x_mark, meta_emb)


def _parse_downsample(tag: str, default: int = 2) -> int:
    if not tag:
        return default
    if not isinstance(tag, str):
        tag = str(tag)
    t = tag.lower().strip()
    if t.startswith("d"):
        t = t[1:]
    try:
        val = int(float(t))
        return max(1, val)
    except Exception:
        return default


def _set_seed(cfg) -> None:
    if getattr(cfg.runtime, "seed", None) is None:
        return
    seed = int(cfg.runtime.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cfg_from_yaml(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        loaded = OmegaConf.load(path)
        data = OmegaConf.to_container(loaded, resolve=True)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _infer_ckpt_cfg(cfg, ckpt_path: str, ckpt_payload: dict | None) -> dict | None:
    if isinstance(ckpt_payload, dict) and isinstance(ckpt_payload.get("cfg"), dict):
        return ckpt_payload.get("cfg")

    if getattr(cfg.eval, "on_run_id", ""):
        try:
            run_dir = resolve_run_dir(cfg.eval.on_run_id, cfg.paths.runs_dir)
            run_cfg = os.path.join(run_dir, "config.yaml")
            loaded = _load_cfg_from_yaml(run_cfg)
            if isinstance(loaded, dict):
                return loaded
        except (FileNotFoundError, RuntimeError):
            pass

    ckpt_dir_cfg = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    loaded = _load_cfg_from_yaml(ckpt_dir_cfg)
    if isinstance(loaded, dict):
        return loaded
    return None


def _apply_ckpt_cfg(cfg, ckpt_cfg: dict | None) -> None:
    if not isinstance(ckpt_cfg, dict):
        return
    if bool(getattr(cfg.eval, "use_ckpt_model", True)):
        model_cfg = ckpt_cfg.get("model", {})
        if isinstance(model_cfg, dict) and model_cfg:
            cfg.model = OmegaConf.create(model_cfg)

    if bool(getattr(cfg.eval, "use_ckpt_data", True)):
        data_cfg = ckpt_cfg.get("data", {})
        if isinstance(data_cfg, dict) and data_cfg:
            cfg.data = OmegaConf.create(data_cfg)

    if bool(getattr(cfg.eval, "use_ckpt_metadata", True)):
        meta_cfg = ckpt_cfg.get("metadata", {})
        if isinstance(meta_cfg, dict) and meta_cfg:
            cfg.metadata = OmegaConf.create(meta_cfg)


def _require_patch_len(cfg) -> None:
    variant = getattr(cfg.model, "variant", "")
    patch_enabled = bool(getattr(cfg.model.patch, "enabled", False))
    if variant in ("P1", "P2", "P3", "P4") or patch_enabled:
        patch_len = getattr(cfg.model.patch, "patch_len", 0)
        if not patch_len:
            raise ValueError("eval requires model.patch.patch_len for patch models.")


def _run_eval(
    op_code,
    model,
    test_loader,
    device,
    meta_emb,
    pred_len,
    *,
    use_tslib=False,
    level=None,
    downsample=None,
    missing_rate=None,
    shuffle_seed=None,
    shuffle_per_sample=True,
    shuffle_apply_to="x_and_mark",
):
    preds_all = []
    trues_all = []
    channel_mask = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            y_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            if batch_y_mark is not None:
                y_mark = torch.as_tensor(batch_y_mark, dtype=torch.float32, device=device)

            if op_code == "S1":
                x_enc = apply_noise(x_enc, level)
            elif op_code == "S2":
                x_enc = apply_downsample(x_enc, downsample)
            elif op_code == "S3":
                x_enc = apply_bias_scale(x_enc, scale=level, bias=level)
            elif op_code == "R1":
                x_enc = apply_missing_values(x_enc, missing_rate)
            elif op_code == "R2":
                if channel_mask is None:
                    _, _, n_vars = x_enc.shape
                    channel_mask = torch.rand(n_vars, device=device) < missing_rate
                x_enc = x_enc.masked_fill(channel_mask.view(1, 1, -1), 0.0)
            elif op_code == "X1":
                batch_seed = None if shuffle_seed is None else int(shuffle_seed) + int(batch_idx)
                x_enc, x_mark = apply_time_shuffle_full(
                    x_enc,
                    x_mark,
                    seed=batch_seed,
                    per_sample=bool(shuffle_per_sample),
                    apply_to=shuffle_apply_to,
                )
            elif op_code == "X2":
                x_enc, x_mark = apply_time_half_swap(x_enc, x_mark, apply_to=shuffle_apply_to)
            elif op_code in ("T1", "T2", "T3"):
                pass
            else:
                raise ValueError(f"Unknown eval.op_code: {op_code}")

            pred = _predict(model, x_enc, x_mark, meta_emb, y_mark, use_tslib=use_tslib)
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            true = true[:, -pred_len:, :]
            preds_all.append(pred)
            trues_all.append(true)

    pred = torch.cat(preds_all, dim=0)
    true = torch.cat(trues_all, dim=0)
    return compute_metrics(pred, true)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    op_code = cfg.eval.op_code
    if not op_code:
        raise ValueError("eval.op_code is required")

    ckpt_path = _resolve_ckpt(cfg)
    try:
        ckpt_payload = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt_payload = None
    ckpt_cfg = _infer_ckpt_cfg(cfg, ckpt_path, ckpt_payload if isinstance(ckpt_payload, dict) else None)
    _apply_ckpt_cfg(cfg, ckpt_cfg)
    _require_patch_len(cfg)
    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")
    if getattr(cfg.runtime, "deterministic", False):
        _set_seed(cfg)

    split = str(getattr(cfg.eval, "split", "test") or "test").lower().strip()
    if split != "test":
        raise ValueError(f"eval.split={split} is not supported; only 'test' is allowed.")
    test_data, test_loader = data_provider(cfg, flag=split)
    meta_emb = _maybe_meta(cfg, test_data, device)
    meta_emb_eval = meta_emb

    model = build_model(cfg)
    load_state(model, ckpt_path)
    model = model.to(device)
    model.eval()
    use_tslib = str(getattr(cfg.model, "variant", "")) == "TSLIB"

    results = {}

    level = parse_level(cfg.eval.op_hparams_tag or "")
    missing_rate = parse_level(cfg.eval.op_hparams_tag or "", default=0.1)
    missing_rates = list(getattr(cfg.eval, "missing_rates", []) or [])
    shuffle_seed = getattr(cfg.eval, "shuffle_seed", None)
    if shuffle_seed is not None:
        shuffle_seed = int(shuffle_seed)
    shuffle_per_sample = bool(getattr(cfg.eval, "shuffle_per_sample", True))
    shuffle_apply_to = str(getattr(cfg.eval, "shuffle_apply_to", "x_and_mark") or "x_and_mark")
    if shuffle_apply_to not in ("x_and_mark", "x_only"):
        raise ValueError(
            f"Unsupported eval.shuffle_apply_to: {shuffle_apply_to} (expected 'x_and_mark' or 'x_only')"
        )

    if op_code in ("S1", "S2", "S3"):
        if op_code == "S2":
            factor = _parse_downsample(cfg.eval.op_hparams_tag, default=2)
            results["op_params"] = {"downsample_factor": factor}
            metrics = _run_eval(
                "S2",
                model,
                test_loader,
                device,
                meta_emb_eval,
                cfg.data.pred_len,
                use_tslib=use_tslib,
                downsample=factor,
            )
        else:
            results["op_params"] = {"level": level}
            metrics = _run_eval(
                op_code,
                model,
                test_loader,
                device,
                meta_emb_eval,
                cfg.data.pred_len,
                use_tslib=use_tslib,
                level=level,
            )
        results.update(metrics)
    elif op_code in ("X1", "X2"):
        results["op_params"] = {
            "shuffle_seed": shuffle_seed,
            "shuffle_per_sample": shuffle_per_sample,
            "shuffle_apply_to": shuffle_apply_to,
        }
        metrics = _run_eval(
            op_code,
            model,
            test_loader,
            device,
            meta_emb_eval,
            cfg.data.pred_len,
            use_tslib=use_tslib,
            shuffle_seed=shuffle_seed,
            shuffle_per_sample=shuffle_per_sample,
            shuffle_apply_to=shuffle_apply_to,
        )
        results.update(metrics)
    elif op_code == "T1":
        if meta_emb is None:
            raise ValueError("T1 requires metadata enabled")
        base_metrics = _run_eval(
            "T1",
            model,
            test_loader,
            device,
            meta_emb,
            cfg.data.pred_len,
            use_tslib=use_tslib,
        )
        perm = torch.randperm(meta_emb.size(0), device=meta_emb.device)
        shuffled_metrics = _run_eval(
            "T1",
            model,
            test_loader,
            device,
            meta_emb[perm],
            cfg.data.pred_len,
            use_tslib=use_tslib,
        )
        results["base_metrics"] = base_metrics
        results["shuffled_metrics"] = shuffled_metrics
        results["delta"] = {
            "delta_mse": shuffled_metrics["mse"] - base_metrics["mse"],
            "delta_mae": shuffled_metrics["mae"] - base_metrics["mae"],
        }
    elif op_code == "T2":
        if meta_emb is None:
            raise ValueError("T2 requires metadata enabled")
        if not missing_rates:
            missing_rates = [missing_rate]
        metrics_by_rate = {}
        for rate in missing_rates:
            mask = torch.rand(meta_emb.size(0), device=meta_emb.device) < rate
            meta_masked = meta_emb.clone()
            meta_masked[mask] = 0.0
            metrics_by_rate[str(rate)] = _run_eval(
                "T2",
                model,
                test_loader,
                device,
                meta_masked,
                cfg.data.pred_len,
                use_tslib=use_tslib,
            )
        results["metrics_by_rate"] = metrics_by_rate
    elif op_code in ("R1", "R2"):
        if not missing_rates:
            missing_rates = [missing_rate]
        metrics_by_rate = {}
        for rate in missing_rates:
            metrics_by_rate[str(rate)] = _run_eval(
                op_code,
                model,
                test_loader,
                device,
                meta_emb_eval,
                cfg.data.pred_len,
                use_tslib=use_tslib,
                missing_rate=rate,
            )
        results["metrics_by_rate"] = metrics_by_rate
    elif op_code == "T3":
        results.update(
            _run_eval(
                "T3",
                model,
                test_loader,
                device,
                meta_emb_eval,
                cfg.data.pred_len,
                use_tslib=use_tslib,
            )
        )
    else:
        raise ValueError(f"Unknown eval.op_code: {op_code}")

    op_hparams = getattr(cfg.eval, "op_hparams_tag", "") or ""
    op_id = build_op_id(
        cfg.ids.op_id,
        code=getattr(cfg.eval, "code", ""),
        op_code=op_code,
        op_hparams=op_hparams,
        on_run_id=cfg.eval.on_run_id,
    )
    cfg.eval.op_id = op_id
    op_dir = os.path.join(cfg.paths.ops_dir, op_id)
    os.makedirs(op_dir, exist_ok=True)

    with open(os.path.join(op_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    run_id = cfg.eval.on_run_id or ""
    run_code = run_id.split(".")[0] if run_id else ""
    results["meta"] = {
        "eval_code": getattr(cfg.eval, "code", ""),
        "op_code": op_code,
        "op_hparams_tag": getattr(cfg.eval, "op_hparams_tag", ""),
        "run_id": run_id,
        "run_code": run_code,
        "split": split,
        "dataset": cfg.data.name,
        "model_variant": cfg.model.variant,
    }

    with open(os.path.join(op_dir, "op_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(op_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "state": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "op_code": op_code,
            },
            f,
            indent=2,
        )

    print(f"[eval] op_id={op_id}")
    print(f"[eval] results={results}")


if __name__ == "__main__":
    main()
