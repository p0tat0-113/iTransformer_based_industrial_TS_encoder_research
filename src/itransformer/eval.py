from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.analysis.utils import load_run_checkpoint, load_state
from itransformer.data import data_provider
from itransformer.evals.utils import (
    apply_bias_scale,
    apply_missing_values,
    apply_noise,
    apply_downsample,
    compute_metrics,
    parse_level,
)
from itransformer.models.factory import build_model
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


def _predict(model, x_enc, x_mark, meta_emb):
    if meta_emb is None:
        return model(x_enc, x_mark)
    return model(x_enc, x_mark, meta_emb)


def _parse_downsample(tag: str, default: int = 2) -> int:
    if not tag:
        return default
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


def _run_eval(op_code, model, test_loader, device, meta_emb, pred_len, *, level=None, downsample=None, missing_rate=None):
    preds_all = []
    trues_all = []
    channel_mask = None

    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)

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
            elif op_code in ("T1", "T2", "T3"):
                pass
            else:
                raise ValueError(f"Unknown eval.op_code: {op_code}")

            pred = _predict(model, x_enc, x_mark, meta_emb)
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
    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")
    if getattr(cfg.runtime, "deterministic", False):
        _set_seed(cfg)

    test_data, test_loader = data_provider(cfg, flag="test")
    meta_emb = _maybe_meta(cfg, test_data, device)
    meta_emb_eval = meta_emb

    model = build_model(cfg)
    load_state(model, ckpt_path)
    model = model.to(device)
    model.eval()

    results = {}

    level = parse_level(cfg.eval.op_hparams_tag or "")
    missing_rate = parse_level(cfg.eval.op_hparams_tag or "", default=0.1)
    missing_rates = list(getattr(cfg.eval, "missing_rates", []) or [])

    if op_code in ("S1", "S2", "S3"):
        if op_code == "S2":
            factor = _parse_downsample(cfg.eval.op_hparams_tag, default=2)
            results["op_params"] = {"downsample_factor": factor}
            metrics = _run_eval("S2", model, test_loader, device, meta_emb_eval, cfg.data.pred_len, downsample=factor)
        else:
            results["op_params"] = {"level": level}
            metrics = _run_eval(op_code, model, test_loader, device, meta_emb_eval, cfg.data.pred_len, level=level)
        results.update(metrics)
    elif op_code == "T1":
        if meta_emb is None:
            raise ValueError("T1 requires metadata enabled")
        base_metrics = _run_eval("T1", model, test_loader, device, meta_emb, cfg.data.pred_len)
        perm = torch.randperm(meta_emb.size(0), device=meta_emb.device)
        shuffled_metrics = _run_eval("T1", model, test_loader, device, meta_emb[perm], cfg.data.pred_len)
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
            metrics_by_rate[str(rate)] = _run_eval("T2", model, test_loader, device, meta_masked, cfg.data.pred_len)
        results["metrics_by_rate"] = metrics_by_rate
    elif op_code in ("R1", "R2"):
        if not missing_rates:
            missing_rates = [missing_rate]
        metrics_by_rate = {}
        for rate in missing_rates:
            metrics_by_rate[str(rate)] = _run_eval(
                op_code, model, test_loader, device, meta_emb_eval, cfg.data.pred_len, missing_rate=rate
            )
        results["metrics_by_rate"] = metrics_by_rate
    elif op_code == "T3":
        results.update(_run_eval("T3", model, test_loader, device, meta_emb_eval, cfg.data.pred_len))
    else:
        raise ValueError(f"Unknown eval.op_code: {op_code}")

    op_id = getattr(cfg.eval, "op_id", "") or getattr(cfg.ids, "op_id", "")
    op_dir = os.path.join(cfg.paths.ops_dir, op_id)
    os.makedirs(op_dir, exist_ok=True)

    with open(os.path.join(op_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

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
