from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.analysis.cka import linear_cka
from itransformer.analysis.utils import (
    load_run_checkpoint,
    load_run_metrics,
    load_state,
    param_count,
)
from itransformer.data import data_provider
from itransformer.evals.utils import measure_inference_time
from itransformer.models.factory import build_model
from itransformer.utils.ids import build_cmp_id, build_agg_id


def _split_run_ids(value):
    if not value:
        return []
    if isinstance(value, str):
        return [r.strip() for r in value.split(",") if r.strip()]
    return list(value)


def _set_seed(cfg) -> None:
    if getattr(cfg.runtime, "seed", None) is None:
        return
    seed = int(cfg.runtime.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_metrics(data, prefix=""):
    flat = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_metrics(value, full))
    elif isinstance(data, (int, float)):
        flat[prefix] = float(data)
    return flat


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    analysis_code = cfg.analysis.code

    if analysis_code in ("F1", "F2", "F4", "F5"):
        run_ids = _split_run_ids(cfg.analysis.on)
        if len(run_ids) != 1:
            raise ValueError("analysis.on must be a single run_id for F1/F2/F4/F5")
        run_id = run_ids[0]
        report_id = cfg.ids.agg_id or build_agg_id(
            cfg.ids.agg_id,
            dataset=cfg.data.name,
            code=analysis_code,
            run_ids=[run_id],
        )
        out_dir = os.path.join(cfg.paths.agg_dir, report_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

        if getattr(cfg.runtime, "deterministic", False):
            _set_seed(cfg)

        ckpt_path = load_run_checkpoint(run_id, cfg.paths.runs_dir)
        model = build_model(cfg)
        load_state(model, ckpt_path)
        device = torch.device(
            "cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)

        results = {}

        if analysis_code in ("F1", "F2"):
            _, train_loader = data_provider(cfg, flag="train")
            batch = next(iter(train_loader))
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            results["params"] = param_count(model)
            results["ms_per_forward"] = measure_inference_time(model, x_enc, x_mark, repeats=5)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                _ = model(x_enc, x_mark)
                torch.cuda.synchronize()
                results["max_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

            if analysis_code == "F2":
                run_dir = os.path.join(cfg.paths.runs_dir, run_id)
                metrics, source = load_run_metrics(run_dir)
                if metrics:
                    results["metrics_source"] = source
                    results["metrics"] = metrics

        if analysis_code == "F4":
            _, train_loader = data_provider(cfg, flag="train")
            batch = next(iter(train_loader))
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)

            activations = []

            def _hook(_, __, output):
                activations.append(output[0] if isinstance(output, tuple) else output)

            hooks = []
            if hasattr(model, "encoder"):
                for layer in model.encoder.attn_layers:
                    hooks.append(layer.register_forward_hook(_hook))

            _ = model(x_enc, x_mark)
            for h in hooks:
                h.remove()

            if len(activations) >= 2:
                cka = linear_cka(activations[0].detach(), activations[-1].detach())
                results["cka"] = cka
            else:
                results["cka"] = None

        if analysis_code == "F5":
            if hasattr(model, "encoder"):
                for layer in model.encoder.attn_layers:
                    if hasattr(layer.attention, "inner_attention"):
                        layer.attention.inner_attention.output_attention = True
            _, train_loader = data_provider(cfg, flag="train")
            batch = next(iter(train_loader))
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)

            attns = None
            if hasattr(model, "forecast"):
                _, attns = model.forecast(x_enc, x_mark)
            if attns:
                attn = attns[0].detach().cpu().numpy()
                torch.save(attn, os.path.join(out_dir, "attn_first_layer.pt"))
                results["attn_saved"] = True
            else:
                results["attn_saved"] = False

        with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "state": "completed",
                    "report_id": report_id,
                    "analysis": analysis_code,
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        print(f"[analysis] report_id={report_id}")
        print(f"[analysis] out_dir={out_dir}")
        return

    # CMP if left/right are provided, else AGG if on list is provided
    out_dir = None
    report_id = None
    status_state = "initialized"
    if cfg.analysis.left and cfg.analysis.right:
        report_id = build_cmp_id(
            cfg.ids.cmp_id,
            dataset=cfg.data.name,
            code=cfg.analysis.code,
            left=cfg.analysis.left,
            right=cfg.analysis.right,
        )
        out_dir = os.path.join(cfg.paths.cmp_dir, report_id)
        os.makedirs(out_dir, exist_ok=True)

        left_dir = os.path.join(cfg.paths.runs_dir, cfg.analysis.left)
        right_dir = os.path.join(cfg.paths.runs_dir, cfg.analysis.right)
        left_metrics, left_source = load_run_metrics(left_dir)
        right_metrics, right_source = load_run_metrics(right_dir)
        left_flat = _flatten_metrics(left_metrics)
        right_flat = _flatten_metrics(right_metrics)
        delta = {}
        for key in sorted(set(left_flat) & set(right_flat)):
            delta[key] = left_flat[key] - right_flat[key]
        cmp = {
            "left": {"run_id": cfg.analysis.left, "metrics": left_metrics, "source": left_source},
            "right": {"run_id": cfg.analysis.right, "metrics": right_metrics, "source": right_source},
            "delta": delta,
        }
        with open(os.path.join(out_dir, "cmp.json"), "w", encoding="utf-8") as f:
            json.dump(cmp, f, indent=2)
        status_state = "completed"
    else:
        run_ids = []
        if cfg.analysis.on:
            if isinstance(cfg.analysis.on, str):
                run_ids = [r.strip() for r in cfg.analysis.on.split(",") if r.strip()]
            else:
                run_ids = list(cfg.analysis.on)
        report_id = build_agg_id(
            cfg.ids.agg_id,
            dataset=cfg.data.name,
            code=cfg.analysis.code,
            run_ids=run_ids,
        )
        out_dir = os.path.join(cfg.paths.agg_dir, report_id)

    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(out_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    status = {
        "state": status_state,
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[analysis] report_id={report_id}")
    print(f"[analysis] out_dir={out_dir}")


if __name__ == "__main__":
    main()
