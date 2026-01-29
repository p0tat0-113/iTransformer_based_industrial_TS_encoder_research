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


def _run_code_from_cfg(cfg_dict: dict) -> str:
    return cfg_dict.get("run", {}).get("code", "") if isinstance(cfg_dict, dict) else ""


def _run_code_from_run_id(run_id: str) -> str:
    return run_id.split(".")[0] if run_id else ""


def _match_run_code(code: str, prefixes: list[str], codes: list[str]) -> bool:
    if codes and code not in codes:
        return False
    if prefixes and not any(code.startswith(p) for p in prefixes):
        return False
    return True


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


def _extract_mse_mae(metrics: dict) -> dict:
    if not metrics:
        return {"mse": None, "mae": None}
    if "eval" in metrics and isinstance(metrics["eval"], dict):
        test = metrics["eval"].get("test")
        if isinstance(test, dict) and "mse" in test and "mae" in test:
            return {"mse": test.get("mse"), "mae": test.get("mae")}
    if "val" in metrics and isinstance(metrics["val"], dict):
        if "mse" in metrics["val"] and "mae" in metrics["val"]:
            return {"mse": metrics["val"].get("mse"), "mae": metrics["val"].get("mae")}
    if "mse" in metrics and "mae" in metrics:
        return {"mse": metrics.get("mse"), "mae": metrics.get("mae")}
    return {"mse": None, "mae": None}


def _select_cfg_value(cfg_dict: dict, key_path: str):
    cur = cfg_dict
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _iter_run_configs(runs_dir: str):
    if not os.path.isdir(runs_dir):
        return
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        try:
            cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        except Exception:
            continue
        yield name, run_dir, cfg


def _patch_len_from_cfg(cfg_dict: dict):
    patch = cfg_dict.get("model", {}).get("patch", {})
    patch_len = patch.get("patch_len")
    mode = patch.get("mode")
    if (patch_len is None or patch_len == 0) and mode == "mean_pool":
        return cfg_dict.get("data", {}).get("seq_len")
    if "patch_len" in patch:
        return patch_len
    return cfg_dict.get("ssl", {}).get("patch_len")


def _aggregate_rows(rows, group_by, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_by) if group_by else ("__all__",)
        grouped.setdefault(key, []).append(row)

    metric_union = set(metric_keys)
    if not metric_union:
        for row in rows:
            metric_union.update(row["metrics"].keys())

    agg = []
    for key, group_rows in grouped.items():
        entry = {}
        if group_by:
            for idx, k in enumerate(group_by):
                entry[k] = key[idx]
        for m in sorted(metric_union):
            vals = [r["metrics"].get(m) for r in group_rows if isinstance(r["metrics"].get(m), (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
                entry[m] = {"mean": mean, "std": var**0.5}
        agg.append(entry)
    return agg


def _compute_cka_metrics(cfg, run_id: str):
    # Ensure patch_len is set for patch-based models
    if hasattr(cfg, "model") and hasattr(cfg.model, "patch"):
        patch_len = getattr(cfg.model.patch, "patch_len", None)
        if not patch_len:
            mode = getattr(cfg.model.patch, "mode", "")
            if mode == "mean_pool":
                cfg.model.patch.patch_len = cfg.data.seq_len
            else:
                cfg.model.patch.patch_len = getattr(cfg.ssl, "patch_len", None)

    ckpt_path = load_run_checkpoint(run_id, cfg.paths.runs_dir)
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    device = torch.device(
        "cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

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

    if hasattr(model, "forward"):
        _ = model(x_enc, x_mark)

    for h in hooks:
        h.remove()

    if len(activations) < 2:
        return {"first_layer_cka": None, "last_layer_cka": None, "delta_cka": None}

    first = activations[0].detach()
    last = activations[-1].detach()
    first_cka = linear_cka(first, first)
    last_cka = linear_cka(first, last)
    return {
        "first_layer_cka": first_cka,
        "last_layer_cka": last_cka,
        "delta_cka": last_cka - first_cka,
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    analysis_code = cfg.analysis.code

    if analysis_code in ("B-EV-1", "B-EV-2", "B-EV-4"):
        variants = list(getattr(cfg.analysis, "variants", []) or [])
        run_code_prefixes = list(getattr(cfg.analysis, "run_code_prefixes", []) or [])
        run_codes = list(getattr(cfg.analysis, "run_codes", []) or [])
        if not run_code_prefixes and not run_codes:
            if analysis_code in ("B-EV-1", "B-EV-4"):
                run_code_prefixes = ["B-TR"]
            elif analysis_code == "B-EV-2":
                run_code_prefixes = ["B-DS"]
        if analysis_code == "B-EV-1":
            variants = variants or ["P0", "P1", "P2", "P3", "P4"]
            rows = []
            for run_id, run_dir, run_cfg in _iter_run_configs(cfg.paths.runs_dir):
                if run_cfg.get("data", {}).get("name") != cfg.data.name:
                    continue
                if run_cfg.get("model", {}).get("variant") not in variants:
                    continue
                run_code = _run_code_from_cfg(run_cfg)
                if not _match_run_code(run_code, run_code_prefixes, run_codes):
                    continue
                metrics, _ = load_run_metrics(run_dir)
                cost = metrics.get("cost", {})
                if not cost:
                    continue
                row = {
                    "run_id": run_id,
                    "run_code": run_code,
                    "variant": run_cfg.get("model", {}).get("variant"),
                    "patch_len": _patch_len_from_cfg(run_cfg),
                    "seed": run_cfg.get("runtime", {}).get("seed"),
                    "metrics": {
                        "wall_time_sec_total": cost.get("wall_time_sec_total"),
                        "gpu_mem_peak_mb": cost.get("gpu_mem_peak_mb"),
                        "params_count": cost.get("params_count"),
                    },
                }
                rows.append(row)
            agg = _aggregate_rows(rows, ["variant", "patch_len"], [])
        elif analysis_code == "B-EV-2":
            variants = variants or ["P1", "P2", "P3", "P4"]
            rows = []
            for run_id, run_dir, run_cfg in _iter_run_configs(cfg.paths.runs_dir):
                if run_cfg.get("data", {}).get("name") != cfg.data.name:
                    continue
                if run_cfg.get("model", {}).get("variant") not in variants:
                    continue
                if run_cfg.get("train", {}).get("mode") != "lp":
                    continue
                run_code = _run_code_from_cfg(run_cfg)
                if not _match_run_code(run_code, run_code_prefixes, run_codes):
                    continue
                metrics, _ = load_run_metrics(run_dir)
                test = metrics.get("eval", {}).get("test", {})
                if not test:
                    continue
                pre_cost = {}
                ckpt_path = run_cfg.get("train", {}).get("ssl_ckpt_path") or ""
                if ckpt_path:
                    pre_dir = os.path.dirname(ckpt_path)
                    if not os.path.isabs(pre_dir):
                        pre_dir = os.path.abspath(pre_dir)
                    pre_metrics, _ = load_run_metrics(pre_dir)
                    pre_cost = pre_metrics.get("cost", {})
                row = {
                    "run_id": run_id,
                    "run_code": run_code,
                    "variant": run_cfg.get("model", {}).get("variant"),
                    "patch_len": _patch_len_from_cfg(run_cfg),
                    "seed": run_cfg.get("runtime", {}).get("seed"),
                    "metrics": {
                        "test_mse": test.get("mse"),
                        "test_mae": test.get("mae"),
                        "pre_wall_time_sec_total": pre_cost.get("wall_time_sec_total"),
                        "pre_gpu_mem_peak_mb": pre_cost.get("gpu_mem_peak_mb"),
                        "pre_params_count": pre_cost.get("params_count"),
                    },
                }
                rows.append(row)
            agg = _aggregate_rows(rows, ["variant", "patch_len"], [])
        else:  # B-EV-4
            variants = variants or ["P1", "P2", "P3", "P4"]
            rows = []
            for run_id, run_dir, run_cfg in _iter_run_configs(cfg.paths.runs_dir):
                if run_cfg.get("data", {}).get("name") != cfg.data.name:
                    continue
                if run_cfg.get("model", {}).get("variant") not in variants:
                    continue
                run_code = _run_code_from_cfg(run_cfg)
                if not _match_run_code(run_code, run_code_prefixes, run_codes):
                    continue
                run_cfg_obj = OmegaConf.create(run_cfg)
                metrics = _compute_cka_metrics(run_cfg_obj, run_id)
                row = {
                    "run_id": run_id,
                    "run_code": run_code,
                    "variant": run_cfg.get("model", {}).get("variant"),
                    "patch_len": _patch_len_from_cfg(run_cfg),
                    "seed": run_cfg.get("runtime", {}).get("seed"),
                    "metrics": metrics,
                }
                rows.append(row)
            agg = _aggregate_rows(rows, ["variant", "patch_len"], [])

        report_id = build_agg_id(
            cfg.ids.agg_id,
            dataset=cfg.data.name,
            code=analysis_code,
            run_ids=[],
        )
        out_dir = os.path.join(cfg.paths.agg_dir, report_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        payload = {
            "rows": rows,
            "agg": agg,
            "meta": {
                "analysis_code": analysis_code,
                "dataset": cfg.data.name,
                "run_code_prefixes": run_code_prefixes,
                "run_codes": run_codes,
            },
        }
        with open(os.path.join(out_dir, "agg.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "state": "completed",
                    "report_id": report_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        print(f"[analysis] report_id={report_id}")
        print(f"[analysis] out_dir={out_dir}")
        return

    if analysis_code in ("C-RB-1", "C-RB-2"):
        op_code = "R1" if analysis_code == "C-RB-1" else "R2"
        variants = list(getattr(cfg.analysis, "variants", []) or []) or ["P0", "P1", "P2", "P3", "P4"]
        group_by = list(getattr(cfg.analysis, "group_by", []) or []) or ["variant", "patch_len"]
        run_code_prefixes = list(getattr(cfg.analysis, "run_code_prefixes", []) or [])
        run_codes = list(getattr(cfg.analysis, "run_codes", []) or [])
        if not run_code_prefixes and not run_codes:
            run_code_prefixes = ["C-DS"]
        rows = []
        if os.path.isdir(cfg.paths.ops_dir):
            for name in sorted(os.listdir(cfg.paths.ops_dir)):
                op_dir = os.path.join(cfg.paths.ops_dir, name)
                if not os.path.isdir(op_dir):
                    continue
                cfg_path = os.path.join(op_dir, "config.yaml")
                res_path = os.path.join(op_dir, "op_results.json")
                if not os.path.exists(cfg_path) or not os.path.exists(res_path):
                    continue
                try:
                    op_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
                except Exception:
                    continue
                if op_cfg.get("data", {}).get("name") != cfg.data.name:
                    continue
                if op_cfg.get("eval", {}).get("op_code") != op_code:
                    continue
                if op_cfg.get("model", {}).get("variant") not in variants:
                    continue
                run_id = op_cfg.get("eval", {}).get("on_run_id", "")
                run_code = _run_code_from_run_id(run_id)
                if not _match_run_code(run_code, run_code_prefixes, run_codes):
                    continue
                try:
                    with open(res_path, "r", encoding="utf-8") as f:
                        op_results = json.load(f)
                except Exception:
                    continue
                metrics_by_rate = op_results.get("metrics_by_rate")
                if not isinstance(metrics_by_rate, dict):
                    continue
                row = {
                    "op_id": name,
                    "run_id": run_id,
                    "run_code": run_code,
                    "variant": op_cfg.get("model", {}).get("variant"),
                    "patch_len": _patch_len_from_cfg(op_cfg),
                    "seed": op_cfg.get("runtime", {}).get("seed"),
                    "metrics_by_rate": metrics_by_rate,
                }
                rows.append(row)

        grouped = {}
        for row in rows:
            key = tuple(row.get(k) for k in group_by) if group_by else ("__all__",)
            grouped.setdefault(key, []).append(row)

        agg = []
        for key, group_rows in grouped.items():
            entry = {}
            if group_by:
                for idx, k in enumerate(group_by):
                    entry[k] = key[idx]
            rates = {}
            for row in group_rows:
                for rate, metrics in row.get("metrics_by_rate", {}).items():
                    rates.setdefault(rate, {}).setdefault("mse", []).append(metrics.get("mse"))
                    rates.setdefault(rate, {}).setdefault("mae", []).append(metrics.get("mae"))
            metrics_by_rate = {}
            for rate, vals in rates.items():
                metrics_by_rate[rate] = {}
                for k, series in vals.items():
                    series = [v for v in series if isinstance(v, (int, float))]
                    if series:
                        mean = sum(series) / len(series)
                        var = sum((v - mean) ** 2 for v in series) / max(1, len(series))
                        metrics_by_rate[rate][k] = {"mean": mean, "std": var**0.5}
            entry["metrics_by_rate"] = metrics_by_rate
            agg.append(entry)

        report_id = build_agg_id(
            cfg.ids.agg_id,
            dataset=cfg.data.name,
            code=analysis_code,
            run_ids=[],
        )
        out_dir = os.path.join(cfg.paths.agg_dir, report_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        payload = {
            "rows": rows,
            "agg": agg,
            "meta": {
                "analysis_code": analysis_code,
                "dataset": cfg.data.name,
                "run_code_prefixes": run_code_prefixes,
                "run_codes": run_codes,
            },
        }
        with open(os.path.join(out_dir, "agg.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "state": "completed",
                    "report_id": report_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        print(f"[analysis] report_id={report_id}")
        print(f"[analysis] out_dir={out_dir}")
        return

    if analysis_code in ("F1", "F2", "F4", "F5"):
        run_ids = _split_run_ids(cfg.analysis.on)
        if len(run_ids) != 1:
            raise ValueError("analysis.on must be a single run_id for F1/F2/F4/F5")
        run_id = run_ids[0]
        report_id = build_agg_id(
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
    run_ids = []
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
        left_metrics, _ = load_run_metrics(left_dir)
        right_metrics, _ = load_run_metrics(right_dir)
        left_vals = _extract_mse_mae(left_metrics)
        right_vals = _extract_mse_mae(right_metrics)
        delta = {
            "delta_mse": None,
            "delta_mae": None,
        }
        if left_vals["mse"] is not None and right_vals["mse"] is not None:
            delta["delta_mse"] = left_vals["mse"] - right_vals["mse"]
        if left_vals["mae"] is not None and right_vals["mae"] is not None:
            delta["delta_mae"] = left_vals["mae"] - right_vals["mae"]
        cmp = {
            "left": {"run_id": cfg.analysis.left, **left_vals},
            "right": {"run_id": cfg.analysis.right, **right_vals},
            "delta": delta,
            "meta": {
                "analysis_code": cfg.analysis.code,
                "dataset": cfg.data.name,
            },
        }
        with open(os.path.join(out_dir, "cmp.json"), "w", encoding="utf-8") as f:
            json.dump(cmp, f, indent=2)
        status_state = "completed"
    else:
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

    # AGG: build rows + agg if run_ids are provided
    if run_ids:
        group_by = list(getattr(cfg.analysis, "group_by", []) or [])
        metric_keys = list(getattr(cfg.analysis, "metric_keys", []) or [])
        rows = []
        for run_id in run_ids:
            run_dir = os.path.join(cfg.paths.runs_dir, run_id)
            metrics, _ = load_run_metrics(run_dir)
            flat = _flatten_metrics(metrics)
            row_metrics = {k: flat.get(k) for k in metric_keys} if metric_keys else flat
            row = {"run_id": run_id, "metrics": row_metrics}
            # include grouping fields from run config
            cfg_path = os.path.join(run_dir, "config.yaml")
            if os.path.exists(cfg_path):
                run_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
                row["run_code"] = _run_code_from_cfg(run_cfg)
                for key in group_by:
                    row[key] = _select_cfg_value(run_cfg, key)
            rows.append(row)

        # aggregate (mean/std) by group key
        agg = []
        grouped = {}
        for row in rows:
            key = tuple(row.get(k) for k in group_by) if group_by else ("__all__",)
            grouped.setdefault(key, []).append(row)

        metric_union = set(metric_keys)
        if not metric_union:
            for row in rows:
                metric_union.update(row["metrics"].keys())
        for key, group_rows in grouped.items():
            entry = {}
            if group_by:
                for idx, k in enumerate(group_by):
                    entry[k] = key[idx]
            for m in sorted(metric_union):
                vals = [r["metrics"].get(m) for r in group_rows if isinstance(r["metrics"].get(m), (int, float))]
                if vals:
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
                    entry[m] = {"mean": mean, "std": var**0.5}
            agg.append(entry)

        payload = {
            "rows": rows,
            "agg": agg,
            "meta": {
                "analysis_code": cfg.analysis.code,
                "dataset": cfg.data.name,
            },
        }
        with open(os.path.join(out_dir, "agg.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        status_state = "completed"

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
