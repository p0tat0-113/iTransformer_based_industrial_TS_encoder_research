from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf


_KNOWN_ENTRIES = {"pretrain", "downstream", "train", "eval", "analysis"}


def load_plan(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    plan = OmegaConf.to_container(cfg, resolve=True)
    validate_plan(plan)
    return plan


def _override_keys(overrides: List[str]) -> List[str]:
    keys = []
    for item in overrides or []:
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _override_map(overrides: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in overrides or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            mapping[key] = value.strip()
    return mapping


def _require_keys(keys: List[str], required: List[str], context: str) -> None:
    missing = [k for k in required if k not in keys]
    if missing:
        raise ValueError(f"{context}: missing overrides {missing}")


def _validate_item(item: Dict[str, Any], kind: str) -> None:
    entry = item.get("entry")
    if not entry:
        raise ValueError(f"{kind}: missing entry")
    if entry not in _KNOWN_ENTRIES:
        raise ValueError(f"{kind}: unknown entry '{entry}'")
    if not item.get("id") and not item.get("id_template"):
        raise ValueError(f"{kind}: missing id or id_template")

    overrides = item.get("overrides", [])
    if overrides is not None and not isinstance(overrides, list):
        raise ValueError(f"{kind}: overrides must be a list")
    keys = _override_keys(overrides)
    override_map = _override_map(overrides)

    if kind == "run":
        _require_keys(keys, ["data", "model"], f"{kind}")
    elif kind == "op":
        _require_keys(keys, ["data", "eval.op_code"], f"{kind}")
        if "eval.on_run_id" not in keys and "eval.ckpt_path" not in keys:
            raise ValueError(f"{kind}: requires eval.on_run_id or eval.ckpt_path")
    elif kind in ("cmp", "agg"):
        _require_keys(keys, ["data", "analysis.code"], f"{kind}")
        if "analysis.left" in keys or "analysis.right" in keys:
            _require_keys(keys, ["analysis.left", "analysis.right"], f"{kind}")
        else:
            code = override_map.get("analysis.code", "")
            allow_no_on = code in {"B-EV-1", "B-EV-2", "B-EV-4", "C-RB-1", "C-RB-2"}
            if "analysis.on" not in keys and not allow_no_on:
                raise ValueError(f"{kind}: requires analysis.on or analysis.left/right")


def validate_plan(plan: Dict[str, Any]) -> None:
    for item in plan.get("runs", []) or []:
        _validate_item(item, "run")
    for item in plan.get("ops", []) or []:
        _validate_item(item, "op")
    for item in plan.get("cmps", []) or []:
        _validate_item(item, "cmp")
    for item in plan.get("aggs", []) or []:
        _validate_item(item, "agg")


def _cartesian(choices: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(choices.keys())
    values = [choices[k] for k in keys]
    for combo in itertools.product(*values):
        yield {k: combo[i] for i, k in enumerate(keys)}


def _format_list(values: List[str], fmt_vars: Dict[str, Any]) -> List[str]:
    rendered = []
    for item in values:
        rendered.append(item.format(**fmt_vars))
    return rendered


def _expand_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded = []
    for item in items or []:
        sweep = item.get("sweep")
        if not sweep:
            expanded.append(item)
            continue

        id_template = item.get("id_template")
        if not id_template:
            raise ValueError("sweep requires id_template")

        for combo in _cartesian(sweep):
            new_item = {k: v for k, v in item.items() if k not in ("sweep",)}
            run_id = id_template.format(**combo)
            fmt_vars = dict(combo)
            fmt_vars["id"] = run_id
            overrides = _format_list(new_item.get("overrides", []), fmt_vars)
            new_item["id"] = run_id
            new_item["overrides"] = overrides
            expanded.append(new_item)
    return expanded


def build_specs(plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    runs = _expand_items(plan.get("runs", []))
    ops = _expand_items(plan.get("ops", []))
    cmps = _expand_items(plan.get("cmps", []))
    aggs = _expand_items(plan.get("aggs", []))
    return {"runs": runs, "ops": ops, "cmps": cmps, "aggs": aggs}
