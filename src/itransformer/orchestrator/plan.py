from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf


def load_plan(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


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
