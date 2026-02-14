from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional


ENTRYPOINTS = {
    "pretrain": "itransformer.pretrain",
    "downstream": "itransformer.downstream",
    "train": "itransformer.train",
    "eval": "itransformer.eval",
    "analysis": "itransformer.analysis_entry",
}


def _status_path(artifacts_root: str, kind: str, spec_id: str, *, plan_id: str | None = None) -> str:
    if kind == "run" and plan_id:
        return os.path.join(artifacts_root, "runs", plan_id, spec_id, "status.json")
    base = {
        "run": "runs",
        "op": "ops",
        "cmp": "cmp",
        "agg": "agg",
    }[kind]
    return os.path.join(artifacts_root, base, spec_id, "status.json")


def _load_status(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _override_key_exists(overrides: List[str], key: str) -> bool:
    prefix = f"{key}="
    return any(str(item).startswith(prefix) for item in (overrides or []))


def _run_spec(spec: Dict[str, str], log_path: str, extra_overrides: Optional[List[str]] = None) -> int:
    entry = spec["entry"]
    module = ENTRYPOINTS.get(entry, entry)
    overrides = list(spec.get("overrides", []) or [])
    if extra_overrides:
        overrides.extend(extra_overrides)
    cmd = ["python", "-m", module] + overrides
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=logf)
    return proc.returncode


def execute_plan(
    plan_id: str,
    specs: Dict[str, List[Dict[str, str]]],
    artifacts_root: str,
    resume: bool = False,
    only: Optional[str] = None,
    filter_substr: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    out_dir = os.path.join(artifacts_root, "plans", plan_id)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    manifest = {"plan_id": plan_id, "timestamp": datetime.now(timezone.utc).isoformat(), "specs": []}
    specs_dir = os.path.join(out_dir, "specs")
    os.makedirs(specs_dir, exist_ok=True)
    with open(os.path.join(specs_dir, "specs.json"), "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)

    order = [("run", specs.get("runs", [])), ("op", specs.get("ops", [])), ("cmp", specs.get("cmps", [])), ("agg", specs.get("aggs", []))]

    for kind, items in order:
        if only and kind != only:
            continue
        for spec in items:
            spec_id = spec.get("id")
            entry = spec.get("entry")
            if not spec_id:
                raise ValueError(f"Spec missing id for kind={kind}")
            if filter_substr and filter_substr not in spec_id:
                continue

            status_path = _status_path(artifacts_root, kind, spec_id, plan_id=plan_id)
            status = _load_status(status_path)
            if resume and status and status.get("state") == "completed":
                manifest["specs"].append(
                    {"id": spec_id, "kind": kind, "entry": entry, "status": "skipped"}
                )
                continue

            extra_overrides: List[str] = []
            if kind == "run":
                # Keep runs grouped by plan to avoid a flat runs/ directory.
                run_overrides = list(spec.get("overrides", []) or [])
                if not _override_key_exists(run_overrides, "paths.runs_dir"):
                    plan_runs_dir = os.path.join(artifacts_root, "runs", plan_id)
                    extra_overrides.append(f"paths.runs_dir={plan_runs_dir}")

            log_path = os.path.join(logs_dir, f"{kind}__{spec_id}.log")
            ret = _run_spec(spec, log_path, extra_overrides=extra_overrides)
            manifest["specs"].append(
                {
                    "id": spec_id,
                    "kind": kind,
                    "entry": entry,
                    "status": "completed" if ret == 0 else "failed",
                    "returncode": ret,
                    "log": log_path,
                }
            )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
