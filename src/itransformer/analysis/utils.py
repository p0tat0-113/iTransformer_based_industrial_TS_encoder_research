import json
import os
import torch


def _candidate_run_dirs(runs_dir: str, run_id: str) -> list[str]:
    candidates: list[str] = []
    direct = os.path.join(runs_dir, run_id)
    if os.path.isdir(direct):
        candidates.append(direct)

    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            nested = os.path.join(runs_dir, name, run_id)
            if os.path.isdir(nested):
                candidates.append(nested)

    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        uniq.append(path)
    return uniq


def resolve_run_dir(run_id: str, runs_dir: str) -> str:
    matches = _candidate_run_dirs(runs_dir, run_id)
    if not matches:
        raise FileNotFoundError(
            f"Run directory not found for run_id={run_id} under runs_dir={runs_dir}"
        )
    if len(matches) > 1:
        preview = ", ".join(matches[:3])
        more = "" if len(matches) <= 3 else f" ... (+{len(matches)-3} more)"
        raise RuntimeError(
            "Ambiguous run_id lookup; multiple directories match "
            f"run_id={run_id}: {preview}{more}"
        )
    return matches[0]


def iter_run_dirs(runs_dir: str):
    if not os.path.isdir(runs_dir):
        return
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(cfg_path):
            yield name, run_dir
            continue
        # Support grouped layout: runs/<plan_id>/<run_id>/config.yaml
        for sub in sorted(os.listdir(run_dir)):
            nested_dir = os.path.join(run_dir, sub)
            if not os.path.isdir(nested_dir):
                continue
            nested_cfg = os.path.join(nested_dir, "config.yaml")
            if os.path.exists(nested_cfg):
                yield sub, nested_dir


def load_run_checkpoint(run_id: str, runs_dir: str) -> str:
    run_dir = resolve_run_dir(run_id, runs_dir)
    candidates = [
        os.path.join(run_dir, "downstream_checkpoint.pt"),
        os.path.join(run_dir, "pretrain_checkpoint.pt"),
        os.path.join(run_dir, "checkpoint.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def load_state(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)


def load_run_metrics(run_dir: str):
    candidates = [
        "downstream_metrics.json",
        "metrics.json",
        "pretrain_metrics.json",
    ]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), name
    return {}, ""
