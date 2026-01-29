import json
import os
import torch


def load_run_checkpoint(run_id: str, runs_dir: str) -> str:
    run_dir = os.path.join(runs_dir, run_id)
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
