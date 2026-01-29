from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.models import ITransformer
from itransformer.utils.ids import build_run_id
from itransformer.utils.metadata import load_or_build_embeddings


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Resolve run_id (allow override from config interpolation)
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

    # Persist resolved config
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Stub status
    status = {
        "state": "initialized",
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    meta_emb = None
    train_data = None
    train_loader = None
    if cfg.metadata.enabled:
        train_data, train_loader = data_provider(cfg, flag="train")
        sensor_ids = getattr(train_data, "sensor_ids", None)
        if not sensor_ids:
            raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
        meta_emb = load_or_build_embeddings(cfg, sensor_ids)

    model = ITransformer(cfg)

    # Minimal wiring check: run a single forward pass if data is available.
    if train_loader is not None:
        batch = next(iter(train_loader))
        batch_x, _, batch_x_mark, _ = batch
        x_mark = None
        if batch_x_mark is not None:
            x_mark = torch.as_tensor(batch_x_mark).float()
        _ = model(torch.as_tensor(batch_x).float(), x_mark, meta_emb=meta_emb)

    print(f"[train] run_id={run_id}")
    print(f"[train] run_dir={run_dir}")


if __name__ == "__main__":
    main()
