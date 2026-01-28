from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from itransformer.utils.ids import build_run_id


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Resolve run_id (allow override from config interpolation)
    run_id = cfg.run.id
    if not run_id:
        run_id = build_run_id(
            cfg.ids.run_id,
            code=cfg.run.code,
            dataset=cfg.data.name,
            variant=cfg.model.variant,
            hparams_tag=cfg.run.hparams_tag,
            seed=cfg.runtime.seed,
        )

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

    print(f"[train] run_id={run_id}")
    print(f"[train] run_dir={run_dir}")


if __name__ == "__main__":
    main()
