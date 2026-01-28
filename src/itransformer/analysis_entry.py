from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from itransformer.utils.ids import build_cmp_id, build_agg_id


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    out_dir = None
    report_id = None

    # CMP if left/right are provided, else AGG if on list is provided
    if cfg.analysis.left and cfg.analysis.right:
        report_id = build_cmp_id(
            cfg.ids.cmp_id,
            dataset=cfg.data.name,
            code=cfg.analysis.code,
            left=cfg.analysis.left,
            right=cfg.analysis.right,
        )
        out_dir = os.path.join(cfg.paths.cmp_dir, report_id)
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
        "state": "initialized",
        "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[analysis] report_id={report_id}")
    print(f"[analysis] out_dir={out_dir}")


if __name__ == "__main__":
    main()
