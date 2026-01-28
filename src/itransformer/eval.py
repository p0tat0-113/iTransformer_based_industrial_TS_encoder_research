from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from itransformer.utils.ids import build_op_id


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    op_id = build_op_id(
        cfg.ids.op_id,
        code=cfg.eval.code,
        op_code=cfg.eval.op_code,
        op_hparams=cfg.eval.op_hparams_tag,
        on_run_id=cfg.eval.on_run_id,
    )

    op_dir = os.path.join(cfg.paths.ops_dir, op_id)
    os.makedirs(op_dir, exist_ok=True)

    config_path = os.path.join(op_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    status = {
        "state": "initialized",
        "op_id": op_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(op_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[eval] op_id={op_id}")
    print(f"[eval] op_dir={op_dir}")


if __name__ == "__main__":
    main()
