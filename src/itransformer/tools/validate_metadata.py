from __future__ import annotations

import sys

import hydra

from itransformer.data import data_provider
from itransformer.utils.metadata import validate_metadata_jsonl


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg) -> None:
    data_set, _ = data_provider(cfg, flag="train")
    sensor_ids = getattr(data_set, "sensor_ids", None)
    if not sensor_ids:
        raise ValueError("Dataset does not expose sensor_ids for metadata matching.")

    errors, warnings = validate_metadata_jsonl(
        cfg.metadata.path,
        sensor_ids,
        cfg.metadata.template,
        strict=cfg.metadata_builder.strict,
    )

    if warnings:
        print(f"[metadata] warnings: {len(warnings)}")
        for msg in warnings[:20]:
            print(f"  - {msg}")
        if len(warnings) > 20:
            print("  - ...")

    if errors:
        print(f"[metadata] errors: {len(errors)}")
        for msg in errors[:20]:
            print(f"  - {msg}")
        sys.exit(1)

    print("[metadata] validation passed")


if __name__ == "__main__":
    main()
