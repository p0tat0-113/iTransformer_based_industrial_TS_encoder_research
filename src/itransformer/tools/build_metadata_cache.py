from __future__ import annotations

import json
import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.utils.metadata import (
    build_texts,
    load_metadata_jsonl,
    load_or_build_embeddings,
    validate_metadata_jsonl,
)


def _write_texts(path: str, sensor_ids, texts) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sensor_id, text in zip(sensor_ids, texts):
            f.write(json.dumps({"sensor_id": str(sensor_id), "text": text}, ensure_ascii=False))
            f.write("\n")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg) -> None:
    # Load dataset to get sensor_ids
    train_data, _ = data_provider(cfg, flag="train")
    sensor_ids = getattr(train_data, "sensor_ids", None)
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
        for msg in warnings[:5]:
            print(f"  - {msg}")
        if len(warnings) > 5:
            print("  - ...")
    if errors:
        raise ValueError("\n".join(errors[:10]))

    meta_map = load_metadata_jsonl(cfg.metadata.path)

    texts = build_texts(
        sensor_ids,
        meta_map,
        cfg.metadata.template,
        cfg.metadata.unk_token,
        cfg.metadata.unk_template,
    )

    if cfg.metadata_builder.dump_texts:
        _write_texts(cfg.metadata_builder.texts_path, sensor_ids, texts)
        print(f"[metadata] texts written: {cfg.metadata_builder.texts_path}")

    if cfg.metadata.cache.build:
        try:
            _ = load_or_build_embeddings(cfg, sensor_ids)
            print(f"[metadata] embeddings cached: {cfg.metadata.cache.path}")
        except NotImplementedError as exc:
            print("[metadata] embedding provider not implemented:", exc)
            print("[metadata] You can still use --metadata_builder.dump_texts=true to export texts.")
    else:
        print("[metadata] cache.build=false; skipped embedding build.")

    status = {
        "state": "completed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": cfg.data.name,
        "data_path": cfg.data.data_path,
        "metadata_path": cfg.metadata.path,
    }
    status_path = os.path.join(cfg.paths.artifacts_root, "metadata", "build_status.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
