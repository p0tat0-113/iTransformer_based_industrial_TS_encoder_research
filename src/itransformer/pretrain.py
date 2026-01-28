from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.ssl import PatchMAE, VarMAE
from itransformer.utils.ids import build_run_id
from itransformer.utils.metadata import load_or_build_embeddings


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
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

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_data, train_loader = data_provider(cfg, flag="train")
    sensor_ids = getattr(train_data, "sensor_ids", None)
    if not sensor_ids:
        raise ValueError("Dataset does not expose sensor_ids for metadata matching.")

    meta_emb = None
    if cfg.metadata.enabled:
        meta_emb = load_or_build_embeddings(cfg, sensor_ids)
        meta_emb = meta_emb.to(device)

    if cfg.ssl.type == "var_mae":
        model = VarMAE(cfg)
    elif cfg.ssl.type == "patch_mae":
        model = PatchMAE(cfg)
    else:
        raise ValueError(f"Unsupported ssl.type: {cfg.ssl.type}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    losses = []
    for epoch in range(cfg.ssl.pretrain_epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch_x, _, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss = model(x_enc, x_mark, meta_emb=meta_emb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().item())

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        losses.append(avg_loss)
        print(f"[pretrain] epoch={epoch+1} loss={avg_loss:.6f}")

    ckpt_path = os.path.join(run_dir, "pretrain_checkpoint.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, ckpt_path)

    metrics_path = os.path.join(run_dir, "pretrain_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"loss": losses}, f, indent=2)

    status = {
        "state": "completed",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print(f"[pretrain] run_id={run_id}")
    print(f"[pretrain] run_dir={run_dir}")


if __name__ == "__main__":
    main()
