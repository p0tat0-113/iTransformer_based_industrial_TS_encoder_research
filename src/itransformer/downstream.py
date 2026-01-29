from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import hydra
import torch
from omegaconf import OmegaConf

from itransformer.data import data_provider
from itransformer.models.factory import build_model
from itransformer.utils.metadata import load_or_build_embeddings
from itransformer.utils.metrics import mae, mse


def _freeze_modules(modules, freeze: bool) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad = not freeze


def _load_ssl_checkpoint(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    filtered = {k: v for k, v in state.items() if not k.startswith("projector.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[downstream] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[downstream] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")


def _evaluate(model, loader, device, pred_len, meta_emb=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            if meta_emb is None:
                out = model(x_enc, x_mark)
            else:
                out = model(x_enc, x_mark, meta_emb)
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            true = true[:, -pred_len:, :]
            preds.append(out)
            trues.append(true)
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    return {
        "mse": mse(pred, true).item(),
        "mae": mae(pred, true).item(),
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    run_id = cfg.run.id
    run_dir = os.path.join(cfg.paths.runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if cfg.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_data, train_loader = data_provider(cfg, flag="train")
    val_data, val_loader = data_provider(cfg, flag="val")
    test_data, test_loader = data_provider(cfg, flag="test")

    model = build_model(cfg)
    model = model.to(device)

    meta_emb = None
    if getattr(cfg.metadata, "enabled", False):
        sensor_ids = getattr(train_data, "sensor_ids", None)
        if not sensor_ids:
            raise ValueError("Dataset does not expose sensor_ids for metadata matching.")
        meta_emb = load_or_build_embeddings(cfg, sensor_ids).to(device)

    if cfg.train.mode in ("ft", "lp"):
        if not cfg.train.ssl_ckpt_path:
            raise ValueError("train.ssl_ckpt_path is required for ft/lp")
        _load_ssl_checkpoint(model, cfg.train.ssl_ckpt_path)

    # Freeze rules
    if hasattr(model, "enc_embedding"):
        freeze_modules = [model.enc_embedding, model.encoder]
    else:
        # patch model
        freeze_modules = [model.value_proj, model.encoder]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(cfg.train.epochs):
        if cfg.train.mode == "lp":
            _freeze_modules(freeze_modules, True)
        elif cfg.train.mode == "ft" and cfg.train.freeze_epochs > 0:
            _freeze_modules(freeze_modules, epoch < cfg.train.freeze_epochs)
        else:
            _freeze_modules(freeze_modules, False)

        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch_x, batch_y, batch_x_mark, _ = batch
            x_enc = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
            x_mark = None
            if batch_x_mark is not None:
                x_mark = torch.as_tensor(batch_x_mark, dtype=torch.float32, device=device)
            true = torch.as_tensor(batch_y, dtype=torch.float32, device=device)
            true = true[:, -cfg.data.pred_len :, :]

            optimizer.zero_grad()
            if meta_emb is None:
                out = model(x_enc, x_mark)
            else:
                out = model(x_enc, x_mark, meta_emb)
            loss = criterion(out, true)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().item())

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"[downstream] epoch={epoch+1} loss={avg_loss:.6f}")

    metrics = {
        "val": _evaluate(model, val_loader, device, cfg.data.pred_len, meta_emb=meta_emb),
        "test": _evaluate(model, test_loader, device, cfg.data.pred_len, meta_emb=meta_emb),
    }

    with open(os.path.join(run_dir, "downstream_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)},
               os.path.join(run_dir, "downstream_checkpoint.pt"))

    with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "state": "completed",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"[downstream] run_id={run_id}")
    print(f"[downstream] run_dir={run_dir}")


if __name__ == "__main__":
    main()
