from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _parse_list(value: str, cast=int):
    if value is None or value == "":
        return []
    items = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(cast(token))
    return items


def _build_run_id(run_code: str, dataset: str, variant: str, hparams_tag: str, seed: int) -> str:
    return f"{run_code}.{dataset}.{variant}.{hparams_tag}.sd{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal patch_len sweeper for B experiments.")
    parser.add_argument("--data", default="ETTh1", help="Dataset name (e.g., ETTh1).")
    parser.add_argument("--variants", default="P1,P2,P3,P4", help="Comma list of model variants.")
    parser.add_argument("--patch-lens", default="8,16,32,64", help="Comma list of patch_len values.")
    parser.add_argument("--seeds", default="0", help="Comma list of seeds.")
    parser.add_argument("--run-code", default="B-TR", help="run.code prefix.")
    parser.add_argument("--hparam-tag", default="pl{patch_len}", help="Template for run.hparams_tag.")
    parser.add_argument("--epochs", type=int, default=1, help="pretrain epochs per run.")
    parser.add_argument("--device", default="cpu", help="runtime.device (cpu/cuda).")
    parser.add_argument("--num-workers", type=int, default=0, help="train.num_workers.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs.")
    parser.add_argument("--sweep-dir", default="./artifacts/sweeps", help="Sweep output directory.")
    args = parser.parse_args()

    variants = _parse_list(args.variants, cast=str)
    patch_lens = _parse_list(args.patch_lens, cast=int)
    seeds = _parse_list(args.seeds, cast=int)

    sweep_id = f"sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = os.path.join(args.sweep_dir, sweep_id)
    logs_dir = os.path.join(sweep_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    manifest = {
        "sweep_id": sweep_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data": args.data,
            "variants": variants,
            "patch_lens": patch_lens,
            "seeds": seeds,
            "run_code": args.run_code,
            "hparam_tag": args.hparam_tag,
            "epochs": args.epochs,
            "device": args.device,
            "num_workers": args.num_workers,
        },
        "runs": [],
    }

    for variant in variants:
        for patch_len in patch_lens:
            for seed in seeds:
                hparams_tag = args.hparam_tag.format(patch_len=patch_len, variant=variant, seed=seed)
                run_id = _build_run_id(args.run_code, args.data, variant, hparams_tag, seed)
                run_dir = os.path.join("./artifacts/runs", run_id)
                status_path = os.path.join(run_dir, "status.json")
                if args.resume and os.path.exists(status_path):
                    try:
                        with open(status_path, "r", encoding="utf-8") as f:
                            status = json.load(f)
                        if status.get("state") == "completed":
                            manifest["runs"].append(
                                {
                                    "run_id": run_id,
                                    "variant": variant,
                                    "patch_len": patch_len,
                                    "seed": seed,
                                    "status": "skipped",
                                }
                            )
                            continue
                    except Exception:
                        pass

                cmd = [
                    sys.executable,
                    "-m",
                    "itransformer.pretrain",
                    f"data={args.data}",
                    f"model={variant}",
                    "ssl=patch_mae",
                    f"ssl.patch_len={patch_len}",
                    f"ssl.pretrain_epochs={args.epochs}",
                    "metadata.enabled=false",
                    f"runtime.device={args.device}",
                    f"runtime.seed={seed}",
                    f"train.num_workers={args.num_workers}",
                    f"run.code={args.run_code}",
                    f"run.hparams_tag={hparams_tag}",
                ]

                entry = {
                    "run_id": run_id,
                    "variant": variant,
                    "patch_len": patch_len,
                    "seed": seed,
                    "cmd": " ".join(cmd),
                    "status": "pending",
                }
                manifest["runs"].append(entry)

                if args.dry_run:
                    entry["status"] = "dry_run"
                    continue

                log_path = os.path.join(logs_dir, f"{run_id}.log")
                with open(log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.run(cmd, stdout=logf, stderr=logf)
                entry["status"] = "completed" if proc.returncode == 0 else "failed"
                entry["returncode"] = proc.returncode
                entry["log"] = log_path

    with open(os.path.join(sweep_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[sweep] sweep_id={sweep_id}")
    print(f"[sweep] manifest={os.path.join(sweep_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
