#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RUN_MARKERS = (
    "config.yaml",
    "downstream_checkpoint.pt",
    "pretrain_checkpoint.pt",
    "checkpoint.pt",
    "downstream_metrics.json",
    "metrics.json",
    "pretrain_metrics.json",
    "status.json",
)


def _looks_like_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in RUN_MARKERS)


def _load_run_to_plan_map(plans_dir: Path) -> tuple[dict[str, set[str]], int]:
    run_to_plans: dict[str, set[str]] = {}
    plan_count = 0
    for plan_dir in sorted(plans_dir.iterdir()) if plans_dir.exists() else []:
        if not plan_dir.is_dir():
            continue
        specs_path = plan_dir / "specs" / "specs.json"
        if not specs_path.exists():
            continue
        try:
            payload = json.loads(specs_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        plan_count += 1
        for item in payload.get("runs", []) or []:
            run_id = item.get("id")
            if not run_id:
                continue
            run_to_plans.setdefault(str(run_id), set()).add(plan_dir.name)
    return run_to_plans, plan_count


def _collect_flat_runs(runs_dir: Path) -> list[Path]:
    flat: list[Path] = []
    for child in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if _looks_like_run_dir(child):
            flat.append(child)
    return flat


def _print_examples(title: str, rows: list[str], max_print: int) -> None:
    print(f"\n[{title}] {len(rows)}")
    if not rows:
        return
    for line in rows[:max_print]:
        print(f"  - {line}")
    if len(rows) > max_print:
        print(f"  ... (+{len(rows) - max_print} more)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move flat runs (artifacts/runs/<run_id>) into plan-grouped layout "
        "(artifacts/runs/<plan_id>/<run_id>) using plan specs mappings."
    )
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root directory")
    parser.add_argument("--runs-dir", default=None, help="Runs directory (default: <artifacts-root>/runs)")
    parser.add_argument("--plans-dir", default=None, help="Plans directory (default: <artifacts-root>/plans)")
    parser.add_argument("--apply", action="store_true", help="Actually move directories (default: dry-run)")
    parser.add_argument(
        "--max-print",
        type=int,
        default=30,
        help="Maximum example rows to print per category",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    runs_dir = Path(args.runs_dir) if args.runs_dir else artifacts_root / "runs"
    plans_dir = Path(args.plans_dir) if args.plans_dir else artifacts_root / "plans"

    run_to_plans, scanned_plans = _load_run_to_plan_map(plans_dir)
    flat_runs = _collect_flat_runs(runs_dir)

    moves: list[tuple[Path, Path]] = []
    ambiguous: list[str] = []
    unmapped: list[str] = []
    conflicts: list[str] = []

    for src_dir in flat_runs:
        run_id = src_dir.name
        plans = sorted(run_to_plans.get(run_id, set()))
        if not plans:
            unmapped.append(run_id)
            continue
        if len(plans) > 1:
            ambiguous.append(f"{run_id} -> {plans}")
            continue
        plan_id = plans[0]
        dst_dir = runs_dir / plan_id / run_id
        if dst_dir.exists():
            conflicts.append(f"{run_id} -> {dst_dir}")
            continue
        moves.append((src_dir, dst_dir))

    print(f"scanned plans: {scanned_plans}")
    print(f"run ids in plan specs: {len(run_to_plans)}")
    print(f"flat run dirs found: {len(flat_runs)}")
    print(f"move candidates: {len(moves)}")

    _print_examples(
        "MOVE",
        [f"{src.name} -> {dst.parent.name}/{dst.name}" for src, dst in moves],
        max(0, int(args.max_print)),
    )
    _print_examples("UNMAPPED", unmapped, max(0, int(args.max_print)))
    _print_examples("AMBIGUOUS", ambiguous, max(0, int(args.max_print)))
    _print_examples("CONFLICT", conflicts, max(0, int(args.max_print)))

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to move directories.")
        return

    moved = 0
    failed: list[str] = []
    for src_dir, dst_dir in moves:
        try:
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_dir), str(dst_dir))
            moved += 1
        except Exception as exc:
            failed.append(f"{src_dir.name}: {exc}")

    print(f"\nMoved {moved}/{len(moves)} run directories.")
    if failed:
        _print_examples("FAILED", failed, max(0, int(args.max_print)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
