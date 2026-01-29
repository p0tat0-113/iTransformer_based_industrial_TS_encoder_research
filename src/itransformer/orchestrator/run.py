from __future__ import annotations

import argparse
import os

from itransformer.orchestrator.executor import execute_plan
from itransformer.orchestrator.plan import build_specs, load_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment plan orchestrator.")
    parser.add_argument("plan", nargs="?", help="Path to exp_plan.yaml or plan=<path>")
    parser.add_argument("--plan", dest="plan_flag", help="Path to exp_plan.yaml")
    parser.add_argument("--resume", action="store_true", help="Skip completed specs")
    parser.add_argument("--only", choices=["run", "op", "cmp", "agg"], help="Run only specific spec type")
    parser.add_argument("--filter", dest="filter_substr", default=None, help="Substring filter for spec id")
    parser.add_argument("--artifacts-root", default="./artifacts", help="Artifacts root directory")
    args = parser.parse_args()

    plan_path = args.plan_flag or args.plan
    if plan_path and plan_path.startswith("plan="):
        plan_path = plan_path.split("=", 1)[1]
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    plan = load_plan(plan_path)
    plan_id = plan.get("plan_id") or "plan"
    specs = build_specs(plan)

    execute_plan(
        plan_id=plan_id,
        specs=specs,
        artifacts_root=args.artifacts_root,
        resume=args.resume,
        only=args.only,
        filter_substr=args.filter_substr,
    )


if __name__ == "__main__":
    main()
