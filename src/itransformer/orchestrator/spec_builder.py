from __future__ import annotations

from typing import Any, Dict, List

from itransformer.orchestrator.plan import build_specs


def build(plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return build_specs(plan)
