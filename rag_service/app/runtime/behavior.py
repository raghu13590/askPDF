"""Framework-neutral orchestration decisions derived from runtime metadata."""

from __future__ import annotations

from typing import Any, Mapping


def runtime_behavior(run: Any) -> Mapping[str, Any]:
    metadata = getattr(run, "run_metadata_json", None) or {}
    value = metadata.get("runtime_behavior") if isinstance(metadata, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def continuation_is_linked(run: Any) -> bool:
    return str(runtime_behavior(run).get("continuation_semantics") or "") == "linked_run"


def supports_course_correction(run: Any) -> bool:
    return bool(runtime_behavior(run).get("supports_course_correction"))


def product_owns_budget_boundary(run: Any) -> bool:
    return str(runtime_behavior(run).get("budget_boundary_owner") or "") == "product"


def product_owns_grounding(run: Any) -> bool:
    return str(runtime_behavior(run).get("grounding_owner") or "") == "product"
