"""Single authoritative normalization for neutral runtime task results."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.contracts import (
    RuntimeArtifact,
    RuntimeCourseCorrectionOutcome,
    RuntimeTaskResult,
    RuntimeTaskResultStatus,
)
from runtime_protocol.validation import validate_runtime_result_envelope


class RuntimeTaskResultValidationError(ValueError):
    """A runtime task result violates the neutral result contract."""

_CANONICAL = ("text", "summary", "answer")
_ALIASES = ("output", "content", "result", "message")
_NON_ANSWER = frozenset({"reasoning", "thinking", "tool_use", "tool_call"})
_CONTROL = frozenset({"status", "warnings", "gaps", "uncovered_gaps", "error", "usage", "framework_details", "artifacts", "structured", "structured_output", "correction_outcomes", *_CANONICAL})


def _text(value: Any, depth: int = 0) -> str | None:
    if depth > 3:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value and value not in {"{}", "[]", "null"} else None
    if isinstance(value, Mapping):
        if str(value.get("type") or "").lower() in _NON_ANSWER:
            return None
        for key in (*_CANONICAL, *_ALIASES):
            if key in value:
                found = _text(value[key], depth + 1)
                if found:
                    return found
    if isinstance(value, (list, tuple)):
        values = [_text(item, depth + 1) for item in value]
        values = [item for item in values if item]
        return "\n\n".join(values).strip() or None
    return None


def _structured(value: Any, depth: int = 0) -> bool:
    if depth > 4 or value in (None, "", [], {}):
        return False
    if isinstance(value, Mapping):
        if str(value.get("type") or "").lower() in _NON_ANSWER:
            return False
        return any(_structured(item, depth + 1) for key, item in value.items() if key != "type")
    if isinstance(value, (list, tuple)):
        return any(_structured(item, depth + 1) for item in value)
    return True


def _objects(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(item) for item in value if isinstance(item, Mapping)) if isinstance(value, (list, tuple)) else ()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(item).strip() for item in value if str(item).strip()) if isinstance(value, (list, tuple)) else ()


def normalize_runtime_task_result(value: Any, *, artifacts: tuple[RuntimeArtifact, ...] = (), usage: Mapping[str, Any] | None = None, framework_details: Mapping[str, Any] | None = None, structured_output_requested: bool = False, structured_validation_error: Exception | None = None) -> RuntimeTaskResult:
    data = dict(value) if isinstance(value, Mapping) else {}
    if isinstance(value, Mapping):
        try:
            validate_runtime_result_envelope({"status": value.get("status"), "task_result": value})
        except Exception as exc:
            raise RuntimeTaskResultValidationError(str(exc)) from exc
        status = RuntimeTaskResultStatus(str(data["status"]))
    else:
        # Raw strings are internal model projections, not wire envelopes.
        status = RuntimeTaskResultStatus.COMPLETED
    selected = next((data[key] for key in _CANONICAL if data.get(key) is not None), value if isinstance(value, str) else None)
    text = _text(selected)
    if text is None:
        for key in _ALIASES:
            text = _text(data.get(key))
            if text:
                break
    structured = data.get("structured_output")
    if structured is None and isinstance(data.get("structured"), Mapping):
        structured = data["structured"]
    warnings = list(_objects(data.get("warnings")))
    gaps = _strings(data.get("gaps", data.get("uncovered_gaps")))
    if structured is None and text is None:
        extensions = {key: item for key, item in data.items() if key not in _CONTROL and _structured(item)}
        if extensions:
            structured = extensions
    if structured_validation_error is not None:
        warnings.append({"code": "structured_output_invalid", "message": "Structured output could not be normalized; usable text was preserved."})
        structured = None
    elif structured_output_requested and structured is None:
        warnings.append({"code": "structured_output_missing", "message": "The requested structured output was not returned; usable text was preserved."})
    error = dict(data["error"]) if isinstance(data.get("error"), Mapping) else None
    if not (text or structured or artifacts):
        status = RuntimeTaskResultStatus.FAILED
        error = error or {"code": "task_result_empty", "retryable": True}
    elif warnings or gaps:
        status = RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS
    return RuntimeTaskResult(
        status=status, text=text,
        structured_output=dict(structured) if isinstance(structured, Mapping) else None,
        artifacts=artifacts, warnings=tuple(warnings), gaps=gaps,
        usage=dict(usage or data.get("usage") or {}), error=error,
        framework_details=dict(framework_details or data.get("framework_details") or {}),
        correction_outcomes=tuple(
            RuntimeCourseCorrectionOutcome(
                correction_id=str(item.get("correction_id") or ""), operation_id=str(item.get("operation_id") or ""),
                state=str(item.get("state") or "unresolved"),
                runtime_plan_revision=int(item["runtime_plan_revision"]) if item.get("runtime_plan_revision") else None,
                linked_run_id=str(item["linked_run_id"]) if item.get("linked_run_id") else None,
                todo_ids=tuple(str(item) for item in item.get("todo_ids") or []),
                artifact_ids=tuple(str(item) for item in item.get("artifact_ids") or []),
                explanation=str(item["explanation"]) if item.get("explanation") is not None else None,
                unresolved_reason=str(item["unresolved_reason"]) if item.get("unresolved_reason") is not None else None,
            ) for item in data.get("correction_outcomes") or [] if isinstance(item, Mapping)
        ),
    )


def runtime_task_result_summary(result: RuntimeTaskResult) -> dict[str, Any]:
    return {
        "outcome": result.status.value,
        "output_shape": "structured" if result.structured_output is not None else "text" if result.text else "artifacts" if result.artifacts else "empty",
        "warning_count": len(result.warnings), "gap_count": len(result.gaps),
        "artifact_ids": [item.artifact_id for item in result.artifacts if item.artifact_id],
        "usage": dict(result.usage),
    }
