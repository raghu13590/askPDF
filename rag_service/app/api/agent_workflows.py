from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any, Dict, Literal, Mapping, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.tool_registry import tool_contracts_by_id
from app.agent_workflows.repository import (
    AgentWorkflowRepository,
    AgentRunInterruptError,
    BUILDER_TEST_RUN_KIND,
)
from app.agent_workflows.service import AgentRunService
from app.agent_workflows.execution_stream import AgentExecutionEventSink, retain_background_task
from app.agent_workflows.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.agent_workflows.workflow_runtime import (
    default_agent_workflow_key,
    workflow_is_chat_eligible,
    workflow_supports_replans,
)
from app.agent_workflows.chat_cancellation import (
    CHAT_CANCEL_AWAITING_HUMAN,
    CHAT_CANCEL_UNSUPPORTED,
    ChatRunCancelResult,
)
from app.agent_workflows.trace_details import detail_manifest
from app.agent_workflows.trace_payloads import is_current_debug_payload
from app.agent_workflows.canonical_trace import build_parallel_groups_safely
from runtime_protocol.contracts import AgentRuntimeEvent, AgentRuntimeRequest
logger = logging.getLogger(__name__)


async def latest_builder_test(*args: Any, **kwargs: Any):
    return await AgentWorkflowRepository().latest_builder_test(*args, **kwargs)


async def request_builder_test_cancel(*args: Any, **kwargs: Any):
    return await AgentWorkflowRepository().request_builder_test_cancel(*args, **kwargs)


def spec_fingerprint(*args: Any, **kwargs: Any):
    spec = args[0] if args else kwargs["spec"]
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _normalized_visit_index(value: Any) -> Optional[int]:
    """Return a safe positive visit index from retained runtime data."""

    if value is None:
        return 1
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized >= 1 else None
from app.runtime.catalog import catalog_payload, definition_from_run, definition_from_workflow
from app.runtime.builder_registry import BuilderSelectionError, builder_for_definition
from app.runtime.builder import BuilderTestContext, UnsupportedRequestOverrideError
from runtime_protocol.contracts import AgentDefinition, RuntimeOperationId, RuntimeValidationResult
from app.runtime.capability_resolver import (
    capability_envelope,
    capability_discovery_error,
    deployment_id,
    resolve_deployment_capability_resolution,
    resolve_definition_capability_resolution,
    resolve_run_capability_resolution,
)
from runtime_protocol.errors import RuntimeError
from app.runtime.registry import RuntimeSelectionError, get_runtime_registry
from app.runtime.operational_limits import required_positive_float
from app.runtime.operational_limits import validate_bounded_json
from app.db import AgentRunStatus, get_thread, get_thread_settings
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET
from app.models.requests import ThreadChatRequest
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
)
from app.services.agent_task_repository import get_task
from app.time_utils import iso_utc_z, maybe_iso_utc_z


router = APIRouter(tags=["agent-workflows"])


async def request_chat_run_cancel(run_id: str, *, thread_id: str):
    """Compatibility seam for API callers; cancellation routes through the adapter registry."""

    result = await AgentRunService().cancel_agent_run(run_id, thread_id=thread_id)
    if isinstance(result, Mapping):
        return ChatRunCancelResult(
            status=str(result["status"]),
            run_id=result.get("run_id"),
            run_status=result.get("run_status"),
        )
    return result


async def _require_ready_thread(thread_id: str):
    try:
        return await require_thread_embedding_ready(thread_id)
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc


class WorkflowValidationRequest(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)
    framework: str = Field(..., min_length=1)
    builder_id: str = Field(..., min_length=1)


class ThreadAgentConfigValidationRequest(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)


class InternalAgentWorkflowSaveRequest(BaseModel):
    workflow_id: Optional[str] = Field(default=None, min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    spec_json: Dict[str, Any] = Field(default_factory=dict)
    framework: str = Field(..., min_length=1)
    builder_id: str = Field(..., min_length=1)


class AgentRunResumeRequest(BaseModel):
    action: str = Field(..., min_length=1)
    interrupt_id: str = Field(..., min_length=1)
    edited_payload: Optional[Dict[str, Any]] = None
    client_metadata: Optional[Dict[str, Any]] = None
    selected_option_ids: Optional[list[str]] = None
    resume_token: Optional[str] = None
    resume_version: Optional[int] = None
    thread_id: str = Field(..., min_length=1)
    approval_scope: Optional[Literal["once", "session", "always"]] = None
    approval_feedback: Optional[str] = None
    approval_modifications: Optional[Dict[str, Any]] = None


class AgentRunCancelRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)


class AgentRunInputOperationRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    input: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def bounded_input(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return validate_bounded_json(value, field_name="input")

class BuilderTransientMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20000)


class BuilderTestRunRequest(ThreadChatRequest):
    builder_session_id: str = Field(..., min_length=1, max_length=200)
    base_workflow_id: str = Field(..., min_length=1)
    spec: Dict[str, Any] = Field(default_factory=dict)
    allow_external_tools: bool = False
    transient_messages: list[BuilderTransientMessage] = Field(default_factory=list, max_length=100)
    workflow_spec_fingerprint: Optional[str] = Field(default=None, max_length=128)


class BuilderTestRunResumeRequest(AgentRunResumeRequest):
    llm_model: str = Field(..., min_length=1)
    use_web_search: bool = False
    use_reranker: Optional[bool] = True
    context_window: int = DEFAULT_TOKEN_BUDGET
    replans: Optional[int] = None
    system_role_override: Optional[str] = None
    tool_instructions_override: Optional[Dict[str, str]] = None
    custom_instructions_override: Optional[str] = None
    hitl_web_approval: bool = False
    client_timezone: Optional[str] = None
    client_locale: Optional[str] = None
    client_now_iso: Optional[str] = None


def _workflow_payload(workflow) -> Dict[str, Any]:
    spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
    known_builtin_keys = set(builtin_workflow_keys())
    row_key = str(workflow.id or "").strip()
    builtin_key = row_key if workflow.is_builtin and row_key in known_builtin_keys else None
    return {
        "id": workflow.id,
        "workflow_id": workflow.id,
        "builtin_key": builtin_key,
        "name": workflow.name,
        "description": workflow.description,
        "visibility": workflow.visibility,
        "is_builtin": workflow.is_builtin,
        "is_default": builtin_key == default_agent_workflow_key(),
        "supports_replans": workflow_supports_replans(spec),
        "supports_long_running_tasks": bool(
            ((spec.get("runtime") or {}).get("features") or {}).get("supports_long_running_tasks")
        ),
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
        **catalog_payload(workflow),
    }


def _definition_for_workflow(workflow) -> AgentDefinition:
    return definition_from_workflow(workflow)


def _provider_for_workflow(workflow):
    return builder_for_definition(_definition_for_workflow(workflow))


def _validation_payload(validation: RuntimeValidationResult) -> Dict[str, Any]:
    payload = dict(validation.diagnostics)
    payload.update({
        "valid": validation.valid,
        "issues": [issue.to_dict() for issue in validation.issues],
        "errors": [issue.code for issue in validation.issues],
        "normalized_spec": validation.normalized_spec,
        "runtime_metadata": dict(validation.runtime_metadata),
    })
    return payload


def _workflow_spec_payload(workflow) -> Dict[str, Any]:
    try:
        validation = {
            "valid": bool(workflow.validation_result_json.get("valid", True)),
            **(workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {}),
        }
    except Exception as exc:
        validation = {
            "valid": False,
            "errors": [f"validation failed: {exc}"],
            "warnings": [],
            "schema_version": getattr(workflow, "schema_version", None),
            "workflow_id": None,
        }
    return {
        "id": str((workflow.metadata_json or {}).get("version_id") or f"{workflow.id}:v{workflow.version}"),
        "workflow_id": workflow.id,
        "framework": getattr(workflow, "framework", None),
        "builder_id": getattr(workflow, "builder_id", None),
        "category": getattr(workflow, "category", None),
        "version": workflow.version,
        "schema_version": workflow.schema_version,
        "spec_json": workflow.spec_json if isinstance(workflow.spec_json, dict) else {},
        "validation": validation,
        "validation_result_json": workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {},
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
    }


def _is_valid_workflow_for_service(workflow) -> bool:
    if not workflow or workflow.schema_version != 1 or not isinstance(workflow.spec_json, dict):
        return False
    validation = workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {}
    return bool(validation.get("valid", True))


def _debug_payload_for_response(run) -> Dict[str, Any] | None:
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else None
    if not debug:
        return None
    if not is_current_debug_payload(debug):
        logger.error(
            "Invalid retained debug trace contract | correlation_id=trace:%s version=%r",
            run.id,
            debug.get("version"),
        )
        return None
    trace = debug.get("trace") if isinstance(debug.get("trace"), dict) else None
    summary = debug.get("summary") if isinstance(debug.get("summary"), dict) else None
    if trace is None or summary is None:
        logger.error("Malformed retained debug trace | run_id=%s", run.id)
        return None
    compact_debug = {key: value for key, value in debug.items() if key != "graph"}
    visualizations = compact_debug.get("visualizations") if isinstance(compact_debug.get("visualizations"), dict) else {}
    topology_kind = next(
        (
            str(key)
            for key, value in visualizations.items()
            if isinstance(value, Mapping) and ("nodes" in value or "edges" in value)
        ),
        None,
    )
    topology_available = topology_kind is not None
    return {
        **compact_debug,
        "trace": trace,
        "summary": dict(summary),
        "detail_manifest": detail_manifest(debug.get("details")),
        "topology": {
            "available": topology_available,
            "kind": topology_kind,
            "operation_refs": topology_available,
        },
    }


def _debug_trace_failure_for_response(run) -> Dict[str, Any] | None:
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else None
    if not debug:
        return None
    if not is_current_debug_payload(debug):
        return {"code": "debug_trace_contract_invalid", "retryable": False, "run_id": str(run.id)}
    if not isinstance(debug.get("trace"), dict) or not isinstance(debug.get("summary"), dict):
        return {"code": "debug_trace_shape_invalid", "retryable": False, "run_id": str(run.id)}
    return None


def _turn_summary_payload(turn) -> Dict[str, Any]:
    trace_refs = turn.agent_trace_refs_json if isinstance(turn.agent_trace_refs_json, dict) else {}
    return {
        "id": turn.id,
        "kind": turn.agent_run_turn_kind,
        "sequence": turn.agent_run_sequence,
        "trace_refs": trace_refs,
    }


def _pending_interrupt_payload(run) -> Dict[str, Any] | None:
    pending = run.pending_interrupt_json if isinstance(run.pending_interrupt_json, dict) else None
    return dict(pending) if pending else None


def _run_payload(run, turns=None) -> Dict[str, Any]:
    turns = turns or []
    payload = {
        "id": run.id,
        "thread_id": run.thread_id,
        "user_id": run.user_id,
        "workflow_id": run.workflow_id,
        "framework": getattr(run, "framework", None),
        "builder_id": getattr(run, "builder_id", None),
        "definition_category": getattr(run, "definition_category", None),
        "task_id": run.task_id,
        "parent_run_id": run.parent_run_id,
        "task_attempt": run.task_attempt,
        "turns": [_turn_summary_payload(turn) for turn in turns],
        "resolved_spec_json": run.resolved_spec_json,
        "status": run.status,
        "runtime_binding_status": getattr(run, "runtime_binding_status", "active"),
        "pending_interrupt": _pending_interrupt_payload(run),
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "error_json": run.error_json,
        "metrics_json": run.metrics_json,
        "parallel_summary": (run.metrics_json or {}).get("parallel_summary") if isinstance(run.metrics_json, dict) else None,
        "corrective": (run.metrics_json or {}).get("corrective") if isinstance(run.metrics_json, dict) else None,
        "retrieval_quality_report": (run.metrics_json or {}).get("retrieval_quality_report") if isinstance(run.metrics_json, dict) else None,
        "grounding_report": (run.metrics_json or {}).get("grounding_report") if isinstance(run.metrics_json, dict) else None,
        "debug": _debug_payload_for_response(run),
        "debug_trace_failure": _debug_trace_failure_for_response(run),
        "run_kind": (run.run_metadata_json or {}).get("run_kind"),
        "builder_session_id": (run.run_metadata_json or {}).get("builder_session_id"),
        "final_output": (run.debug_trace_json or {}).get("final_output") if isinstance(run.debug_trace_json, dict) else None,
        "observability": {
            "event_projection_version": 2,
            "topology_available": bool(
                isinstance(run.debug_trace_json, dict)
                and isinstance(run.debug_trace_json.get("visualizations"), dict)
                and any(
                    isinstance(value, Mapping) and ("nodes" in value or "edges" in value)
                    for value in run.debug_trace_json["visualizations"].values()
                )
            ),
        },
    }
    return payload


def _sse(event: Dict[str, Any], sequence: int) -> str:
    name = str(event.get("event") or "message")
    payload = {"id": sequence, "event": name, "data": event.get("data") or {}}
    return f"id: {sequence}\nevent: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


class _BuilderProviderEventSink:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()

    async def emit(self, event: Mapping[str, Any]) -> None:
        await self.queue.put(dict(event))


async def _stream_builder_provider_call(call: Any):
    sink = _BuilderProviderEventSink()

    async def execute() -> None:
        try:
            await call(sink)
        finally:
            await sink.queue.put(None)

    task = asyncio.create_task(execute())
    sequence = 0
    try:
        while True:
            event = await sink.queue.get()
            if event is None:
                break
            sequence += 1
            yield _sse(event, sequence)
        await task
    finally:
        if not task.done():
            task.cancel()


def _run_summary_payload(run) -> Dict[str, Any]:
    metrics = run.metrics_json if isinstance(run.metrics_json, dict) else {}
    error = run.error_json if isinstance(run.error_json, dict) else None
    return {
        "id": run.id,
        "thread_id": run.thread_id,
        "workflow_id": run.workflow_id,
        "task_id": run.task_id,
        "parent_run_id": run.parent_run_id,
        "task_attempt": run.task_attempt,
        "status": run.status,
        "pending_interrupt": _pending_interrupt_payload(run),
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "parallel_summary": metrics.get("parallel_summary") if isinstance(metrics.get("parallel_summary"), dict) else None,
        "corrective": metrics.get("corrective") if isinstance(metrics.get("corrective"), dict) else None,
        "retrieval_quality_report": metrics.get("retrieval_quality_report") if isinstance(metrics.get("retrieval_quality_report"), dict) else None,
        "grounding_report": metrics.get("grounding_report") if isinstance(metrics.get("grounding_report"), dict) else None,
        "metrics": {
            "duration_ms": metrics.get("duration_ms"),
            "route": metrics.get("route"),
            "node_event_count": metrics.get("node_event_count", 0),
            "tool_event_count": metrics.get("tool_event_count", 0),
            "tool_warning_count": metrics.get("tool_warning_count", 0),
            "tool_error_count": metrics.get("tool_error_count", 0),
            "error_count": metrics.get("error_count", 0),
            "replan_count": metrics.get("replan_count", 0),
            "evaluation_confidence": metrics.get("evaluation_confidence"),
            "corrective": metrics.get("corrective") if isinstance(metrics.get("corrective"), dict) else None,
        },
        "error": {
            "code": error.get("code"),
            "raw_message": error.get("raw_message"),
            "retryable": error.get("retryable"),
        } if error else None,
    }


def _capabilities_for_workflow(spec_json: Dict[str, Any]) -> Dict[str, Any]:
    runtime = spec_json.get("runtime") if isinstance(spec_json.get("runtime"), dict) else {}
    features = runtime.get("features") if isinstance(runtime.get("features"), dict) else {}
    config = spec_json.get("config") if isinstance(spec_json.get("config"), dict) else {}
    return {
        "required_tool_ids": sorted({str(value) for value in config.get("allowed_tool_ids") or [] if value}),
        "node_tool_requirements": {},
        "supports_parallel_dispatch": bool(features.get("supports_parallel_dispatch")),
        "supports_corrective_retrieval": bool(features.get("supports_corrective_retrieval")),
        "parallel_policy": config.get("parallel_policy") if isinstance(config.get("parallel_policy"), dict) else None,
        "corrective_policy": config.get("corrective_policy") if isinstance(config.get("corrective_policy"), dict) else None,
    }


def _agent_workflow_tool_contract_catalog(*, excluded_node_types: set[str] | None = None) -> Dict[str, Any]:
    excluded_node_types = excluded_node_types or set()
    contracts: Dict[str, Any] = {}
    for contract_id, records in sorted(tool_contracts_by_id().items()):
        canonical_tools = sorted(
            str(record.get("tool_name"))
            for record in records
            if isinstance(record.get("tool_name"), str) and record.get("tool_name")
        )
        first = records[0] if records else {}
        contracts[contract_id] = {
            "id": contract_id,
            "category": first.get("category"),
            "display_name": first.get("display_name"),
            "description": first.get("description"),
            "canonical_tools": canonical_tools,
            "allowed_node_types": sorted(
                {
                    str(node_type)
                    for record in records
                    for node_type in record.get("allowed_node_types", [])
                    if node_type and str(node_type) not in excluded_node_types
                }
            ),
            "required_node_capabilities": sorted(
                {
                    str(capability)
                    for record in records
                    for capability in record.get("required_node_capabilities", [])
                    if capability
                }
            ),
            "artifact_keys": sorted(
                {
                    str(artifact_key)
                    for record in records
                    for artifact_key in record.get("artifact_keys", [])
                    if artifact_key
                }
            ),
            "warning_codes": sorted(
                {
                    str(warning_code)
                    for record in records
                    for warning_code in record.get("warning_codes", [])
                    if warning_code
                }
            ),
        }
    return contracts


@router.get("/agent-workflows")
async def list_agent_workflows():
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    workflows = await repo.list_workflows(include_custom=True)
    valid_workflows = []
    for workflow in workflows:
        try:
            spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
            if not workflow_is_chat_eligible(spec):
                continue
            if _is_valid_workflow_for_service(workflow):
                valid_workflows.append(workflow)
        except Exception:
            continue
    return {"agent_workflows": [_workflow_payload(workflow) for workflow in valid_workflows]}


@router.get("/agent-runtimes")
async def list_agent_runtimes(response: Response):
    response.headers["Cache-Control"] = "no-store"
    registry = get_runtime_registry()
    adapters = registry.adapters()

    async def resolve(adapter: Any) -> dict[str, Any]:
        runtime_id = deployment_id(adapter)
        try:
            resolution = await resolve_deployment_capability_resolution(adapter)
            error = resolution.error
            capabilities = resolution.capabilities
        except Exception as exc:
            logger.exception("Runtime deployment discovery failed | runtime_id=%s", runtime_id)
            error = capability_discovery_error(exc, adapter)
            capabilities = None
        return capability_envelope(
            capabilities=capabilities,
            resource="deployment",
            runtime_id=runtime_id,
            framework=adapter.framework,
            builder_id=adapter.builder_id,
            error=error,
        )

    deployments = await asyncio.gather(*(resolve(adapter) for adapter in adapters))
    return {"agent_runtimes": deployments}


@router.get("/agent-runtimes/{runtime_id}/capabilities")
async def get_agent_runtime_capabilities(runtime_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    registry = get_runtime_registry()
    adapter = registry.get_deployment(runtime_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail="Agent runtime deployment not found")
    resolution = await resolve_deployment_capability_resolution(adapter)
    return capability_envelope(
        capabilities=resolution.capabilities,
        resource="deployment",
        runtime_id=runtime_id,
        framework=adapter.framework,
        builder_id=adapter.builder_id,
        error=resolution.error,
    )


@router.get("/agent-workflows/{workflow_id}/capabilities")
async def get_agent_workflow_capabilities(workflow_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    include_custom = workflow_id not in builtin_workflow_keys()
    workflow = await repo.get_workflow(workflow_id, include_custom=include_custom)
    if (
        not workflow
        or not workflow_is_chat_eligible(workflow.spec_json)
        or not _is_valid_workflow_for_service(workflow)
    ):
        raise HTTPException(status_code=404, detail="Agent workflow not found")

    definition = definition_from_workflow(workflow)
    registry = get_runtime_registry()
    try:
        adapter = registry.get(definition)
    except RuntimeSelectionError as exc:
        return capability_envelope(
            capabilities=None,
            resource="definition",
            runtime_id=f"{definition.framework}:{definition.builder_id}",
            framework=definition.framework,
            builder_id=definition.builder_id,
            definition_id=definition.definition_id,
            error=RuntimeError(
                "runtime_selection_failed",
                "No compatible runtime deployment is available",
                details={"framework": definition.framework, "builder_id": definition.builder_id},
            ).to_dict(),
        )
    resolution = await resolve_definition_capability_resolution(definition, registry=registry)
    return capability_envelope(
        capabilities=resolution.capabilities,
        resource="definition",
        runtime_id=deployment_id(adapter),
        framework=definition.framework,
        builder_id=definition.builder_id,
        definition_id=definition.definition_id,
        error=resolution.error,
    )


@router.get("/agent-runs/{run_id}/events")
async def stream_agent_run_events(
    run_id: str,
    request: Request,
    thread_id: str = Query(..., min_length=1),
    after_sequence: int = Query(default=0, ge=0),
):
    run = await _owned_run_for_operation(run_id, thread_id)
    repository = AgentWorkflowRepository()

    async def events():
        sequence = after_sequence
        poll_interval = required_positive_float("AGENT_EVENT_POLL_INTERVAL_SECONDS")
        heartbeat_interval = required_positive_float("AGENT_SSE_HEARTBEAT_INTERVAL_SECONDS")
        idle_seconds = 0.0
        canonical_events: list[AgentRuntimeEvent] = []
        public_event_ids: dict[str, str] = {}

        def canonical_event(row: Any, event_id: str) -> AgentRuntimeEvent:
            return AgentRuntimeEvent(
                event_id=event_id,
                run_id=str(getattr(row, "agent_run_id", run.id)),
                sequence=int(getattr(row, "sequence", 0) or 0),
                attempt=int(getattr(row, "attempt", 1) or 1),
                kind=str(getattr(row, "kind", "runtime.event")),
                payload=dict(getattr(row, "payload_json", None) or {}),
                occurred_at=maybe_iso_utc_z(getattr(row, "occurred_at", None)),
                terminal=bool(getattr(row, "terminal", False)),
                source_metadata=dict(getattr(row, "source_metadata_json", None) or {}),
            )

        while True:
            if await request.is_disconnected():
                return
            all_rows = await repository.list_run_events(run.id)
            event_id_counts: dict[str, int] = {}
            for row in all_rows:
                raw_event_id = str(getattr(row, "event_id", "") or "")
                event_id_counts[raw_event_id] = event_id_counts.get(raw_event_id, 0) + 1
            for row in all_rows:
                raw_event_id = str(getattr(row, "event_id", "") or "")
                row_id = str(getattr(row, "id", "") or getattr(row, "sequence", ""))
                public_event_ids.setdefault(
                    row_id,
                    raw_event_id
                    if event_id_counts[raw_event_id] == 1
                    else f"{raw_event_id}:journal:{row_id}",
                )
            if not canonical_events and sequence > 0:
                canonical_events.extend(
                    canonical_event(row, public_event_ids[str(getattr(row, "id", "") or getattr(row, "sequence", ""))])
                    for row in all_rows
                    if int(getattr(row, "sequence", 0) or 0) <= sequence
                )
            rows = [row for row in all_rows if int(getattr(row, "sequence", 0) or 0) > sequence]
            if rows:
                idle_seconds = 0.0
                for row in rows:
                    if await request.is_disconnected():
                        return
                    sequence = int(getattr(row, "sequence", sequence) or sequence)
                    row_key = str(getattr(row, "id", "") or getattr(row, "sequence", ""))
                    event_id = public_event_ids[row_key]
                    canonical_events.append(canonical_event(row, event_id))
                    payload = dict(getattr(row, "payload_json", None) or {})
                    terminal = bool(payload.get("terminal")) or str(getattr(row, "kind", "")) in {
                        "run.completed", "run.failed", "run.cancelled", "run.clarification",
                    }
                    value = {
                        "id": getattr(row, "id", None),
                        "event_id": event_id,
                        "run_id": run.id,
                        "sequence": sequence,
                        "attempt": getattr(row, "attempt", 1),
                        "kind": getattr(row, "kind", "runtime.event"),
                        "payload": payload,
                        "occurred_at": maybe_iso_utc_z(getattr(row, "occurred_at", None)),
                        "created_at": maybe_iso_utc_z(getattr(row, "created_at", None)),
                        "terminal": terminal,
                        "parallel_groups": build_parallel_groups_safely(canonical_events),
                    }
                    yield f"id: {sequence}\nevent: run_event\ndata: {json.dumps(value, separators=(',', ':'))}\n\n"
                    if terminal:
                        return
            else:
                idle_seconds += poll_interval
                if idle_seconds >= heartbeat_interval:
                    if await request.is_disconnected():
                        return
                    yield f": heartbeat {sequence}\n\n"
                    idle_seconds = 0.0
            await asyncio.sleep(poll_interval)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/agent-runs/{run_id}/capabilities")
async def get_agent_run_capabilities(
    run_id: str,
    response: Response,
    thread_id: str = Query(..., min_length=1),
):
    response.headers["Cache-Control"] = "no-store"
    if not await get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if run is None or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Agent run not found")

    definition = definition_from_run(run)
    registry = get_runtime_registry()
    task = await get_task(run.task_id, thread_id=thread_id) if getattr(run, "task_id", None) else None
    try:
        adapter = registry.get(definition)
    except RuntimeSelectionError as exc:
        adapter = None
        error = RuntimeError(
            "runtime_selection_failed",
            "No compatible runtime deployment is available",
            details={"framework": definition.framework, "builder_id": definition.builder_id},
        ).to_dict()
        resolution = None
    else:
        resolution = await resolve_run_capability_resolution(
            definition, registry=registry, run=run, task=task
        )
    return capability_envelope(
        capabilities=resolution.capabilities if resolution is not None else None,
        resource="run",
        runtime_id=deployment_id(adapter) if adapter is not None else f"{definition.framework}:{definition.builder_id}",
        framework=definition.framework,
        builder_id=definition.builder_id,
        definition_id=definition.definition_id,
        run_id=run.id,
        run_status=run.status,
        error=resolution.error if resolution is not None else error,
    )


@router.get("/agent-runs/{run_id}/state")
async def get_agent_run_state(
    run_id: str,
    thread_id: str = Query(..., min_length=1),
):
    run = await _owned_run_for_operation(run_id, thread_id)
    try:
        state = await AgentRunService().inspect_agent_run(run)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    return {"run_id": run.id, "state": state}


async def _owned_run_for_operation(run_id: str, thread_id: str):
    if not await get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    run = await AgentWorkflowRepository().get_run(run_id)
    if run is None or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


async def _execute_run_operation(
    run_id: str,
    operation: RuntimeOperationId,
    *,
    thread_id: str,
    input: Optional[Dict[str, Any]] = None,
    idempotency_key: str,
) -> Dict[str, Any]:
    if operation in {
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
        RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
        RuntimeOperationId.RUN_STEER_LIVE,
    } and not str((input or {}).get("text") or "").strip():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_runtime_input",
                "safe_message": "Input text must be a non-empty string",
                "retryable": False,
            },
        )
    run = await _owned_run_for_operation(run_id, thread_id)
    try:
        result = await AgentRunService().operate_agent_run(
            run,
            operation,
            input=input,
            idempotency_key=idempotency_key,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    return {"run_id": run.id, "operation": operation.value, "result": result}


@router.post("/agent-runs/{run_id}/followups")
async def send_agent_run_followup(
    run_id: str,
    req: AgentRunInputOperationRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    return await _execute_run_operation(
        run_id,
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
        thread_id=req.thread_id,
        input=req.input,
        idempotency_key=idempotency_key,
    )


@router.post("/agent-runs/{run_id}/interrupt-with-input")
async def interrupt_agent_run_with_input(
    run_id: str,
    req: AgentRunInputOperationRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    return await _execute_run_operation(
        run_id,
        RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
        thread_id=req.thread_id,
        input=req.input,
        idempotency_key=idempotency_key,
    )


@router.post("/agent-runs/{run_id}/steer-live")
async def steer_agent_run_live(
    run_id: str,
    req: AgentRunInputOperationRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    return await _execute_run_operation(
        run_id,
        RuntimeOperationId.RUN_STEER_LIVE,
        thread_id=req.thread_id,
        input=req.input,
        idempotency_key=idempotency_key,
    )


@router.get("/agent-workflows/builtins/{builtin_key}/source")
async def get_builtin_agent_workflow_source(builtin_key: str):
    """Return the immutable-on-disk definition used to seed a built-in workflow."""
    workflow = next(
        (item for item in load_builtin_workflows() if item.get("builtin_key") == builtin_key),
        None,
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Built-in agent workflow source not found")
    definition = AgentDefinition(
        definition_id=builtin_key,
        framework=str(workflow.get("framework") or "").strip(),
        builder_id=str(workflow.get("builder_id") or "").strip(),
    )
    try:
        return dict(await builder_for_definition(definition).source(builtin_key))
    except (BuilderSelectionError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="Built-in agent workflow source not found") from exc


@router.post("/agent-workflows/validate")
async def validate_agent_workflow(req: WorkflowValidationRequest):
    definition = AgentDefinition(
        definition_id=str(req.spec.get("workflow_id") or "validation"),
        framework=req.framework,
        builder_id=req.builder_id,
    )
    try:
        provider = builder_for_definition(definition)
        validation = await provider.validate(definition, req.spec)
    except BuilderSelectionError as exc:
        raise HTTPException(status_code=400, detail={"code": "builder_unavailable", "message": str(exc)}) from exc
    report = _validation_payload(validation)
    report.setdefault("framework", req.framework)
    report.setdefault("builder_id", req.builder_id)
    return report


@router.post("/internal/agent-workflows/test-runs/stream")
async def stream_internal_agent_workflow_test(req: BuilderTestRunRequest):
    embedding_context = await _require_ready_thread(req.thread_id)
    thread = embedding_context.thread
    workflow = await AgentWorkflowRepository().get_workflow(req.base_workflow_id, include_custom=True)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Base agent workflow not found")
    if req.use_web_search and not req.allow_external_tools:
        raise HTTPException(
            status_code=409,
            detail={"code": "external_tool_confirmation_required", "message": "Confirm external tool calls before testing with web search."},
        )
    try:
        candidate = dict(req.spec)
        provider = _provider_for_workflow(workflow)
        definition = _definition_for_workflow(workflow)
        builder_capabilities = await provider.capabilities(definition)
        if not builder_capabilities.transient_tests:
            raise HTTPException(
                status_code=409,
                detail={"code": "runtime_capability_unsupported", "message": "Builder tests are not enabled for this definition"},
            )
        resolved = dict(await provider.resolve(
            definition,
            candidate,
            thread_settings={"hitl_web_approval": req.hitl_web_approval},
        ))
    except (BuilderSelectionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_test_workflow", "message": str(exc)}) from exc

    repo = AgentWorkflowRepository()
    run = await repo.create_run(
        thread_id=req.thread_id,
        workflow_id=workflow.id,
        resolved_spec_json=resolved,
        run_metadata_json={
            "run_kind": BUILDER_TEST_RUN_KIND,
            "builder_session_id": req.builder_session_id,
            "base_workflow_id": req.base_workflow_id,
            "spec_fingerprint": spec_fingerprint(resolved),
            "client_spec_fingerprint": req.workflow_spec_fingerprint,
        },
    )

    async def events():
        runtime_request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            input={"question": req.question},
        )
        context = BuilderTestContext(
            run=run,
            test_request=req,
            embedding_model=embedding_context.embedding_model,
            builder_session_id=req.builder_session_id,
        )

        async def call(sink: Any) -> None:
            await provider.transient_test(runtime_request, context=context, event_sink=sink)

        async for event in _stream_builder_provider_call(call):
            yield event

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/internal/agent-workflows/test-runs/latest")
async def get_latest_internal_agent_workflow_test(
    builder_session_id: str = Query(..., min_length=1),
    base_workflow_id: Optional[str] = Query(None),
):
    run = await latest_builder_test(builder_session_id, base_workflow_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    turns = await AgentWorkflowRepository().list_chat_turns_for_run(run.id)
    payload = _run_payload(run, turns)
    try:
        payload["runtime_inspection"] = await AgentRunService().inspect_agent_run(run)
    except RuntimeError as exc:
        if exc.code not in {"runtime_capability_unsupported", "runtime_capability_unavailable"}:
            raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    return {"agent_run": payload}


@router.post("/internal/agent-workflows/test-runs/{run_id}/cancel")
async def cancel_internal_agent_workflow_test(run_id: str):
    run = await request_builder_test_cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    return {"status": "cancel_requested", "run_id": run.id}


@router.post("/internal/agent-workflows/test-runs/{run_id}/resume/stream")
async def resume_internal_agent_workflow_test(run_id: str, req: BuilderTestRunResumeRequest):
    embedding_context = await _require_ready_thread(req.thread_id)
    thread = embedding_context.thread
    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if run is None or run.thread_id != req.thread_id or (run.run_metadata_json or {}).get("run_kind") != BUILDER_TEST_RUN_KIND:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    definition = definition_from_run(run)
    try:
        provider = builder_for_definition(definition)
        builder_capabilities = await provider.capabilities(definition)
    except BuilderSelectionError as exc:
        raise HTTPException(status_code=400, detail={"code": "builder_unavailable", "message": str(exc)}) from exc
    if not builder_capabilities.transient_tests:
        raise HTTPException(
            status_code=409,
            detail={"code": "runtime_capability_unsupported", "message": "Builder tests are not enabled for this definition"},
        )
    try:
        resolution = await repo.resolve_pending_interrupt(
            run_id,
            interrupt_id=req.interrupt_id,
            action=req.action,
            edited_payload=req.edited_payload,
            client_metadata=req.client_metadata,
            selected_option_ids=req.selected_option_ids,
            resume_token=req.resume_token,
            resume_version=req.resume_version,
            expected_thread_id=req.thread_id,
        )
    except AgentRunInterruptError as exc:
        raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc
    if resolution is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    decision = (resolution.interrupt or {}).get("decision") if isinstance(resolution.interrupt, dict) else None
    if not isinstance(decision, dict):
        raise HTTPException(status_code=409, detail="Builder test interrupt cannot be resumed")

    async def events():
        runtime_request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            input={"decision": decision},
        )
        context = BuilderTestContext(
            run=resolution.run,
            test_request=req,
            embedding_model=embedding_context.embedding_model,
            builder_session_id=str((run.run_metadata_json or {}).get("builder_session_id") or ""),
            resume_decision=decision,
        )

        async def call(sink: Any) -> None:
            await provider.resume_transient_test(runtime_request, context=context, event_sink=sink)

        async for event in _stream_builder_provider_call(call):
            yield event

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/agent-workflows/{workflow_id}")
async def get_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    include_custom = workflow_id not in builtin_workflow_keys()
    workflow = await repo.get_workflow(workflow_id, include_custom=include_custom)
    if (
        not workflow
        or not workflow_is_chat_eligible(workflow.spec_json)
        or not _is_valid_workflow_for_service(workflow)
    ):
        raise HTTPException(status_code=404, detail="Agent workflow not found")
    spec_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": spec_payload,
        "current_version": spec_payload,
        "capabilities": _capabilities_for_workflow(workflow.spec_json if isinstance(workflow.spec_json, dict) else {}),
    }


@router.post("/internal/agent-workflows")
async def save_internal_agent_workflow(req: InternalAgentWorkflowSaveRequest):
    repo = AgentWorkflowRepository()
    try:
        workflow_id = (req.workflow_id or "").strip() or None
        if workflow_id is None:
            workflow_id = f"custom_workflow_{uuid.uuid4().hex[:12]}"
        spec_json = dict(req.spec_json)
        spec_json["workflow_id"] = workflow_id
        workflow, version = await repo.save_internal_workflow_version(
            workflow_id=workflow_id,
            name=req.name,
            description=req.description,
            spec_json=spec_json,
            framework=req.framework,
            builder_id=req.builder_id,
        )
    except (BuilderSelectionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    version_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": version_payload,
        "version": version_payload,
    }


@router.delete("/internal/agent-workflows/{workflow_id}")
async def delete_internal_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    try:
        workflow = await repo.mark_custom_workflow_deleted(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if workflow is None:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    return {
        "status": "deleted",
        "agent_workflow": _workflow_payload(workflow),
    }


@router.get("/internal/agent-workflows/catalog")
async def get_internal_agent_workflow_catalog(
    framework: str = Query(..., min_length=1),
    builder_id: str = Query(..., min_length=1),
):
    definition = AgentDefinition(
        definition_id="catalog",
        framework=framework,
        builder_id=builder_id,
    )
    try:
        catalog = await builder_for_definition(definition).catalog(definition)
    except BuilderSelectionError as exc:
        raise HTTPException(status_code=503, detail={"code": "builder_unavailable", "message": str(exc)}) from exc
    return dict(catalog.payload)


@router.get("/internal/agent-workflows/{workflow_id}")
async def get_internal_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    workflow = await repo.get_workflow(workflow_id, include_custom=True)
    if not workflow or workflow.is_builtin:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    spec_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": spec_payload,
        "current_version": spec_payload,
    }


@router.post("/threads/{thread_id}/agent-config/validate")
async def validate_thread_agent_config(thread_id: str, req: ThreadAgentConfigValidationRequest):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    thread_settings = await get_thread_settings(thread_id)
    agent_settings = thread_settings.get("agent_workflow") if isinstance(thread_settings, dict) else None
    agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
    workflow_id = agent_settings.get("workflow_id") or default_agent_workflow_key()

    workflow = await repo.get_workflow(
        workflow_id,
        include_custom=workflow_id not in builtin_workflow_keys(),
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Agent workflow not found")

    if not workflow_is_chat_eligible(workflow.spec_json or {}):
        return {
            "valid": False,
            "workflow_id": workflow.id,
            "workflow_version": workflow.version,
            "validation": {
                "valid": False,
                "errors": ["long_running_workflow_requires_agent_task"],
                "issues": [{
                    "code": "long_running_workflow_requires_agent_task",
                    "severity": "error",
                    "message": "This workflow is available only through the Deep Research task workspace.",
                }],
            },
            "resolved_spec_json": {},
        }

    provider = _provider_for_workflow(workflow)
    definition = _definition_for_workflow(workflow)
    try:
        request_overrides = provider.filter_request_overrides(
            definition,
            req.overrides,
            reject_unsupported=True,
        )
        resolved_spec = await provider.resolve(
            definition,
            workflow.spec_json,
            thread_settings=thread_settings,
            request_overrides=request_overrides,
        )
    except UnsupportedRequestOverrideError as exc:
        issues = [
            {
                "code": "unsupported_request_override",
                "severity": "error",
                "message": f"The selected builder does not support request override: {key}",
                "path": f"overrides.{key}",
            }
            for key in exc.keys
        ]
        return {
            "valid": False,
            "workflow_id": workflow.id,
            "workflow_version": workflow.version,
            "validation": {
                "valid": False,
                "errors": [issue["code"] for issue in issues],
                "issues": issues,
            },
            "resolved_spec_json": dict(workflow.spec_json or {}),
        }
    except ValueError as exc:
        candidate = dict(workflow.spec_json or {})
        candidate_config = dict(candidate.get("config") or {})
        for source in (thread_settings or {}, req.overrides or {}):
            if isinstance(source, dict):
                candidate_config.update({key: value for key, value in source.items() if value is not None})
        candidate["config"] = candidate_config
        validation = await provider.validate(definition, candidate)
        report = _validation_payload(validation)
        report["errors"] = report.get("errors") or [str(exc)]
        return {
            "valid": False,
            "workflow_id": workflow.id,
            "workflow_version": workflow.version,
            "validation": report,
            "resolved_spec_json": candidate,
        }

    validation = await provider.validate(definition, resolved_spec)
    return {
        "valid": validation.valid,
        "workflow_id": workflow.id,
        "workflow_version": workflow.version,
        "validation": _validation_payload(validation),
        "resolved_spec_json": resolved_spec,
    }


@router.get("/threads/{thread_id}/agent-runs")
async def list_thread_agent_runs(
    thread_id: str,
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentWorkflowRepository()
    runs = await repo.list_runs_for_thread(thread_id, limit=limit, status=status)
    return {
        "thread_id": thread_id,
        "limit": limit,
        "status": status,
        "agent_runs": [_run_summary_payload(run) for run in runs],
    }


@router.get("/agent-runs/{run_id}")
async def get_agent_run(
    run_id: str,
    thread_id: str = Query(..., min_length=1),
):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Agent run not found")

    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Agent run not found")
    turns = await repo.list_chat_turns_for_run(run.id)
    return {"agent_run": _run_payload(run, turns)}


@router.get("/agent-runs/{run_id}/operations/{operation_id}/details")
async def get_agent_run_operation_details(
    run_id: str,
    operation_id: str,
    visit_index: int = Query(..., ge=1),
    thread_id: str = Query(..., min_length=1),
):
    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if run is None or run.thread_id != thread_id or not await get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else {}
    for detail in debug.get("details") if isinstance(debug.get("details"), list) else []:
        if not isinstance(detail, dict):
            continue
        detail_visit_index = _normalized_visit_index(detail.get("visit_index"))
        if detail_visit_index is None:
            continue
        if str(detail.get("operation_id") or "") == operation_id and detail_visit_index == visit_index:
            return {"run_id": run.id, "detail": detail}
    for event in reversed(await repo.list_run_events(run_id)):
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        event_visit_index = _normalized_visit_index(payload.get("visit_index"))
        if event_visit_index is None:
            continue
        if str(payload.get("operation_id") or "") == operation_id and event_visit_index == visit_index:
            return {
                "run_id": run.id,
                "detail": {
                    "operation_id": operation_id,
                    "operation_type": payload.get("operation_type"),
                    "visit_index": visit_index,
                    "status": str(event.kind).split(".")[-1],
                    "event": payload,
                    "safety": {"bounded": True},
                },
            }
    raise HTTPException(status_code=404, detail="Operation visit details are unavailable")


@router.post("/agent-runs/{run_id}/cancel")
async def cancel_chat_agent_run(
    run_id: str,
    req: AgentRunCancelRequest,
):
    if not await get_thread(req.thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    try:
        result = await request_chat_run_cancel(run_id, thread_id=req.thread_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    if result.status == "missing":
        raise HTTPException(status_code=404, detail="Agent run not found")
    if result.status == CHAT_CANCEL_UNSUPPORTED:
        raise HTTPException(status_code=409, detail="This run uses its own cancellation endpoint")
    if result.status == CHAT_CANCEL_AWAITING_HUMAN:
        raise HTTPException(status_code=409, detail="Use the human-review actions for this paused run")
    return {
        "status": result.status,
        "run_id": result.run_id,
        "run_status": result.run_status,
    }


@router.post("/agent-runs/{run_id}/resume")
async def resume_agent_run(
    run_id: str,
    req: AgentRunResumeRequest,
    accept: Optional[str] = Header(default=None),
):
    if await get_thread(req.thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    service = AgentRunService()

    async def execute_resume(*, event_sink: Any = None):
        return await service.resume_agent_run(
            run_id,
            interrupt_id=req.interrupt_id,
            action=req.action,
            edited_payload=req.edited_payload,
            client_metadata=req.client_metadata,
            selected_option_ids=req.selected_option_ids,
            resume_token=req.resume_token,
            resume_version=req.resume_version,
            expected_thread_id=req.thread_id,
            execution_event_sink=event_sink,
            approval_scope=req.approval_scope,
            approval_feedback=req.approval_feedback,
            approval_modifications=req.approval_modifications,
        )

    if "text/event-stream" in str(accept or "").lower():
        sink = AgentExecutionEventSink(include_details=False)

        async def run_resume() -> None:
            try:
                result = await execute_resume(event_sink=sink)
                if result is None:
                    await sink.queue.put({"event": "__missing__", "data": {}})
                    return
                compact_run = {
                    "id": result.run.id,
                    "thread_id": result.run.thread_id,
                    "workflow_id": result.run.workflow_id,
                    "status": result.run.status,
                    "pending_interrupt": _pending_interrupt_payload(result.run),
                }
                await sink.queue.put({
                    "event": "__result__",
                    "data": {
                        "agent_run": compact_run,
                        "interrupt": result.interrupt,
                        "outcome": result.outcome,
                        "duplicate": result.duplicate,
                    },
                })
            except AgentRunInterruptError as exc:
                await sink.queue.put({"event": "__error__", "data": {"error": {"code": exc.code, "raw_message": str(exc), "retryable": False}}})
            except RuntimeError as exc:
                await sink.queue.put({"event": "__error__", "data": {"error": exc.to_dict()}})
            except Exception as exc:
                await sink.queue.put({"event": "__error__", "data": {"error": {"code": "agent_run_resume_failed", "raw_message": str(exc), "retryable": True}}})

        async def events():
            sequence = 0
            task = asyncio.create_task(run_resume())
            retain_background_task(task)
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(
                            sink.queue.get(),
                            timeout=required_positive_float("AGENT_SSE_HEARTBEAT_INTERVAL_SECONDS"),
                        )
                    except asyncio.TimeoutError:
                        sequence += 1
                        yield _sse({"event": "heartbeat", "data": {"run_id": run_id}}, sequence)
                        continue
                    event = str(item.get("event") or "message")
                    data = item.get("data") or {}
                    if event == "__missing__":
                        sequence += 1
                        yield _sse({"event": "stream.error", "data": {"run_id": run_id, "error": {"code": "agent_run_not_found", "raw_message": "Agent run not found", "retryable": False}}}, sequence)
                        break
                    if event == "__error__":
                        sequence += 1
                        yield _sse({"event": "stream.error", "data": {"run_id": run_id, **data}}, sequence)
                        break
                    if event == "__result__":
                        break
                    sequence += 1
                    yield _sse(item, sequence)
                    if event in {"run.completed", "run.failed", "run.cancelled"}:
                        break
            finally:
                sink.detach_delivery()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        result = await execute_resume()
    except AgentRunInterruptError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=exc.to_dict()) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    repo = AgentWorkflowRepository()
    turns = await repo.list_chat_turns_for_run(result.run.id)
    return {
        "agent_run": _run_payload(result.run, turns),
        "interrupt": result.interrupt,
        "outcome": result.outcome,
        "duplicate": result.duplicate,
    }
