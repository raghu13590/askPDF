import asyncio
from types import SimpleNamespace

import pytest

from app.runtime.adapter import AgentRuntimeAdapter, RuntimeInvocationContext
from runtime_protocol.errors import RuntimeError
from app.runtime.catalog import (
    continuation_from_run,
    definition_from_run,
    definition_from_workflow,
    event_from_source,
    request_from_run,
    result_to_product_payload,
)
from runtime_protocol.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeApprovalResponse,
    RuntimeCourseCorrection,
    RuntimeCourseCorrectionOutcome,
    RuntimeCourseCorrectionReceipt,
    RuntimeSteeringInput,
    RuntimeCapabilities,
    RuntimeCapabilitySemantics,
    RuntimeFeatureDescriptor,
    RuntimeFeatureId,
    RuntimeSupportLevel,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimePlanChange,
    RuntimeSupportLevel,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    RuntimeArtifact,
    RuntimeTaskContext,
    RuntimeTaskResult,
    RuntimeTaskResultStatus,
    RuntimeUsageSnapshot,
    TaskOrchestrationDelta,
    ensure_protocol_compatible,
    validated_disabled_operation_ids,
)
from runtime_protocol.transport import result_from_dict
from runtime_protocol.validation import RuntimeProtocolValidationError
from app.runtime.observability import normalize_runtime_event
from app.agent_workflows.interrupts import AgentRunInterruptError, normalize_pending_interrupt_payload
from app.runtime.product_capabilities import project_public_capabilities
from app.runtime.task_results import normalize_runtime_task_result, runtime_task_result_summary
from app.runtime.behavior import snapshot_runtime_behavior
from app.services.agent_task_runtime_projection import runtime_delta_conflict_details
from langgraph_runtime.workflows.deep_research_execution import (
    RuntimeBudgetMeter,
    RuntimeExecutionServices,
    runtime_execution_services_factory,
)


def _runtime_budget_snapshot(limits):
    dimensions = {
        "model_calls": int(limits.get("max_model_calls", 10)),
        "model_tokens": int(limits.get("max_model_tokens", 1000)),
        "tool_calls": int(limits.get("max_tool_calls", 10)),
        "elapsed_active_ms": int(limits.get("max_active_runtime_ms", 1000)),
    }
    zero = {key: 0 for key in dimensions}
    return {"tranche_index": 1, "tranche_limits": dimensions, "tranche_usage": zero, "lifetime_usage": dict(zero)}


def test_runtime_behavior_snapshot_requires_all_neutral_ownership_fields():
    behavior = {
        "continuation_semantics": "same_run_safe_boundary",
        "supports_course_correction": True,
        "supports_orchestration_delta": True,
        "usage_accounting_owner": "runtime",
        "budget_boundary_owner": "product",
        "grounding_owner": "product",
    }

    assert snapshot_runtime_behavior(behavior) == behavior
    with pytest.raises(ValueError, match="grounding_owner"):
        snapshot_runtime_behavior({key: value for key, value in behavior.items() if key != "grounding_owner"})
from langgraph_runtime.workflows.deep_research_nodes import deep_task_scheduler
from langgraph_runtime.adapter import _result_from_graph
from langgraph_runtime.workflows import deep_research_nodes
from langgraph_runtime.runtime_support.task_results import (
    RuntimeTaskResultValidationError,
    normalize_runtime_task_result as normalize_runtime_service_task_result,
)
from langgraph_runtime.router_runtime import _install_budget_meter


def test_course_correction_contract_is_json_only_and_versioned():
    correction = RuntimeCourseCorrection(
        correction_id="correction-1",
        operation_id="operation-1",
        instruction="Revise only the remaining work.",
        scope="remaining_work",
        observed_task_version=4,
        observed_plan_revision=2,
    )
    receipt = RuntimeCourseCorrectionReceipt(
        correction_id=correction.correction_id,
        operation_id=correction.operation_id,
        status="accepted",
        run_id="run-1",
        run_status="running",
    )

    assert correction.to_dict()["protocol_version"]
    assert receipt.to_dict()["status"] == "accepted"
    with pytest.raises(ValueError, match="scope"):
        RuntimeCourseCorrection(
            correction_id="correction-2",
            operation_id="operation-2",
            instruction="Rewrite completed work.",
            scope="all_work",
            observed_task_version=4,
        )


def test_correction_outcomes_round_trip_independently_for_multiple_redirects():
    outcomes = (
        RuntimeCourseCorrectionOutcome(
            correction_id="correction-1", operation_id="operation-1",
            state="satisfied", runtime_plan_revision=2, todo_ids=("targeted-1",),
        ),
        RuntimeCourseCorrectionOutcome(
            correction_id="correction-2", operation_id="operation-2",
            state="unresolved", runtime_plan_revision=3,
            unresolved_reason="Comparison evidence was unavailable.",
        ),
    )
    task_result = RuntimeTaskResult(
        status=RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS,
        text="Partial answer", warnings=({"code": "course_correction_unresolved"},),
        correction_outcomes=outcomes,
    )
    restored = result_from_dict(AgentRuntimeResult(status="completed", task_result=task_result).to_dict())

    assert restored.task_result is not None
    assert restored.task_result.correction_outcomes == outcomes


def test_invalid_plans_are_not_fabricated():
    assert not hasattr(deep_research_nodes, "_fallback_research_plan")


def test_neutral_contracts_are_frozen_and_json_compatible():
    binding = ContinuationBinding(
        binding_type="langgraph_checkpoint",
        payload={"binding_id": "opaque-binding-1"},
    )
    request = AgentRuntimeRequest(
        run_id="run-1",
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        continuation=binding,
    )
    result = AgentRuntimeResult(status="completed", output={"answer": "ok"}, continuation=binding)
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="run.completed",
        terminal=True,
    )
    capabilities = RuntimeCapabilities(operations={
        RuntimeOperationId.RUN_EVENTS: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.NATIVE,
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
        ),
        RuntimeOperationId.RUN_RESUME: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.CONDITIONAL,
            owner=RuntimeOperationOwner.RUNTIME,
            enabled=True,
        semantics=RuntimeCapabilitySemantics.RESUME_FROM_INTERRUPT,
        ),
    })

    assert request.to_dict()["continuation"]["payload"]["binding_id"] == "opaque-binding-1"
    assert result.to_dict()["status"] == "completed"
    assert event.to_dict()["terminal"] is True
    assert capabilities.to_dict()["operations"]["run.resume"]["support"] == "conditional"
    assert capabilities.to_dict()["operations"]["run.resume"]["owner"] == "runtime"
    assert list(capabilities.to_dict()["operations"]) == ["run.events", "run.resume"]


def test_task_orchestration_delta_round_trips_with_idempotency_and_versions():
    delta = TaskOrchestrationDelta(
        event_id="evt-1",
        attempt_id="run-1:attempt:1",
        operation_id="operation-1",
        idempotency_key="delta:run-1:1",
        observed_task_version=4,
        observed_plan_revision=2,
        plan_changes=(RuntimePlanChange(
            runtime_revision=3,
            parent_runtime_revision=2,
            acknowledged_product_revision=2,
            reason="course_correction",
            planner_visit=3,
            plan={"objective": "Redirect remaining work", "todos": []},
            correction_ids=("correction-1",),
        ),),
        todo_changes=({"id": "todo-1", "status": "completed"},),
        artifacts=({"artifact_id": "runtime:a1", "sha256": "abc"},),
    )
    restored = result_from_dict(AgentRuntimeResult(status="completed", orchestration_delta=delta).to_dict())

    assert restored.orchestration_delta == delta


def test_runtime_delta_emits_only_unacknowledged_ordered_plan_history():
    history = [
        {
            "runtime_revision": revision,
            "parent_runtime_revision": revision - 1,
            "reason": "initial" if revision == 1 else "course_correction",
            "planner_visit": revision,
            "plan": {"objective": f"revision {revision}", "todos": []},
            "correction_ids": [] if revision == 1 else ["correction-1"],
        }
        for revision in (1, 2)
    ]
    result = _result_from_graph(
        {
            "status": "completed", "agent_run_id": "run-1", "agent_task_id": "task-1",
            "task_version": 3, "task_plan_changes": history,
        },
        observed_plan_revision=1, acknowledged_runtime_plan_revision=1,
        operation_id="resume-1", attempt_id="run-1:attempt:1",
        boundary_event_id="run-1:attempt:1:operation:resume-1:result",
    )

    assert result.orchestration_delta is not None
    assert [value.runtime_revision for value in result.orchestration_delta.plan_changes] == [2]
    assert result.orchestration_delta.plan_changes[0].acknowledged_product_revision == 1


def test_runtime_delta_allows_version_advances_owned_by_the_active_run():
    delta = TaskOrchestrationDelta(
        event_id="run-1:attempt:1:operation:operation-1:result",
        attempt_id="run-1:attempt:1",
        operation_id="operation-1",
        idempotency_key="delta:run-1",
        observed_task_version=4,
        observed_plan_revision=2,
    )

    assert runtime_delta_conflict_details(
        task=SimpleNamespace(version=35, active_run_id="run-1"),
        agent_run_id="run-1",
        delta=delta,
        current_plan_revision=2,
    ) is None
    assert runtime_delta_conflict_details(
        task=SimpleNamespace(version=35, active_run_id="replacement-run"),
        agent_run_id="run-1",
        delta=delta,
        current_plan_revision=2,
    )["reason"] == "active_run_changed"


@pytest.mark.asyncio
async def test_runtime_execution_retries_empty_subagent_result_once():
    state = {
        "task_todos": [{
            "id": "T1",
            "status": "running",
            "attempt": 1,
            "max_attempts": 2,
            "artifact_ids": [],
        }],
    }
    services = RuntimeExecutionServices(
        todos=None,
        artifacts=None,
        budgets=None,
        cancellation=SimpleNamespace(requested=lambda: False),
        events=None,
        memory=None,
        state=state,
    )

    todos = await services.record_result_packets([{
        "todo_id": "T1",
        "status": "failed",
        "summary": "",
        "retryable": True,
        "error": {"code": "task_result_empty"},
    }])
    services.state = {"task_todos": todos}
    scheduled = await services.schedule_ready("task-1", limit=1)

    assert todos[0]["status"] == "ready"
    assert scheduled[0].status == "running"
    assert scheduled[0].attempt == 2


def test_incompatible_runtime_protocol_is_rejected_before_execution():
    with pytest.raises(ValueError, match="protocol"):
        ensure_protocol_compatible("2.0", "2.0")


@pytest.mark.asyncio
async def test_runtime_budget_meter_accumulates_parallel_usage_without_lost_updates():
    limits = {"max_model_calls": 50, "max_model_tokens": 1000, "max_tool_calls": 50, "max_active_runtime_ms": 1000}
    meter = RuntimeBudgetMeter(
        _runtime_budget_snapshot(limits), limits,
    )

    await asyncio.gather(*(
        meter.consume(model_calls=1, model_tokens=10, artifact_bytes=7)
        for _ in range(8)
    ))

    snapshot = await meter.snapshot()
    assert snapshot["tranche_usage"]["model_calls"] == 8
    assert snapshot["lifetime_usage"]["model_tokens"] == 80
    assert snapshot["lifetime_usage"]["artifact_bytes"] == 56


@pytest.mark.asyncio
async def test_scheduler_stops_at_runtime_budget_boundary_and_returns_checkpoint_snapshot():
    limits = {
        "max_model_calls": 1,
        "max_model_tokens": 1000,
        "max_tool_calls": 10,
        "max_active_runtime_ms": 1000,
        "max_concurrency": 1,
        "max_fanout": 1,
    }
    meter = RuntimeBudgetMeter(_runtime_budget_snapshot(limits), limits)
    await meter.consume(model_calls=1)
    state = {
        "agent_task_id": "task-1",
        "agent_run_id": "run-1",
        "task_limits": limits,
        "task_budget_usage": _runtime_budget_snapshot({}),
        "task_todos": [{
            "id": "todo-1", "title": "Remaining research", "status": "pending",
            "priority": 1, "required": True, "profile_id": "document_researcher",
            "dependency_ids": [], "attempt": 0, "max_attempts": 2,
            "progress": 0, "artifact_ids": [], "version": 1,
        }],
    }
    result = await deep_task_scheduler(state, {"configurable": {
        "deep_research_services_factory": runtime_execution_services_factory,
        "runtime_budget_meter": meter,
        "cancellation_checker": lambda: False,
    }})

    assert result["task_work_items"] == []
    assert result["task_budget_boundary"]["dimensions"] == ["model_calls"]
    assert result["task_budget_usage"]["tranche_usage"]["model_calls"] == 1


@pytest.mark.asyncio
async def test_runtime_artifact_bytes_are_present_in_boundary_budget_snapshot():
    state = {
        "agent_task_id": "task-1",
        "task_limits": {"max_model_calls": 10, "max_model_tokens": 1000, "max_tool_calls": 10, "max_active_runtime_ms": 1000},
        "task_budget_usage": _runtime_budget_snapshot({}),
        "runtime_artifacts": [],
    }
    configurable = {"cancellation_checker": lambda: False}
    services = runtime_execution_services_factory(state, configurable)

    artifact = await services.persist_artifact(
        task_id="task-1",
        kind="intermediate_report",
        content="research so far",
    )
    snapshot = await services.budget_snapshot()

    assert snapshot["lifetime_usage"]["artifact_bytes"] == artifact["byte_size"]


@pytest.mark.asyncio
async def test_budget_review_interrupt_exposes_research_so_far_and_retry_choices(monkeypatch):
    captured = {}

    async def fake_call_model(*_args, **_kwargs):
        return '{"pass": true, "issues": []}', {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "accept_partial"}

    monkeypatch.setattr(deep_research_nodes, "_call_model", fake_call_model)
    monkeypatch.setattr(deep_research_nodes, "interrupt", fake_interrupt)
    state = {
        "agent_task_id": "task-1",
        "agent_run_id": "run-1",
        "final_answer": "This is the evidence-backed research completed so far.",
        "task_budget_boundary": {
            "status": "requested", "dimensions": ["model_calls"], "tranche_index": 1,
        },
        "task_incomplete_reasons": ["todo-remaining"],
        "task_evidence_manifest": [],
        "warnings": [],
        "task_limits": {"max_model_calls": 10, "max_model_tokens": 1000, "max_tool_calls": 10, "max_active_runtime_ms": 1000},
        "task_budget_usage": _runtime_budget_snapshot({}),
    }

    result = await deep_research_nodes.evidence_critic(state, {"configurable": {
        "deep_research_services_factory": runtime_execution_services_factory,
        "cancellation_checker": lambda: False,
    }})

    assert captured["type"] == "budget_review"
    assert captured["provisional_answer"] == state["final_answer"]
    assert captured["allowed_actions"] == ["continue", "accept_partial", "steer"]
    assert result["task_budget_review_route"] == "accept_partial"


@pytest.mark.asyncio
async def test_resumed_runtime_uses_product_authorized_reset_tranche():
    exhausted = {
        "tranche_index": 1,
        "tranche_limits": {"model_calls": 1, "model_tokens": 1000, "tool_calls": 10, "elapsed_active_ms": 1000},
        "tranche_usage": {"model_calls": 1},
        "lifetime_usage": {"model_calls": 1},
        "boundary": {"status": "requested", "dimensions": ["model_calls"], "tranche_index": 1},
    }
    reset = {
        "tranche_index": 2,
        "tranche_limits": {"model_calls": 1, "model_tokens": 1000, "tool_calls": 10, "elapsed_active_ms": 1000},
        "tranche_usage": {"model_calls": 0},
        "lifetime_usage": {"model_calls": 1},
        "boundary": None,
    }
    config = {"configurable": {}}
    _install_budget_meter(
        config,
        {
            "agent_task_id": "task-1",
            "task_budget_usage": exhausted,
            "task_limits": {"max_model_calls": 1},
        },
        authoritative_budget=reset,
    )

    snapshot = await config["configurable"]["runtime_budget_meter"].snapshot()
    assert snapshot["tranche_index"] == 2
    assert snapshot["tranche_usage"]["model_calls"] == 0
    assert snapshot["lifetime_usage"]["model_calls"] == 1
    assert snapshot.get("boundary") is None


def test_runtime_operation_descriptor_rejects_invalid_enabled_states():
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, True)
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, False)


def test_runtime_operation_descriptor_requires_a_typed_owner():
    with pytest.raises(TypeError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, enabled=True)
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, "runtime", True)


@pytest.mark.parametrize("response_operation", [None, "", "interrupt.respond"])
def test_pending_interrupt_requires_an_implemented_response_operation(response_operation):
    payload = {"interrupt_id": "interrupt-1"}
    if response_operation is not None:
        payload["response_operation"] = response_operation

    with pytest.raises(AgentRunInterruptError) as caught:
        normalize_pending_interrupt_payload(payload)

    assert caught.value.code == "interrupt_response_operation_invalid"


def test_public_capability_projection_preserves_approval_response():
    capabilities = RuntimeCapabilities(operations={
        RuntimeOperationId.RUN_APPROVAL_RESPOND: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.NATIVE,
            owner=RuntimeOperationOwner.RUNTIME,
            enabled=True,
        ),
    })

    projected = project_public_capabilities(capabilities)

    assert projected.operations[RuntimeOperationId.RUN_APPROVAL_RESPOND].enabled is True


def test_continuation_requires_authoritative_runtime_binding():
    run = SimpleNamespace(id="run-1", runtime_binding_json={})

    assert continuation_from_run(run) is None


@pytest.mark.asyncio
async def test_optional_adapter_methods_have_structured_unsupported_defaults():
    class MinimalAdapter(AgentRuntimeAdapter):
        framework = "minimal"
        builder_id = "minimal_builder"

        async def capabilities(self, definition):
            return RuntimeCapabilities()

        async def validate(self, definition, spec, *, options=None):
            return RuntimeValidationResult(valid=True)

        async def start(self, request, *, context, event_sink=None):
            return AgentRuntimeResult(status="completed")

    adapter = MinimalAdapter()
    request = AgentRuntimeRequest("run-1", "thread-1", "definition-1", "minimal", "minimal_builder")
    operations = (
        ("run.get", lambda: adapter.get_run(request)),
        ("run.list", lambda: adapter.list_runs(thread_id="thread-1")),
        ("run.wait", lambda: adapter.wait(request)),
        ("run.events", lambda: adapter.stream_events(request)),
        ("run.resume", lambda: adapter.resume(request, interrupt={}, context=RuntimeInvocationContext())),
        ("runtime_continuation_unavailable", lambda: adapter.continue_run(request, context=RuntimeInvocationContext())),
        ("run.cancel", lambda: adapter.cancel(request)),
        ("run.approval.respond", lambda: adapter.respond_to_approval(request, RuntimeApprovalResponse("approve", scope="once"))),
        ("run.send_followup", lambda: adapter.send_followup(request, {})),
        ("run.interrupt_with_input", lambda: adapter.interrupt_with_input(request, {})),
        ("run.steer_live", lambda: adapter.steer_live(request, RuntimeSteeringInput("focus"))),
        ("run.inspect_state", lambda: adapter.inspect_state(request)),
        ("run.replay", lambda: adapter.replay(request)),
        ("run.fork", lambda: adapter.fork(request)),
        ("subagent.list", lambda: adapter.list_subagents(request)),
        ("subagent.send", lambda: adapter.send_to_subagent(request, "subagent-1", {})),
        ("subagent.cancel", lambda: adapter.cancel_subagent(request, "subagent-1")),
        ("artifact.list", lambda: adapter.list_artifacts(request)),
        ("run.continuation.cleanup", lambda: adapter.delete_continuation(ContinuationBinding("test", {}))),
        ("trace.project", lambda: adapter.project_trace([], run_id="run-1")),
    )
    for operation_id, invoke in operations:
        with pytest.raises(RuntimeError) as caught:
            await invoke()
        if operation_id == "runtime_continuation_unavailable":
            assert caught.value.code == operation_id
            continue
        assert caught.value.code == "runtime_capability_unsupported"
        assert caught.value.retryable is False
        assert caught.value.details == {
            "operation_id": operation_id,
            "framework": "minimal",
            "builder_id": "minimal_builder",
            "support_level": "unsupported",
            "explanation": caught.value.details["explanation"],
        }


def test_runtime_event_can_carry_an_opaque_continuation_binding():
    binding = ContinuationBinding(
        binding_type="hermes_session",
        payload={"session_id": "session-1", "upstream_run_id": "hermes-run-7"},
    )
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="runtime.session_started",
        continuation=binding,
    )
    assert event.to_dict()["continuation"]["payload"]["upstream_run_id"] == "hermes-run-7"


def test_manual_state_update_is_not_a_runtime_operation():
    assert "run.update_state" not in {operation.value for operation in RuntimeOperationId}


def test_catalog_identity_is_concrete_and_category_is_metadata_only():
    workflow = SimpleNamespace(
        id="router_rag_agent",
        name="Router Agent",
        version=1,
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
        metadata_json={"builtin_key": "router_rag_agent"},
        spec_json={"runtime": {"features": {"supports_replans": False}}},
    )

    definition = definition_from_workflow(workflow)

    assert definition.definition_id == "router_rag_agent"
    assert definition.framework == "langgraph"
    assert definition.builder_id == "langgraph_graph"
    assert definition.category == "router"
    assert definition.capabilities == {"supports_replans": False}


def test_definition_rejects_unknown_disabled_operations():
    with pytest.raises(ValueError, match="unknown operations: run.not_real"):
        validated_disabled_operation_ids(["run.not_real"])


def test_run_identity_and_typed_projection_round_trip():
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        task_id=None,
        parent_run_id=None,
        runtime_binding_json={
            "binding_type": "langgraph_checkpoint",
            "payload": {"binding_id": "opaque-binding-1"},
        },
        run_metadata_json={},
    )

    binding = continuation_from_run(run)
    request = request_from_run(run, input={"question": "hello"})
    result = AgentRuntimeResult(status="clarification", clarification={"options": ["one", "two"]})
    event = event_from_source(
        {"event": "run.completed", "data": {"event_id": "runtime-event-1"}},
        run_id="run-1",
        sequence=2,
    )

    assert binding is not None
    assert binding.payload["binding_id"] == "opaque-binding-1"
    assert request.continuation == binding
    assert request.input == {"question": "hello"}
    assert result.status == "clarification"
    assert result_to_product_payload(result)["clarification_options"] == ["one", "two"]
    assert event.event_id == "runtime-event-1"
    assert event.terminal is True


def test_workflow_and_run_definition_metadata_are_identical():
    spec = {
        "runtime": {"features": {"supports_replans": False}},
        "config": {
            "allowed_tool_ids": ["search_documents"],
            "task_policy": {"profiles": ["research"]},
        },
    }
    workflow = SimpleNamespace(
        id="router_rag_agent",
        name="Router Agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
        metadata_json={},
        spec_json=spec,
    )
    run = SimpleNamespace(
        id="run-1",
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category="router",
        resolved_spec_json=spec,
    )

    workflow_definition = definition_from_workflow(workflow)
    run_definition = definition_from_run(run)
    assert run_definition.definition_metadata == workflow_definition.definition_metadata
    assert run_definition.capabilities == workflow_definition.capabilities


def test_validation_contract_is_json_compatible():
    result = RuntimeValidationResult(
        valid=False,
        issues=(RuntimeValidationIssue(code="invalid_workflow", message="bad spec", path="config.graph"),),
        runtime_metadata={"framework": "langgraph"},
    )

    assert result.to_dict()["issues"][0]["path"] == "config.graph"
    assert result.to_dict()["valid"] is False


def test_runtime_task_context_and_artifact_are_json_compatible():
    artifact = RuntimeArtifact(kind="intermediate_report", content="report", todo_id="todo-1")
    context = RuntimeTaskContext(
        task_id="task-1",
        objective="research",
        todos=({"id": "todo-1", "status": "pending"},),
        artifact_manifests=(artifact.to_dict(),),
        artifact_contents={artifact.artifact_id or "runtime": "report"},
    )

    assert artifact.to_dict()["kind"] == "intermediate_report"
    assert context.to_dict()["todos"][0]["id"] == "todo-1"


def test_runtime_usage_snapshot_distinguishes_measured_from_unknown_counters():
    usage = RuntimeUsageSnapshot.from_mapping(
        {"input_tokens": 90, "output_tokens": 10, "tool_calls": 3},
        operation_id="operation-1",
    )

    assert usage.model_tokens == 100
    assert usage.model_calls is None
    assert set(usage.measured_dimensions) == {"model_tokens", "tool_calls"}


def test_langgraph_result_quality_is_identical_in_result_and_delta():
    result = _result_from_graph({
        "status": "completed",
        "agent_task_id": "task-1",
        "agent_run_id": "run-1",
        "task_version": 1,
        "final_answer": "A useful answer with disclosed limitations.",
        "task_result_warnings": [{
            "code": "evidence_critic_issues",
            "details": {"issues": ["One source could not be corroborated."]},
        }],
        "task_result_gaps": ["Primary-source confirmation is missing."],
    }, operation_id="operation-1")

    assert result.task_result is not None
    assert result.task_result.status is RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS
    assert result.orchestration_delta is not None
    delta_result = result.orchestration_delta.result or {}
    assert delta_result["warnings"] == list(result.task_result.warnings)
    assert delta_result["incomplete_reasons"] == list(result.task_result.gaps)
    assert delta_result["task_result"] == result.task_result.to_dict()


def test_runtime_task_result_preserves_text_when_optional_structure_is_invalid():
    result = normalize_runtime_task_result(
        "A useful provisional answer",
        structured_output_requested=True,
        structured_validation_error=ValueError("invalid schema"),
        framework_details={"framework": "langgraph"},
    )

    assert result.status is RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS
    assert result.text == "A useful provisional answer"
    assert result.warnings[0]["code"] == "structured_output_invalid"
    assert result.framework_details == {"framework": "langgraph"}
    assert runtime_task_result_summary(result)["output_shape"] == "text"


@pytest.mark.parametrize("value", [
    {"text": "answer"},
    {"status": "not-a-status", "text": "answer"},
    {"status": 1, "text": "answer"},
])
def test_control_plane_runtime_result_rejects_missing_or_unknown_status(value):
    with pytest.raises(RuntimeProtocolValidationError):
        normalize_runtime_task_result(value)


@pytest.mark.parametrize("value", [
    {"status": "completed", "text": "answer"},
    {"status": "completed_with_warnings", "text": "answer", "warnings": [{"code": "W1"}]},
    {"status": "failed", "text": "partial"},
    {"status": "timed_out", "text": "partial"},
    {"status": "cancelled", "text": "partial"},
])
def test_result_parser_accepts_each_declared_task_result_status(value):
    result = normalize_runtime_task_result(value)
    restored = result_from_dict(AgentRuntimeResult(status="completed", task_result=result).to_dict())
    assert restored.task_result is not None
    assert restored.task_result.status.value == value["status"]


def test_result_parser_requires_status_on_wire_envelopes():
    with pytest.raises(RuntimeProtocolValidationError):
        result_from_dict(AgentRuntimeResult(status="completed").to_dict() | {"status": None})

    with pytest.raises(RuntimeProtocolValidationError):
        result_from_dict({"status": "completed", "task_result": {"text": "answer"}})


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out"])
def test_runtime_service_result_preserves_failure_status_with_partial_text(status):
    result = normalize_runtime_service_task_result({"status": status, "text": "partial output"})

    assert result.status.value == status
    assert result.text == "partial output"


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out"])
@pytest.mark.parametrize("quality", [
    {"warnings": [{"code": "partial"}]},
    {"gaps": ["source unavailable"]},
])
def test_runtime_service_result_does_not_promote_failure_with_quality_metadata(status, quality):
    result = normalize_runtime_service_task_result({
        "status": status,
        "text": "partial output",
        **quality,
    })

    assert result.status.value == status
    assert result.text == "partial output"


@pytest.mark.parametrize("value", [
    {"status": "unknown", "text": "partial output"},
    {"status": "unknown", "structured_output": {"answer": "partial output"}},
])
def test_runtime_service_result_rejects_unknown_status(value):
    with pytest.raises(RuntimeTaskResultValidationError, match="unknown runtime task result status"):
        normalize_runtime_service_task_result(value)


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out"])
def test_deep_research_subagent_conversion_preserves_declared_failure(status):
    neutral = normalize_runtime_service_task_result({"status": status, "text": "partial output"})

    result = deep_research_nodes._subagent_result_from_neutral(
        neutral, summary="partial output", structured_value={}
    )

    assert result.status == status
    assert result.summary == "partial output"


def test_runtime_task_result_rejects_empty_success_and_normalizes_empty_output_failure():
    with pytest.raises(ValueError, match="usable output"):
        RuntimeTaskResult(status=RuntimeTaskResultStatus.COMPLETED)

    result = normalize_runtime_task_result("{}", structured_output_requested=True)
    assert result.status is RuntimeTaskResultStatus.FAILED
    assert result.error == {"code": "task_result_empty", "retryable": True}


def test_runtime_task_result_preserves_nested_text_from_noncanonical_envelope():
    result = normalize_runtime_task_result({
        "status": "completed",
        "output": {
            "content": [
                {"type": "reasoning", "text": "internal analysis"},
                {"type": "text", "text": "The grounded answer."},
            ],
        },
    })

    assert result.status is RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS
    assert result.text == "The grounded answer."
    assert result.warnings == ({
        "code": "task_result_envelope_noncanonical",
        "message": "Usable output was preserved from a noncanonical result field.",
    },)


def test_runtime_task_result_does_not_treat_reasoning_only_blocks_as_an_answer():
    result = normalize_runtime_task_result({
        "status": "completed",
        "content": [{"type": "reasoning", "text": "internal analysis"}],
    })

    assert result.status is RuntimeTaskResultStatus.FAILED
    assert result.text is None
    assert result.error == {"code": "task_result_empty", "retryable": True}
    assert result.warnings == ()


def test_runtime_task_result_preserves_unwrapped_structured_extensions():
    result = normalize_runtime_task_result({
        "status": "completed",
        "definition": "The generals must agree despite traitorous participants.",
        "conditions": {"IC1": "Loyal lieutenants obey the same order."},
        "warnings": [{"code": "W001", "description": "Solvability is covered later."}],
    })

    assert result.status is RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS
    assert result.text is None
    assert result.structured_output == {
        "definition": "The generals must agree despite traitorous participants.",
        "conditions": {"IC1": "Loyal lieutenants obey the same order."},
    }
    assert [warning["code"] for warning in result.warnings] == [
        "W001", "task_result_envelope_noncanonical",
    ]


def test_langgraph_node_events_normalize_to_topology_linked_operations():
    kind, payload = normalize_runtime_event(
        "node.completed",
        {"node_id": "planner", "node_type": "planner", "visit_index": 2, "elapsed_ms": 17},
    )

    assert kind == "operation.completed"
    assert payload["operation_id"] == "planner"
    assert payload["operation_type"] == "planner"
    assert payload["visit_index"] == 2
    assert payload["topology_ref"] == {"kind": "graph_node", "id": "planner"}


def test_runtime_operations_remain_topology_optional():
    kind, payload = normalize_runtime_event(
        "operation.started",
        {"operation_id": "hermes_session", "operation_type": "agent_session"},
    )

    assert kind == "operation.started"
    assert payload["operation_id"] == "hermes_session"
    assert "topology_ref" not in payload


def test_feature_identifiers_and_operation_metadata_are_closed_vocabularies():
    descriptor = RuntimeFeatureDescriptor(RuntimeSupportLevel.NATIVE, True)
    assert RuntimeCapabilities(features={RuntimeFeatureId.TOOLS: descriptor}).to_dict()["features"] == {
        "tools": {"support": "native", "enabled": True, "disabled_reason": None}
    }
    with pytest.raises(ValueError, match="RuntimeFeatureId"):
        RuntimeCapabilities(features={"tools": descriptor})
