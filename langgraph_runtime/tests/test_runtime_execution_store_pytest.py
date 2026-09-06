from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

from langgraph_runtime.execution_store import ExecutionConflictError, ExecutionStore, LeaseLostError, _json_safe


@pytest.mark.asyncio
async def test_cancel_terminalizes_an_execution_waiting_for_human() -> None:
    store = ExecutionStore(database_url="")
    record = await store.create(
        "run-awaiting-human",
        "start",
        {"run_id": "run-awaiting-human"},
        {},
    )
    record.status = "awaiting_human"

    outcome = await store.request_cancel(record.run_id)

    assert outcome.is_terminal is True
    assert outcome.run_status == "cancelled"
    assert (await store.get(record.run_id)).status == "cancelled"
    assert (await store.get(record.run_id)).cancel_requested is False


@pytest.mark.asyncio
async def test_request_cancel_returns_typed_outcomes_and_preserves_terminal_state() -> None:
    store = ExecutionStore()

    unknown = await store.request_cancel("missing")
    assert unknown.outcome == "unknown"

    await store.create("run-cancel", "start", {"run_id": "run-cancel"}, {"request": {"run_id": "run-cancel"}})
    queued = await store.request_cancel("run-cancel")
    repeated = await store.request_cancel("run-cancel")
    assert queued.outcome == "requested"
    assert queued.run_status == "queued"
    assert repeated.outcome == "requested"
    assert (await store.get("run-cancel")).cancel_requested is True

    await store.set_status("run-cancel", "cancelled")
    terminal = await store.request_cancel("run-cancel")
    assert terminal.outcome == "terminal"
    assert terminal.run_status == "cancelled"
    assert (await store.get("run-cancel")).cancel_requested is False


@pytest.mark.asyncio
async def test_request_cancel_active_run_is_visible_to_worker() -> None:
    store = ExecutionStore()
    await store.create("run-active-cancel", "start", {"run_id": "run-active-cancel"}, {"request": {"run_id": "run-active-cancel"}})
    await store.claim("run-active-cancel")

    outcome = await store.request_cancel("run-active-cancel")

    assert outcome.outcome == "requested"
    assert (await store.is_cancel_requested("run-active-cancel")) is True


@pytest.mark.asyncio
async def test_request_pause_is_durable_and_visible_to_worker() -> None:
    store = ExecutionStore(database_url="")
    await store.create("run-active-pause", "start", {"run_id": "run-active-pause"}, {})
    await store.claim("run-active-pause")

    outcome = await store.request_pause("run-active-pause")

    assert outcome["status"] == "pause_requested"
    assert await store.is_pause_requested("run-active-pause") is True
    assert (await store.get("run-active-pause")).status == "queued"


@pytest.mark.asyncio
async def test_request_pause_preserves_terminal_and_checkpointed_states() -> None:
    store = ExecutionStore(database_url="")
    await store.create("run-terminal-pause", "start", {"run_id": "run-terminal-pause"}, {})
    await store.set_status("run-terminal-pause", "completed")
    assert (await store.request_pause("run-terminal-pause"))["status"] == "terminal"

    await store.create("run-checkpoint-pause", "start", {"run_id": "run-checkpoint-pause"}, {})
    await store.set_status("run-checkpoint-pause", "awaiting_human")
    assert (await store.request_pause("run-checkpoint-pause"))["status"] == "already_paused"


@pytest.mark.asyncio
async def test_checkpoint_execution_persists_resumable_result_and_releases_lease() -> None:
    store = ExecutionStore(database_url="")
    await store.create("run-checkpoint", "start", {"run_id": "run-checkpoint"}, {})
    fencing_token = await store.claim("run-checkpoint")

    event = await store.checkpoint_execution(
        "run-checkpoint",
        {"event_id": "run-checkpoint:paused", "kind": "run.paused", "payload": {}, "terminal": False},
        {"status": "awaiting_human", "pending_interrupt": {"type": "task_pause"}},
        status="awaiting_human",
        continuation={"binding_type": "langgraph.checkpoint", "payload": {"checkpoint_thread_id": "cp-1"}},
        owner_id=store.owner_id,
        fencing_token=fencing_token,
    )

    record = await store.get("run-checkpoint")
    assert event["kind"] == "run.paused"
    assert record.status == "awaiting_human"
    assert record.continuation["payload"]["checkpoint_thread_id"] == "cp-1"
    assert record.owner_id is None


@pytest.mark.asyncio
async def test_checkpoint_execution_is_idempotent_after_event_was_already_persisted() -> None:
    store = ExecutionStore(database_url="")
    await store.create("run-checkpoint-retry", "start", {"run_id": "run-checkpoint-retry"}, {})
    first_token = await store.claim("run-checkpoint-retry")
    event = {"event_id": "run-checkpoint-retry:paused", "kind": "run.paused", "payload": {}, "terminal": False}
    result = {"status": "awaiting_human", "pending_interrupt": {"type": "task_pause"}}
    await store.checkpoint_execution(
        "run-checkpoint-retry", event, result, status="awaiting_human", continuation=None,
        owner_id=store.owner_id, fencing_token=first_token,
    )

    # Simulate a retry after the checkpoint was committed and the lease was
    # reacquired by recovery.
    second_token = await store.claim("run-checkpoint-retry")
    repeated = await store.checkpoint_execution(
        "run-checkpoint-retry", event, result, status="awaiting_human", continuation=None,
        owner_id=store.owner_id, fencing_token=second_token,
    )

    assert repeated["event_id"] == (await store.events_after("run-checkpoint-retry"))[0]["event_id"]
    assert len(await store.events_after("run-checkpoint-retry")) == 1
    assert (await store.get("run-checkpoint-retry")).next_sequence == 2


@pytest.mark.asyncio
async def test_terminal_finalization_advances_past_a_stale_checkpoint_sequence() -> None:
    store = ExecutionStore(database_url="")
    await store.create("run-stale-sequence", "start", {"run_id": "run-stale-sequence"}, {})
    await store.append(
        "run-stale-sequence",
        {"event_id": "run-stale-sequence:checkpoint", "kind": "run.paused", "payload": {}, "terminal": False},
    )
    record = await store.get("run-stale-sequence")
    record.next_sequence = 1
    token = await store.claim("run-stale-sequence")

    terminal = await store.finalize_execution(
        "run-stale-sequence",
        {"event_id": "run-stale-sequence:cancelled", "kind": "run.cancelled", "payload": {}, "terminal": True},
        {"status": "cancelled"},
        status="cancelled", owner_id=store.owner_id, fencing_token=token,
    )

    assert terminal["sequence"] == 2
    assert (await store.get("run-stale-sequence")).next_sequence == 3


@pytest.mark.asyncio
async def test_terminal_continuation_probe_is_immutable_under_repeated_start() -> None:
    store = ExecutionStore()

    await store.create("run-1", "continue_run", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})
    await store.set_status("run-1", "no_continuation")
    await store.append(
        "run-1",
        {"event_id": "run-1:terminal", "kind": "run.continuation_empty", "terminal": True, "payload": {}},
    )

    with pytest.raises(ExecutionConflictError):
        await store.create("run-1", "start", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})
    assert (await store.get("run-1")).status == "no_continuation"
    assert len(await store.events_after("run-1")) == 1


@pytest.mark.asyncio
async def test_failed_start_is_immutable_under_transport_retry() -> None:
    store = ExecutionStore()

    await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})
    await store.set_status("run-2", "failed")
    await store.append(
        "run-2",
        {"event_id": "run-2:terminal", "kind": "run.failed", "terminal": True, "payload": {}},
    )

    record = await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})

    assert record.status == "failed"
    assert len(await store.events_after("run-2")) == 1


@pytest.mark.asyncio
async def test_terminal_request_conflict_requires_explicit_retry() -> None:
    store = ExecutionStore()
    await store.create("run-retry", "start", {"run_id": "run-retry", "input": {"question": "one"}}, {"request": {"run_id": "run-retry"}})
    await store.set_status("run-retry", "cancelled")

    with pytest.raises(ExecutionConflictError):
        await store.create("run-retry", "start", {"run_id": "run-retry", "input": {"question": "two"}}, {"request": {"run_id": "run-retry"}})

    retried = await store.create(
        "run-retry",
        "retry",
        {"run_id": "run-retry", "retry_operation": "start", "retry_request": {"run_id": "run-retry"}},
        {"request": {"run_id": "run-retry"}},
        operation_id="retry-1",
        source_attempt=1,
    )
    assert retried.status == "queued"
    assert retried.attempt == 2
    repeated = await store.create(
        "run-retry",
        "retry",
        {"run_id": "run-retry", "retry_operation": "start", "retry_request": {"run_id": "run-retry"}},
        {"request": {"run_id": "run-retry"}},
        operation_id="retry-1",
        source_attempt=1,
    )
    assert repeated.attempt == 2


@pytest.mark.asyncio
async def test_operation_id_replays_identical_request_and_rejects_conflict() -> None:
    store = ExecutionStore()
    first = await store.create(
        "run-idempotent",
        "start",
        {"run_id": "run-idempotent", "input": {"question": "one"}},
        {"request": {"run_id": "run-idempotent"}},
        operation_id="op-1",
    )
    await store.set_status("run-idempotent", "completed", result={"status": "completed", "output": "done"})
    replay = await store.create(
        "run-idempotent",
        "start",
        {"run_id": "run-idempotent", "input": {"question": "one"}},
        {"request": {"run_id": "run-idempotent"}},
        operation_id="op-1",
    )

    assert first.attempt == replay.attempt == 1
    assert replay.replay_only is True
    assert replay.result == {"status": "completed", "output": "done"}

    with pytest.raises(ExecutionConflictError):
        await store.create(
            "run-idempotent",
            "start",
            {"run_id": "run-idempotent", "input": {"question": "two"}},
            {"request": {"run_id": "run-idempotent"}},
            operation_id="op-1",
        )


@pytest.mark.asyncio
async def test_resume_operation_replaces_boundary_identity_and_fingerprints_decision() -> None:
    store = ExecutionStore()
    request = {"run_id": "run-resume-identity"}
    await store.create(
        "run-resume-identity",
        "start",
        request,
        {"request": request, "context": {"task_context": {"metadata": {"task_version": 2}}}},
        operation_id="start-command",
    )
    await store.set_status("run-resume-identity", "awaiting_human")
    resume_payload = {"request": request, "interrupt": {"action": "continue", "action_version": 1}}
    resumed = await store.create(
        "run-resume-identity",
        "resume",
        request,
        resume_payload,
        operation_id="resume-command",
    )

    assert resumed.status == "queued"
    assert resumed.last_operation_id == "resume-command"
    replay = await store.create(
        "run-resume-identity",
        "resume",
        request,
        resume_payload,
        operation_id="resume-command",
    )
    assert replay.replay_only is True
    assert replay.attempt == resumed.attempt

    with pytest.raises(ExecutionConflictError, match="different input"):
        await store.create(
            "run-resume-identity",
            "resume",
            request,
            {"request": request, "interrupt": {"action": "accept_partial", "action_version": 1}},
            operation_id="resume-command",
        )


@pytest.mark.asyncio
async def test_resume_transport_retry_is_read_only_after_terminal_completion() -> None:
    store = ExecutionStore()
    await store.create("run-resume", "start", {"run_id": "run-resume"}, {"request": {"run_id": "run-resume"}})
    binding = {"binding_type": "checkpoint", "payload": {"id": "cp-1"}}
    await store.append(
        "run-resume",
        {"event_id": "run-resume:terminal", "kind": "run.completed", "terminal": True, "continuation": binding},
    )
    await store.set_status("run-resume", "completed")

    with pytest.raises(ExecutionConflictError):
        await store.create("run-resume", "resume", {"run_id": "run-resume"}, {"request": {"run_id": "run-resume"}})
    assert (await store.get("run-resume")).attempt == 1
    assert len(await store.events_after("run-resume")) == 1


@pytest.mark.asyncio
async def test_event_round_trip_preserves_neutral_continuation_metadata() -> None:
    store = ExecutionStore()
    await store.create("run-3", "start", {"run_id": "run-3"}, {"request": {"run_id": "run-3"}})

    continuation = {
        "binding_type": "langgraph_checkpoint",
        "payload": {"checkpoint_id": "checkpoint-1"},
    }
    await store.append(
        "run-3",
        {
            "event_id": "run-3:paused",
            "kind": "run.interrupted",
            "payload": {"reason": "human_input"},
            "occurred_at": "2026-08-17T12:00:00Z",
            "trace_id": "trace-3",
            "continuation": continuation,
            "terminal": True,
        },
    )

    events = await store.events_after("run-3")
    record = await store.get("run-3")
    assert events[0]["continuation"] == continuation
    assert events[0]["trace_id"] == "trace-3"
    assert record.continuation == continuation


@pytest.mark.asyncio
async def test_postgres_event_round_trip_updates_execution_continuation() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL runtime-store coverage")

    store = ExecutionStore(database_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    await store.initialize()
    run_id = f"runtime-store-{uuid.uuid4().hex}"
    try:
        await store.create(run_id, "start", {"run_id": run_id}, {"request": {"run_id": run_id}})
        fencing_token = await store.claim(run_id)
        assert fencing_token is not None
        continuation = {
            "binding_type": "langgraph_checkpoint",
            "payload": {"checkpoint_id": "postgres-checkpoint"},
        }
        await store.append(
            run_id,
            {
                "event_id": f"{run_id}:paused",
                "kind": "run.interrupted",
                "payload": {"reason": "human_input"},
                "trace_id": "trace-pg",
                "continuation": continuation,
                "terminal": True,
            },
            owner_id=store.owner_id,
            fencing_token=fencing_token,
        )
        with pytest.raises(LeaseLostError):
            await store.append(
                run_id,
                {"event_id": f"{run_id}:stale", "kind": "runtime.event"},
                owner_id="stale-worker",
                fencing_token=fencing_token,
            )

        events = await store.events_after(run_id)
        record = await store.get(run_id)
        assert events[0]["continuation"] == continuation
        assert events[0]["trace_id"] == "trace-pg"
        assert record is not None
        assert record.continuation == continuation
    finally:
        # The Docker test runner drops the isolated database.  Keep direct
        # invocations tidy when they reuse a development test database.
        if store._pool is not None:
            await store._pool.execute("delete from runtime_executions where run_id=$1", run_id)
        await store.close()


@pytest.mark.asyncio
async def test_postgres_request_cancel_matches_in_memory_outcomes() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL runtime-store coverage")

    store = ExecutionStore(database_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    await store.initialize()
    run_id = f"runtime-cancel-{uuid.uuid4().hex}"
    try:
        assert (await store.request_cancel(run_id)).outcome == "unknown"
        await store.create(run_id, "start", {"run_id": run_id}, {"request": {"run_id": run_id}})
        requested = await store.request_cancel(run_id)
        assert requested.outcome == "requested"
        assert requested.run_status == "queued"
        fencing_token = await store.claim(run_id)
        assert fencing_token is not None
        await store.set_status(run_id, "cancelled", owner_id=store.owner_id, fencing_token=fencing_token)
        terminal = await store.request_cancel(run_id)
        assert terminal.outcome == "terminal"
        assert terminal.run_status == "cancelled"
        assert (await store.get(run_id)).cancel_requested is False
    finally:
        if store._pool is not None:
            await store._pool.execute("delete from runtime_executions where run_id=$1", run_id)
        await store.close()


@pytest.mark.asyncio
async def test_postgres_course_correction_matches_in_memory_outcomes() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL runtime-store coverage")

    store = ExecutionStore(database_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    await store.initialize()
    run_id = f"runtime-correction-{uuid.uuid4().hex}"
    correction = {
        "correction_id": "correction-1", "operation_id": "operation-1",
        "instruction": "Change remaining work.", "scope": "remaining_work",
        "observed_task_version": 2, "observed_plan_revision": 1,
    }
    try:
        await store.create(run_id, "start", {"run_id": run_id}, {"request": {"run_id": run_id}})
        assert (await store.request_course_correction(run_id, correction))["status"] == "accepted"
        assert (await store.request_course_correction(run_id, correction))["status"] == "already_accepted"
        assert await store.mark_course_corrections_applied(run_id, ["correction-1"], plan_revision=2) == ["correction-1"]
        assert await store.pending_course_corrections(run_id) == []
        assert [value["kind"] for value in await store.events_after(run_id)] == [
            "course_correction.accepted", "course_correction.applied",
        ]
    finally:
        if store._pool is not None:
            await store._pool.execute("delete from runtime_executions where run_id=$1", run_id)
        await store.close()


def test_json_safe_converts_legacy_runtime_objects() -> None:
    value = _json_safe(
        {
            "run": SimpleNamespace(id="run-1"),
            "items": [SimpleNamespace(value=1)],
        }
    )

    assert value == {"run": {"id": "run-1"}, "items": [{"value": 1}]}


@pytest.mark.asyncio
async def test_runtime_lease_fences_competing_workers_and_mutations() -> None:
    store = ExecutionStore()
    await store.create("leased", "start", {"run_id": "leased"}, {"request": {"run_id": "leased"}})
    first = await store.claim("leased", owner_id="worker-a")
    assert first is not None
    assert await store.claim("leased", owner_id="worker-b") is None
    with pytest.raises(LeaseLostError):
        await store.append("leased", {"event_id": "stale", "kind": "runtime.event"}, owner_id="worker-b", fencing_token=1)
    assert await store.heartbeat("leased", owner_id="worker-a", fencing_token=first)


@pytest.mark.asyncio
async def test_atomic_finalization_commits_terminal_result_status_and_lease_release() -> None:
    store = ExecutionStore()
    await store.create("atomic", "start", {"run_id": "atomic"}, {"request": {"run_id": "atomic"}})
    fencing_token = await store.claim("atomic")
    assert fencing_token is not None

    stored = await store.finalize_execution(
        "atomic",
        {"event_id": "atomic:terminal", "kind": "run.completed", "payload": {"status": "completed"}, "terminal": True},
        {"status": "completed", "output": {"answer": "done"}},
        status="completed",
        owner_id=store.owner_id,
        fencing_token=fencing_token,
    )

    record = await store.get("atomic")
    assert stored["result"]["status"] == "completed"
    assert record.status == "completed"
    assert record.owner_id is None
    assert record.lease_expires_at is None
    assert record.heartbeat_at is None
    assert len(await store.events_after("atomic")) == 1
    assert await store.list_recovery_candidates() == []


@pytest.mark.asyncio
async def test_clarification_result_is_terminal_and_replayable() -> None:
    store = ExecutionStore()
    await store.create("clarification", "start", {"run_id": "clarification"}, {"request": {"run_id": "clarification"}})
    fencing_token = await store.claim("clarification")
    assert fencing_token is not None

    stored = await store.finalize_execution(
        "clarification",
        {
            "event_id": "clarification:terminal",
            "kind": "run.clarification",
            "payload": {"status": "clarification_required"},
            "terminal": True,
        },
        {"status": "clarification_required", "clarification": {"options": ["More detail"]}},
        status="clarification_required",
        owner_id=store.owner_id,
        fencing_token=fencing_token,
    )

    record = await store.get("clarification")
    assert stored["result"]["status"] == "clarification_required"
    assert record.status == "clarification_required"
    assert await store.list_recovery_candidates() == []


@pytest.mark.asyncio
async def test_terminal_event_is_reconciled_or_quarantined_without_recovery() -> None:
    store = ExecutionStore()
    await store.create("reconcile", "start", {"run_id": "reconcile"}, {"request": {"run_id": "reconcile"}})
    await store.append(
        "reconcile",
        {"event_id": "terminal", "kind": "run.completed", "payload": {}, "terminal": True},
        result={"status": "completed", "output": {"answer": "done"}},
    )
    await store.create("quarantine", "start", {"run_id": "quarantine"}, {"request": {"run_id": "quarantine"}})
    await store.append(
        "quarantine",
        {"event_id": "terminal", "kind": "run.completed", "payload": {}, "terminal": True},
    )

    assert await store.list_recovery_candidates() == []
    assert {record.run_id for record in await store.list_terminal_reconciliation_candidates()} == {"reconcile", "quarantine"}
    assert await store.reconcile_terminal_execution("reconcile") == "reconciled"
    assert await store.reconcile_terminal_execution("quarantine") == "quarantined"
    assert (await store.get("reconcile")).status == "completed"
    assert (await store.get("quarantine")).status == "failed"
    replay = await store.create(
        "quarantine", "start", {"run_id": "quarantine"}, {"request": {"run_id": "quarantine"}},
    )
    assert replay.status == "failed"


@pytest.mark.asyncio
async def test_course_correction_is_ordered_idempotent_and_applied_after_ack() -> None:
    store = ExecutionStore()
    await store.create("corrected", "start", {"run_id": "corrected"}, {"request": {"run_id": "corrected"}})
    correction = {
        "correction_id": "correction-1",
        "operation_id": "operation-1",
        "instruction": "Focus on the remaining security analysis.",
        "scope": "remaining_work",
        "observed_task_version": 3,
        "observed_plan_revision": 1,
    }

    accepted = await store.request_course_correction("corrected", correction)
    duplicate = await store.request_course_correction("corrected", correction)

    assert accepted["status"] == "accepted"
    assert duplicate["status"] == "already_accepted"
    assert [value["correction_id"] for value in await store.pending_course_corrections("corrected")] == ["correction-1"]
    assert await store.mark_course_corrections_applied("corrected", ["correction-1"], plan_revision=2) == ["correction-1"]
    assert await store.pending_course_corrections("corrected") == []
    assert [value["kind"] for value in await store.events_after("corrected")] == [
        "course_correction.accepted",
        "course_correction.applied",
    ]


@pytest.mark.asyncio
async def test_course_correction_rejects_conflicting_operation_and_reports_terminal_race() -> None:
    store = ExecutionStore()
    await store.create("corrected", "start", {"run_id": "corrected"}, {"request": {"run_id": "corrected"}})
    correction = {
        "correction_id": "correction-1", "operation_id": "operation-1",
        "instruction": "First instruction", "scope": "remaining_work",
        "observed_task_version": 1, "observed_plan_revision": 0,
    }
    await store.request_course_correction("corrected", correction)
    with pytest.raises(ExecutionConflictError):
        await store.request_course_correction("corrected", {**correction, "instruction": "Different instruction"})

    terminal = await store.get("corrected")
    terminal.status = "completed"
    receipt = await store.request_course_correction("corrected", {**correction, "operation_id": "operation-2"})
    assert receipt["status"] == "terminal"
    assert receipt["run_status"] == "completed"
