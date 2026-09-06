from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import case, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import AgentRunStatus
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRun,
    AgentRunEvent,
    AgentTask,
    AgentTaskArtifact,
    AgentTaskCommand,
    AgentTaskEvent,
    AgentTaskPlanRevision,
    AgentTaskRuntimeDelta,
    AgentTaskSubagentRun,
    AgentTaskTodo,
)
from app.models.deep_research import AgentTaskStatus, DeepResearchPlanProposal
from app.time_utils import parse_datetime_utc, utc_now
from app.agent_workflows.trace_details import sanitize_trace_detail
from app.agent_workflows.trace_payloads import append_runtime_event_to_debug_payload
from runtime_protocol.contracts import TERMINAL_RUNTIME_EVENT_KINDS
from runtime_protocol.events import normalize_product_event_kind
from app.runtime.behavior import continuation_is_linked, supports_course_correction
from app.services.agent_task_budgets import (
    exhausted_dimensions,
    initial_budget_state,
    normalize_budget_state,
    reset_tranche,
)


ACTIVE_TASK_STATUSES = {
    AgentTaskStatus.QUEUED.value,
    AgentTaskStatus.RUNNING.value,
    AgentTaskStatus.PAUSING.value,
    AgentTaskStatus.PAUSED.value,
    AgentTaskStatus.AWAITING_APPROVAL.value,
    AgentTaskStatus.CANCELLING.value,
    AgentTaskStatus.RECOVERY_REQUIRED.value,
}
TERMINAL_TASK_STATUSES = {
    AgentTaskStatus.CANCELLED.value,
    AgentTaskStatus.COMPLETED.value,
    AgentTaskStatus.FAILED.value,
    AgentTaskStatus.EXPIRED.value,
}
# A recovery-required task has been deliberately taken out of the execution
# path: its runtime result is known, but product projection needs intervention.
# It is therefore safe to hide/delete, even though it is kept separate from
# normal terminal states for retry/reconciliation and capability decisions.
DELETABLE_TASK_STATUSES = TERMINAL_TASK_STATUSES | {
    AgentTaskStatus.RECOVERY_REQUIRED.value,
}
ACTIVE_TASK_RUN_STATUSES = {
    AgentRunStatus.RUNNING.value,
    AgentRunStatus.AWAITING_HUMAN.value,
}
TERMINAL_TASK_RUN_STATUSES = {
    AgentRunStatus.COMPLETED.value,
    AgentRunStatus.FAILED.value,
    AgentRunStatus.EXPIRED.value,
    AgentRunStatus.CANCELLED.value,
    AgentRunStatus.REJECTED.value,
}
WEB_ACCESS_EVENT_PREFIX = "web_access."
WEB_ACCESS_ALLOWED = "allowed_for_task"
WEB_ACCESS_DENIED = "denied_for_task"


class AgentTaskConflict(ValueError):
    def __init__(self, code: str, message: str, *, current_version: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.current_version = current_version


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def create_task(
    *,
    thread_id: str,
    project_id: Optional[str],
    user_id: Optional[str],
    workflow_id: str,
    objective: str,
    idempotency_key: str,
    config: Dict[str, Any],
) -> tuple[AgentTask, bool]:
    objective = " ".join(objective.split()).strip()
    async with async_session_maker() as session:
        existing_query = select(AgentTask).where(
            AgentTask.thread_id == thread_id,
            AgentTask.create_idempotency_key == idempotency_key,
        )
        existing_query = existing_query.where(AgentTask.user_id.is_(None)) if user_id is None else existing_query.where(AgentTask.user_id == user_id)
        existing = (await session.execute(existing_query)).scalar_one_or_none()
        if existing:
            return existing, True
        task = AgentTask(
            thread_id=thread_id,
            project_id=project_id,
            user_id=user_id,
            workflow_id=workflow_id,
            objective=objective,
            objective_hash=hashlib.sha256(objective.casefold().encode("utf-8")).hexdigest(),
            create_idempotency_key=idempotency_key,
            config_json=config,
            budgets_json=initial_budget_state((config or {}).get("limits") or {}),
            expires_at=utc_now() + timedelta(hours=24),
        )
        session.add(task)
        try:
            await session.flush()
            await _append_event(session, task, "task.created", payload={"status": task.status})
            await session.commit()
            await session.refresh(task)
            return task, False
        except IntegrityError:
            await session.rollback()
            existing = (await session.execute(existing_query)).scalar_one_or_none()
            if existing:
                return existing, True
            raise


async def get_task(
    task_id: str,
    *,
    thread_id: Optional[str] = None,
    include_deleted: bool = False,
) -> Optional[AgentTask]:
    async with async_session_maker() as session:
        query = select(AgentTask).where(AgentTask.id == task_id)
        if thread_id is not None:
            query = query.where(AgentTask.thread_id == thread_id)
        if not include_deleted:
            query = query.where(AgentTask.deletion_requested_at.is_(None))
        return (await session.execute(query)).scalar_one_or_none()


async def get_task_run(task_id: str) -> Optional[AgentRun]:
    async with async_session_maker() as session:
        task = await session.get(AgentTask, task_id)
        return await session.get(AgentRun, task.active_run_id) if task and task.active_run_id else None


async def list_task_runs(task_id: str) -> list[AgentRun]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun)
            .where(AgentRun.task_id == task_id)
            .order_by(AgentRun.task_attempt, AgentRun.started_at, AgentRun.id)
        )
        return list(result.scalars().all())


async def task_cancel_requested(task_id: str) -> bool:
    task = await get_task(task_id)
    return bool(task and task.status in {AgentTaskStatus.CANCELLING.value, AgentTaskStatus.CANCELLED.value})


async def run_cancel_requested(run_id: str) -> bool:
    """Resolve durable product cancellation from a canonical agent run."""

    if not str(run_id or "").strip():
        raise ValueError("A canonical agent run id is required for cancellation")
    async with async_session_maker() as session:
        run = await session.get(AgentRun, run_id)
        if run is None:
            raise ValueError(f"Agent run {run_id!r} does not exist")
        if run.status == AgentRunStatus.CANCELLED.value:
            return True
        if not run.task_id:
            return False
        task = await session.get(AgentTask, run.task_id)
        if task is None:
            raise ValueError(f"Agent run {run_id!r} has no owning task")
        return task.status in {AgentTaskStatus.CANCELLING.value, AgentTaskStatus.CANCELLED.value}


async def consume_budget(
    task_id: str,
    *,
    model_calls: int = 0,
    model_tokens: int = 0,
    tool_calls: int = 0,
) -> Dict[str, Any]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            limits = (task.config_json or {}).get("limits") or {}
            usage = normalize_budget_state(task.budgets_json, limits)
            increments = {"model_calls": model_calls, "model_tokens": model_tokens, "tool_calls": tool_calls}
            for key, amount in increments.items():
                increment = max(0, int(amount or 0))
                usage["tranche_usage"][key] = int(usage["tranche_usage"].get(key) or 0) + increment
                usage["lifetime_usage"][key] = int(usage["lifetime_usage"].get(key) or 0) + increment
            exhausted = exhausted_dimensions(usage)
            if exhausted and not isinstance(usage.get("boundary"), dict):
                usage["boundary"] = {
                    "status": "requested",
                    "dimensions": exhausted,
                    "tranche_index": usage["tranche_index"],
                    "requested_at": utc_now().isoformat(),
                }
            replace_jsonb_field(task, "budgets_json", usage)
            task.version += 1
            await _append_event(session, task, "task.budget_updated", agent_run_id=task.active_run_id, payload={
                "tranche_index": usage["tranche_index"],
                "tranche_usage": usage["tranche_usage"],
                "lifetime_usage": usage["lifetime_usage"],
                "exhausted_dimensions": exhausted,
            })
        return usage


async def queue_task_after_interrupt(
    task_id: str,
    *,
    reason: str,
    interrupt_id: Optional[str] = None,
    action: Optional[str] = None,
) -> Optional[AgentTask]:
    """Queue the same checkpoint thread after the canonical interrupt resolver succeeds."""
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                return None
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = AgentTaskStatus.QUEUED.value
            task.current_phase = "continuing"
            task.queued_at = utc_now()
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, "task.continuation_queued", agent_run_id=task.active_run_id, payload={"reason": reason, "version": task.version})
            if interrupt_id and action:
                await _append_event(
                    session,
                    task,
                    "task.approval_resolved",
                    agent_run_id=task.active_run_id,
                    payload={"interrupt_id": interrupt_id, "action": action, "version": task.version},
                )
        await session.refresh(task)
        return task


async def requeue_after_wake(task_id: str, *, reason: str) -> Optional[AgentTask]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = AgentTaskStatus.QUEUED.value
            task.current_phase = "continuation_queued"
            task.queued_at = utc_now()
            task.expires_at = utc_now() + timedelta(hours=24)
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, "task.wake_budget_reached", agent_run_id=task.active_run_id, payload={
                "reason": reason, "version": task.version,
            })
        return task


async def list_tasks(thread_id: str, *, limit: int = 50) -> list[AgentTask]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTask)
            .where(AgentTask.thread_id == thread_id, AgentTask.deletion_requested_at.is_(None))
            .order_by(AgentTask.created_at.desc(), AgentTask.id.desc())
            .limit(max(1, min(limit, 100)))
        )
        return list(result.scalars().all())


async def list_todos(task_id: str) -> list[AgentTaskTodo]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskTodo)
            .where(AgentTaskTodo.task_id == task_id)
            .order_by(AgentTaskTodo.priority.desc(), AgentTaskTodo.created_at, AgentTaskTodo.id)
        )
        return list(result.scalars().all())


async def get_latest_plan(task_id: str, *, agent_run_id: Optional[str] = None) -> Optional[AgentTaskPlanRevision]:
    async with async_session_maker() as session:
        query = select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskPlanRevision.agent_run_id == agent_run_id)
        return (await session.execute(query.order_by(AgentTaskPlanRevision.revision.desc()).limit(1))).scalar_one_or_none()


async def list_plans(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskPlanRevision]:
    async with async_session_maker() as session:
        query = select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskPlanRevision.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskPlanRevision.revision, AgentTaskPlanRevision.created_at))
        return list(result.scalars().all())


async def latest_applied_runtime_plan_revision(task_id: str) -> int:
    async with async_session_maker() as session:
        value = (await session.execute(select(func.coalesce(
            func.max(AgentTaskRuntimeDelta.applied_runtime_plan_revision), 0,
        )).where(AgentTaskRuntimeDelta.task_id == task_id))).scalar_one()
        return int(value or 0)


async def list_artifacts(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        query = select(AgentTaskArtifact).where(
            AgentTaskArtifact.task_id == task_id,
            AgentTaskArtifact.validity != "deleted",
        )
        if agent_run_id is not None:
            query = query.where(AgentTaskArtifact.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskArtifact.created_at, AgentTaskArtifact.id))
        return list(result.scalars().all())


async def get_artifact(task_id: str, artifact_id: str) -> Optional[AgentTaskArtifact]:
    async with async_session_maker() as session:
        return (await session.execute(select(AgentTaskArtifact).where(
            AgentTaskArtifact.task_id == task_id,
            AgentTaskArtifact.id == artifact_id,
            AgentTaskArtifact.validity != "deleted",
        ))).scalar_one_or_none()


async def list_artifacts_for_threads(thread_ids: Iterable[str]) -> list[AgentTaskArtifact]:
    ids = {str(value) for value in thread_ids if value}
    if not ids:
        return []
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .join(AgentTask, AgentTask.id == AgentTaskArtifact.task_id)
            .where(AgentTask.thread_id.in_(ids), AgentTaskArtifact.validity != "deleted")
        )
        return list(result.scalars().all())


async def list_task_runtime_runs_for_threads(thread_ids: Iterable[str]) -> list[AgentRun]:
    ids = {str(value) for value in thread_ids if value}
    if not ids:
        return []
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun)
            .where(
                AgentRun.thread_id.in_(ids),
                AgentRun.task_id.is_not(None),
                AgentRun.runtime_binding_json.is_not(None),
            )
        )
        return list(result.scalars().all())


async def list_terminal_task_runtime_runs_before(cutoff: Any, *, limit: int = 100) -> list[AgentRun]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentRun)
            .where(
                AgentRun.task_id.is_not(None),
                AgentRun.completed_at.is_not(None),
                AgentRun.completed_at <= cutoff,
                AgentRun.status.in_(TERMINAL_TASK_RUN_STATUSES),
                AgentRun.runtime_binding_json.is_not(None),
            )
            .order_by(AgentRun.completed_at, AgentRun.id)
            .limit(max(1, min(limit, 500)))
        )
        return [run for run in result.scalars().all() if run.runtime_binding_json]


async def clear_task_runtime_bindings(run_ids: Iterable[str]) -> int:
    ids = {str(value) for value in run_ids if value}
    if not ids:
        return 0
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentRun).where(
                    AgentRun.task_id.is_not(None),
                    AgentRun.id.in_(ids),
                ).with_for_update()
            )).scalars().all())
            for run in rows:
                run.runtime_binding_json = None
                run.runtime_binding_status = "cleaned"
            return len(rows)


async def release_stale_task_leases(*, limit: int = 100) -> int:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.lease_owner.is_not(None),
                    AgentTask.lease_expires_at.is_not(None),
                    AgentTask.lease_expires_at < now,
                    AgentTask.status.in_([
                        AgentTaskStatus.RUNNING.value,
                        AgentTaskStatus.CANCELLING.value,
                    ]),
                )
                .order_by(AgentTask.lease_expires_at, AgentTask.id)
                .with_for_update(skip_locked=True)
                .limit(max(1, min(limit, 500)))
            )).scalars().all())
            for task in rows:
                task.lease_owner = None
                task.lease_expires_at = None
                task.updated_at = now
                await _append_event(
                    session,
                    task,
                    "task.lease_recovered",
                    agent_run_id=task.active_run_id,
                    payload={"status": task.status},
                )
            return len(rows)


async def mark_artifact_deleted(task_id: str, artifact_id: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            artifact = await session.get(AgentTaskArtifact, artifact_id)
            if task is None or artifact is None or artifact.task_id != task_id or artifact.validity == "deleted":
                return
            artifact.validity = "deleted"
            artifact.deleted_at = utc_now()
            await _append_event(session, task, "artifact.deleted", agent_run_id=artifact.agent_run_id, artifact_id=artifact.id, payload={"sha256": artifact.sha256})


async def mark_artifact_invalid(task_id: str, artifact_id: str, *, reason: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            artifact = await session.get(AgentTaskArtifact, artifact_id)
            if task is None or artifact is None or artifact.task_id != task_id or artifact.validity != "valid":
                return
            artifact.validity = "invalid"
            await _append_event(
                session,
                task,
                "artifact.invalidated",
                agent_run_id=artifact.agent_run_id,
                artifact_id=artifact.id,
                payload={"reason": reason, "sha256": artifact.sha256},
            )


async def list_expired_artifacts(*, limit: int = 100) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .where(
                AgentTaskArtifact.validity != "deleted",
                AgentTaskArtifact.retention_until.is_not(None),
                AgentTaskArtifact.retention_until <= utc_now(),
            )
            .order_by(AgentTaskArtifact.retention_until, AgentTaskArtifact.id)
            .limit(max(1, min(limit, 500)))
        )
        return list(result.scalars().all())


async def list_live_artifacts(*, limit: int = 10_000) -> list[AgentTaskArtifact]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTaskArtifact)
            .where(AgentTaskArtifact.validity != "deleted")
            .order_by(AgentTaskArtifact.created_at, AgentTaskArtifact.id)
            .limit(max(1, min(limit, 10_000)))
        )
        return list(result.scalars().all())


async def invalidate_context_summaries(task_id: str, *, source_hash: str) -> int:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            rows = list((await session.execute(select(AgentTaskArtifact).where(
                AgentTaskArtifact.task_id == task_id,
                AgentTaskArtifact.kind == "context_summary",
                AgentTaskArtifact.validity == "valid",
            ).with_for_update())).scalars().all())
            invalidated = 0
            for artifact in rows:
                if (artifact.provenance_json or {}).get("source_hash") == source_hash:
                    continue
                artifact.validity = "invalid"
                invalidated += 1
                await _append_event(session, task, "artifact.invalidated", agent_run_id=artifact.agent_run_id, artifact_id=artifact.id, payload={
                    "reason": "source_hash_changed", "replacement_source_hash": source_hash,
                })
            return invalidated


async def list_subagent_runs(task_id: str, *, agent_run_id: Optional[str] = None) -> list[AgentTaskSubagentRun]:
    async with async_session_maker() as session:
        query = select(AgentTaskSubagentRun).where(AgentTaskSubagentRun.task_id == task_id)
        if agent_run_id is not None:
            query = query.where(AgentTaskSubagentRun.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskSubagentRun.created_at, AgentTaskSubagentRun.id))
        return list(result.scalars().all())


async def list_events(
    task_id: str,
    *,
    agent_run_id: Optional[str] = None,
    after_sequence: int = 0,
    limit: int = 500,
) -> list[AgentTaskEvent]:
    async with async_session_maker() as session:
        query = select(AgentTaskEvent).where(
            AgentTaskEvent.task_id == task_id,
            AgentTaskEvent.sequence > max(0, after_sequence),
        )
        if agent_run_id is not None:
            query = query.where(AgentTaskEvent.agent_run_id == agent_run_id)
        result = await session.execute(query.order_by(AgentTaskEvent.sequence).limit(max(1, min(limit, 1000))))
        return list(result.scalars().all())


async def _append_event(
    session,
    task: AgentTask,
    event_type: str,
    *,
    actor_type: str = "system",
    actor_id: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    todo_id: Optional[str] = None,
    subagent_run_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    policy_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    causal_key: Optional[str] = None,
) -> AgentTaskEvent:
    normalized_type, source_metadata = normalize_product_event_kind(event_type)
    if causal_key is None and agent_run_id and normalized_type in TERMINAL_RUNTIME_EVENT_KINDS:
        causal_key = f"run:{agent_run_id}:terminal"
    if causal_key:
        existing = (await session.execute(select(AgentTaskEvent).where(
            AgentTaskEvent.task_id == task.id,
            AgentTaskEvent.causal_key == causal_key,
        ).with_for_update())).scalar_one_or_none()
        if existing is not None:
            return existing
    latest = await session.execute(
        select(func.coalesce(func.max(AgentTaskEvent.sequence), 0))
        .where(AgentTaskEvent.task_id == task.id)
    )
    event_sequence = int(latest.scalar_one()) + 1
    event = AgentTaskEvent(
        task_id=task.id,
        sequence=event_sequence,
        event_id=f"{task.id}:{event_sequence}",
        causal_key=causal_key,
        event_type=normalized_type,
        actor_type=actor_type,
        actor_id=actor_id,
        agent_run_id=agent_run_id,
        todo_id=todo_id,
        subagent_run_id=subagent_run_id,
        artifact_id=artifact_id,
        payload_json=sanitize_trace_detail(payload or {})[0],
        policy_hash=policy_hash,
        config_hash=config_hash,
        occurred_at=utc_now(),
        terminal=normalized_type in TERMINAL_RUNTIME_EVENT_KINDS,
        source_metadata_json=source_metadata,
    )
    session.add(event)
    await session.flush()
    return event


async def append_event(task_id: str, event_type: str, **kwargs) -> AgentTaskEvent:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            event = await _append_event(session, task, event_type, **kwargs)
        await session.refresh(event)
        return event


async def get_task_web_access(task_id: str) -> str:
    async with async_session_maker() as session:
        event = (await session.execute(
            select(AgentTaskEvent)
            .where(
                AgentTaskEvent.task_id == task_id,
                AgentTaskEvent.event_type == "approval.responded",
            )
            .order_by(AgentTaskEvent.sequence.desc())
            .limit(1)
        )).scalar_one_or_none()
        return str((event.payload_json or {}).get("status") or "undecided") if event else "undecided"


async def set_task_web_access(
    task_id: str,
    status: str,
    *,
    agent_run_id: str,
    interrupt_id: str,
    actor_id: Optional[str] = None,
) -> AgentTask:
    if status not in {WEB_ACCESS_ALLOWED, WEB_ACCESS_DENIED}:
        raise ValueError("unknown task web-access status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one()
            prior_events = list((await session.execute(
                select(AgentTaskEvent)
                .where(
                    AgentTaskEvent.task_id == task_id,
                    AgentTaskEvent.event_type == "approval.responded",
                )
                .order_by(AgentTaskEvent.sequence.desc())
                .limit(100)
            )).scalars().all())
            if any(
                str((event.payload_json or {}).get("interrupt_id") or "") == interrupt_id
                and str((event.payload_json or {}).get("status") or "") == status
                for event in prior_events
            ):
                return task
            task.version += 1
            task.updated_at = utc_now()
            await _append_event(
                session,
                task,
                f"{WEB_ACCESS_EVENT_PREFIX}{status}",
                actor_type="user",
                actor_id=actor_id,
                agent_run_id=agent_run_id,
                payload={"interrupt_id": interrupt_id, "scope": "task", "status": status, "version": task.version},
            )
        await session.refresh(task)
        return task


COMMAND_TRANSITIONS = {
    "start": ({AgentTaskStatus.CREATED.value}, AgentTaskStatus.QUEUED.value),
    "pause": ({AgentTaskStatus.QUEUED.value, AgentTaskStatus.RUNNING.value}, AgentTaskStatus.PAUSING.value),
    "resume": ({AgentTaskStatus.PAUSED.value}, AgentTaskStatus.QUEUED.value),
    "cancel": ({*ACTIVE_TASK_STATUSES, AgentTaskStatus.CREATED.value}, AgentTaskStatus.CANCELLING.value),
    "retry": ({AgentTaskStatus.FAILED.value, AgentTaskStatus.EXPIRED.value, AgentTaskStatus.RECOVERY_REQUIRED.value}, AgentTaskStatus.QUEUED.value),
    "expire": ({AgentTaskStatus.PAUSED.value, AgentTaskStatus.AWAITING_APPROVAL.value}, AgentTaskStatus.EXPIRED.value),
}


async def apply_command(
    task_id: str,
    *,
    action: str,
    idempotency_key: str,
    expected_version: int,
    actor_id: Optional[str] = None,
) -> tuple[AgentTask, AgentTaskCommand, bool]:
    if action not in COMMAND_TRANSITIONS:
        raise AgentTaskConflict("task_command_unknown", f"Unsupported task command: {action}")
    async with async_session_maker() as session:
        async with session.begin():
            duplicate = (await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == action,
                AgentTaskCommand.idempotency_key == idempotency_key,
            ))).scalar_one_or_none()
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                raise AgentTaskConflict("task_not_found", "Agent task not found")
            if duplicate:
                return task, duplicate, True
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            allowed, target = COMMAND_TRANSITIONS[action]
            if task.status not in allowed:
                raise AgentTaskConflict("task_transition_invalid", f"Cannot {action} task from {task.status}", current_version=task.version)
            command = AgentTaskCommand(
                task_id=task.id,
                action=action,
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                actor_id=actor_id,
            )
            session.add(command)
            now = utc_now()
            runtime_submitted = False
            if action == "cancel" and task.active_run_id is not None:
                active_run = (await session.execute(
                    select(AgentRun).where(AgentRun.id == task.active_run_id).with_for_update()
                )).scalar_one_or_none()
                runtime_submitted = bool(
                    active_run
                    and (active_run.run_metadata_json or {}).get("runtime_started") is True
                )
            if action == "pause" and task.status == AgentTaskStatus.QUEUED.value:
                target = AgentTaskStatus.PAUSED.value
            elif action == "cancel" and not runtime_submitted and task.status in {
                AgentTaskStatus.CREATED.value,
                AgentTaskStatus.QUEUED.value,
                AgentTaskStatus.PAUSED.value,
                AgentTaskStatus.AWAITING_APPROVAL.value,
            }:
                target = AgentTaskStatus.CANCELLED.value
            task.status = target
            task.current_phase = target
            task.version += 1
            task.updated_at = now
            if target == AgentTaskStatus.QUEUED.value:
                task.queued_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.expires_at = now + timedelta(hours=24)
                if action == "retry":
                    task.completed_at = None
                    task.paused_at = None
                    task.terminal_reason = None
                    todos = list((await session.execute(
                        select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id).with_for_update()
                    )).scalars().all())
                    for todo in todos:
                        if todo.status not in {"failed", "blocked", "cancelled"}:
                            continue
                        todo.status = "pending"
                        todo.attempt = 0
                        todo.progress = 0
                        todo.result_summary = None
                        todo.terminal_reason = None
                        todo.current_subagent_run_id = None
                        replace_jsonb_field(todo, "artifact_ids_json", [])
                        replace_jsonb_field(todo, "evidence_ids_json", [])
                        todo.version += 1
                        todo.updated_at = now
                    completed = sum(1 for todo in todos if todo.status == "completed")
                    task.completed_todos = completed
                    task.total_todos = len(todos)
                    task.progress = int((completed * 100) / len(todos)) if todos else 0
            if target == AgentTaskStatus.EXPIRED.value:
                task.completed_at = now
                task.terminal_reason = "approval_or_pause_expired"
            if target == AgentTaskStatus.PAUSED.value:
                task.paused_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.expires_at = now + timedelta(days=7)
            if target == AgentTaskStatus.CANCELLED.value:
                task.completed_at = now
                task.terminal_reason = "cancelled_by_user"
                task.lease_owner = None
                task.lease_expires_at = None
            if action == "cancel":
                pending_corrections = list((await session.execute(select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task.id,
                    AgentTaskCommand.action == "steer",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update())).scalars().all())
                for pending_command in pending_corrections:
                    pending_result = dict(pending_command.result_json or {})
                    pending_result.update({
                        "delivery_state": "rejected",
                        "error": {"code": "course_correction_cancelled"},
                    })
                    replace_jsonb_field(pending_command, "result_json", pending_result)
                    pending_command.status = "rejected"
                    pending_command.completed_at = now
                if pending_corrections:
                    await _append_event(
                        session,
                        task,
                        "task.course_correction_rejected",
                        agent_run_id=task.active_run_id,
                        payload={
                            "reason": "task_cancelled",
                            "command_ids": [value.id for value in pending_corrections],
                        },
                    )
            command.status = "accepted" if action == "cancel" and target == AgentTaskStatus.CANCELLING.value else "completed"
            command.result_version = task.version
            replace_jsonb_field(command, "result_json", {"task_id": task.id, "status": task.status, "version": task.version})
            command.completed_at = None if command.status == "accepted" else now
            await _append_event(
                session,
                task,
                f"task.{action}_requested",
                actor_type="user",
                actor_id=actor_id,
                agent_run_id=task.active_run_id,
                payload={"status": task.status, "version": task.version},
            )
        await session.refresh(task)
        await session.refresh(command)
        return task, command, False


async def complete_control_command(command_id: str, *, result: dict[str, Any] | None = None, rejected: bool = False) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            command = await session.get(AgentTaskCommand, command_id, with_for_update=True)
            if command is not None:
                command.status = "rejected" if rejected else "completed"
                command.result_json = dict(result or {})
                command.completed_at = utc_now()


async def complete_pending_cancel_commands(task_id: str, *, result: dict[str, Any]) -> None:
    """Close accepted cancel commands after authoritative runtime confirmation."""

    async with async_session_maker() as session:
        async with session.begin():
            commands = list((await session.execute(
                select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task_id,
                    AgentTaskCommand.action == "cancel",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update()
            )).scalars().all())
            now = utc_now()
            for command in commands:
                command.status = "completed"
                command.result_json = dict(result)
                command.completed_at = now


async def request_task_deletion(
    task_id: str,
    *,
    idempotency_key: str,
    expected_version: int,
    actor_id: Optional[str] = None,
) -> tuple[AgentTask, AgentTaskCommand, bool]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None:
                raise AgentTaskConflict("task_not_found", "Agent task not found")
            duplicate = (await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "delete",
                AgentTaskCommand.idempotency_key == idempotency_key,
            ))).scalar_one_or_none()
            if duplicate:
                return task, duplicate, True
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            if task.status not in DELETABLE_TASK_STATUSES:
                raise AgentTaskConflict(
                    "task_delete_nonterminal",
                    "Only completed, failed, expired, cancelled, or recovery-required tasks can be deleted",
                    current_version=task.version,
                )
            now = utc_now()
            command = AgentTaskCommand(
                task_id=task.id,
                action="delete",
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                actor_id=actor_id,
                status="completed",
                result_version=task.version + 1,
                result_json={"task_id": task.id, "hidden": True},
                completed_at=now,
            )
            session.add(command)
            task.deletion_requested_at = task.deletion_requested_at or now
            pending_corrections = list((await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task.id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).with_for_update())).scalars().all())
            for pending_command in pending_corrections:
                pending_result = dict(pending_command.result_json or {})
                pending_result.update({
                    "delivery_state": "rejected",
                    "error": {"code": "course_correction_task_deleted"},
                })
                replace_jsonb_field(pending_command, "result_json", pending_result)
                pending_command.status = "rejected"
                pending_command.completed_at = now
            task.version += 1
            task.updated_at = now
            await _append_event(session, task, "task.deletion_requested", actor_type="user", actor_id=actor_id)
            if pending_corrections:
                await _append_event(
                    session,
                    task,
                    "task.course_correction_rejected",
                    actor_type="user",
                    actor_id=actor_id,
                    agent_run_id=task.active_run_id,
                    payload={"reason": "task_deleted", "command_ids": [value.id for value in pending_corrections]},
                )
        await session.refresh(task)
        await session.refresh(command)
        return task, command, False


async def list_pending_task_deletions(*, limit: int = 25) -> list[str]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentTask.id)
            .where(AgentTask.deletion_requested_at.is_not(None), AgentTask.deletion_completed_at.is_(None))
            .order_by(AgentTask.deletion_requested_at)
            .limit(max(1, min(limit, 100)))
        )
        return [str(value) for value in result.scalars().all()]


async def mark_task_deletion_completed(task_id: str) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.deletion_completed_at is not None:
                return
            task.deletion_completed_at = utc_now()
            task.updated_at = task.deletion_completed_at
            task.version += 1
            await _append_event(session, task, "task.deletion_completed")


async def claim_next_task(worker_id: str, *, lease_seconds: int = 60) -> Optional[AgentTask]:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.status.in_([
                        AgentTaskStatus.QUEUED.value,
                        AgentTaskStatus.RUNNING.value,
                        AgentTaskStatus.CANCELLING.value,
                    ]),
                    or_(
                        AgentTask.status != AgentTaskStatus.QUEUED.value,
                        AgentTask.current_phase.is_(None),
                        AgentTask.current_phase != "budget_correction_delivery_pending",
                    ),
                    or_(
                        AgentTask.current_phase.is_(None),
                        AgentTask.current_phase != "runtime_projection_recovery_required",
                    ),
                    or_(AgentTask.lease_expires_at.is_(None), AgentTask.lease_expires_at < now),
                )
                # Cancellation recovery must never starve runnable work. A
                # runtime may acknowledge cancellation before it reaches a
                # terminal boundary, so cancelling tasks are retried only
                # after queued/running work that is ready now.
                .order_by(
                    case(
                        (AgentTask.status == AgentTaskStatus.CANCELLING.value, 1),
                        else_=0,
                    ),
                    AgentTask.queued_at,
                    AgentTask.created_at,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )).scalar_one_or_none()
            if task is None:
                return None
            was_queued = task.status == AgentTaskStatus.QUEUED.value
            is_cancelling = task.status == AgentTaskStatus.CANCELLING.value
            if not is_cancelling:
                task.status = AgentTaskStatus.RUNNING.value
                task.current_phase = "executing"
            task.lease_owner = worker_id
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=max(15, lease_seconds))
            task.started_at = task.started_at or now
            if not is_cancelling:
                task.expires_at = None
            if was_queued:
                task.completed_at = None
                task.terminal_reason = None
            task.version += 1
            task.updated_at = now
            await _append_event(session, task, "task.claimed", agent_run_id=task.active_run_id, payload={"worker_id": worker_id, "version": task.version})
        await session.refresh(task)
        return task


async def heartbeat_task(task_id: str, worker_id: str, *, lease_seconds: int = 60) -> bool:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or task.lease_owner != worker_id or task.status != AgentTaskStatus.RUNNING.value:
                return False
            now = utc_now()
            await _accrue_active_runtime(session, task, now=now, cap_ms=max(15, lease_seconds) * 1000)
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=max(15, lease_seconds))
            task.updated_at = now
            return task.status == AgentTaskStatus.RUNNING.value


async def release_task_lease(task_id: str, worker_id: str, *, lease_seconds: int = 60) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            if task is not None and task.lease_owner == worker_id:
                now = utc_now()
                await _accrue_active_runtime(session, task, now=now, cap_ms=max(15, lease_seconds) * 1000)
                task.lease_owner = None
                task.lease_expires_at = None
                task.heartbeat_at = now


async def defer_task_lease(task_id: str, worker_id: str, *, retry_seconds: float) -> None:
    """Release a task while preventing a tight recovery/reclaim loop."""

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one_or_none()
            if task is not None and task.lease_owner == worker_id:
                now = utc_now()
                task.lease_owner = None
                task.lease_expires_at = now + timedelta(seconds=max(0.2, retry_seconds))
                task.heartbeat_at = now


async def _accrue_active_runtime(session: Any, task: AgentTask, *, now: Any, cap_ms: int) -> int:
    if task.active_run_id:
        active_run = await session.get(AgentRun, task.active_run_id)
        behavior = dict((active_run.run_metadata_json or {}).get("runtime_behavior") or {}) if active_run is not None else {}
        if active_run is not None and behavior.get("usage_accounting_owner") == "runtime":
            # Runtime-owned accounting is projected at runtime boundaries.
            # Product heartbeats only own the lease and must not double count.
            return 0
    previous = task.heartbeat_at
    if previous is None or now <= previous:
        return 0
    increment = min(max(0, int((now - previous).total_seconds() * 1000)), max(1, cap_ms))
    if increment <= 0:
        return 0
    limits = (task.config_json or {}).get("limits") or {}
    budgets = normalize_budget_state(task.budgets_json, limits)
    elapsed = int(budgets["tranche_usage"].get("elapsed_active_ms") or 0) + increment
    lifetime_elapsed = int(budgets["lifetime_usage"].get("elapsed_active_ms") or 0) + increment
    budgets["tranche_usage"]["elapsed_active_ms"] = elapsed
    budgets["lifetime_usage"]["elapsed_active_ms"] = lifetime_elapsed
    exhausted = exhausted_dimensions(budgets)
    if exhausted and not isinstance(budgets.get("boundary"), dict):
        budgets["boundary"] = {
            "status": "requested",
            "dimensions": exhausted,
            "tranche_index": budgets["tranche_index"],
            "requested_at": now.isoformat(),
        }
    replace_jsonb_field(task, "budgets_json", budgets)
    maximum = int((budgets.get("tranche_limits") or {}).get("elapsed_active_ms") or 3_600_000)
    task.version += 1
    await _append_event(
        session,
        task,
        "task.budget_updated",
        agent_run_id=task.active_run_id,
        payload={
            "elapsed_active_ms": elapsed,
            "lifetime_elapsed_active_ms": lifetime_elapsed,
            "max_active_runtime_ms": maximum,
            "exhausted_dimensions": exhausted,
        },
    )
    return increment


async def active_runtime_budget_exhausted(task_id: str) -> bool:
    task = await get_task(task_id)
    if task is None:
        return True
    limits = (task.config_json or {}).get("limits") or {}
    state = normalize_budget_state(task.budgets_json, limits)
    return "elapsed_active_ms" in exhausted_dimensions(state)


async def budget_boundary(task_id: str) -> Optional[Dict[str, Any]]:
    task = await get_task(task_id)
    if task is None:
        return None
    state = normalize_budget_state(task.budgets_json, (task.config_json or {}).get("limits") or {})
    boundary = state.get("boundary")
    return dict(boundary) if isinstance(boundary, dict) and boundary.get("status") == "requested" else None


async def attach_run(task_id: str, run: AgentRun, *, parent_run_id: Optional[str] = None) -> AgentRun:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            stored_run = await session.get(AgentRun, run.id)
            if stored_run is None:
                raise AgentTaskConflict("task_run_missing", "Agent run does not exist")

            # The API eagerly prepares a run after the start command while a
            # worker may claim the task at the same time. Serialize attachment
            # on the task row and converge both callers on the run that won.
            # Query by task_id as well as active_run_id so this also repairs a
            # stale task pointer left by an interrupted attachment.
            existing_active = (await session.execute(
                select(AgentRun).where(
                    AgentRun.task_id == task.id,
                    AgentRun.status.in_([
                        AgentRunStatus.RUNNING.value,
                        AgentRunStatus.AWAITING_HUMAN.value,
                    ]),
                ).order_by(AgentRun.task_attempt.desc(), AgentRun.started_at.desc()).limit(1)
            )).scalar_one_or_none()
            if existing_active is not None and existing_active.id == stored_run.id:
                task.active_run_id = existing_active.id
                task.primary_run_id = task.primary_run_id or existing_active.id
                task.latest_run_attempt = max(task.latest_run_attempt, existing_active.task_attempt)
                return existing_active
            if existing_active is not None and existing_active.id != stored_run.id:
                stored_run.status = AgentRunStatus.CANCELLED.value
                stored_run.completed_at = utc_now()
                stored_run.error_json = {
                    "code": "concurrent_task_run_superseded",
                    "retryable": False,
                    "active_run_id": existing_active.id,
                }
                task.active_run_id = existing_active.id
                task.primary_run_id = task.primary_run_id or existing_active.id
                task.latest_run_attempt = max(task.latest_run_attempt, existing_active.task_attempt)
                return existing_active

            next_attempt = task.latest_run_attempt + 1
            stored_run.task_id = task.id
            stored_run.parent_run_id = parent_run_id
            stored_run.task_attempt = next_attempt
            task.primary_run_id = task.primary_run_id or stored_run.id
            task.active_run_id = stored_run.id
            task.latest_run_attempt = next_attempt
            task.version += 1
            await _append_event(session, task, "task.run_attached", agent_run_id=stored_run.id, payload={"attempt": next_attempt})
        await session.refresh(task)
        await session.refresh(stored_run)
        return stored_run


async def persist_plan(
    task_id: str,
    proposal: DeepResearchPlanProposal,
    *,
    agent_run_id: str,
    reason: str,
    planner_visit: int,
    idempotency_key: Optional[str] = None,
) -> tuple[AgentTaskPlanRevision, list[AgentTaskTodo]]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            if idempotency_key:
                revisions = list((await session.execute(
                    select(AgentTaskPlanRevision).where(AgentTaskPlanRevision.task_id == task_id)
                )).scalars().all())
                existing_revision = next(
                    (
                        value for value in revisions
                        if str((value.provenance_json or {}).get("runtime_idempotency_key") or "") == idempotency_key
                    ),
                    None,
                )
                if existing_revision is not None:
                    existing_todos = list((await session.execute(
                        select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id)
                    )).scalars().all())
                    return existing_revision, existing_todos
            latest = int((await session.execute(select(func.coalesce(func.max(AgentTaskPlanRevision.revision), 0)).where(AgentTaskPlanRevision.task_id == task_id))).scalar_one())
            revision_number = latest + 1
            limits = (task.config_json or {}).get("limits") or {}
            run_revision_count = int((await session.execute(
                select(func.count(AgentTaskPlanRevision.id)).where(
                    AgentTaskPlanRevision.task_id == task_id,
                    AgentTaskPlanRevision.agent_run_id == agent_run_id,
                )
            )).scalar_one())
            if reason != "course_correction" and run_revision_count >= int(limits.get("max_plan_revisions", 8)):
                raise AgentTaskConflict("plan_revision_budget_exhausted", "Plan revision limit reached")
            enabled_profiles = set((task.config_json or {}).get("enabled_profiles") or [])
            for todo in proposal.todos:
                if todo.profile_id.value == "evidence_critic":
                    raise AgentTaskConflict("plan_profile_not_schedulable", "Evidence critic is reserved for final review")
                if todo.profile_id.value not in enabled_profiles:
                    raise AgentTaskConflict("plan_profile_not_allowed", f"Profile {todo.profile_id.value} is disabled")
            revision = AgentTaskPlanRevision(
                task_id=task.id,
                agent_run_id=agent_run_id,
                revision=revision_number,
                planner_visit=planner_visit,
                reason=reason,
                objective=proposal.objective,
                completion_criteria_json=proposal.success_criteria,
                ordered_todo_ids_json=[todo.id for todo in proposal.todos],
                plan_json=proposal.model_dump(mode="json"),
                provenance_json={
                    "config_hash": canonical_hash(task.config_json),
                    **({"runtime_idempotency_key": idempotency_key} if idempotency_key else {}),
                },
                content_hash=proposal.content_hash(),
            )
            session.add(revision)
            existing = {
                todo.id: todo
                for todo in (await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task.id))).scalars().all()
            }
            persisted: list[AgentTaskTodo] = []
            proposed_ids = {value.id for value in proposal.todos}
            for value in proposal.todos:
                current = existing.get(value.id)
                if current and current.status == "completed":
                    persisted.append(current)
                    continue
                if current is None:
                    current = AgentTaskTodo(
                        id=value.id,
                        task_id=task.id,
                        title=value.title,
                        description=value.description,
                        completion_criteria=value.completion_criteria,
                        priority=value.priority,
                        required=value.required,
                        dependency_ids_json=value.dependency_ids,
                        profile_id=value.profile_id.value,
                        max_attempts=int(limits.get("max_attempts_per_todo", 2)),
                        created_revision=revision_number,
                        updated_revision=revision_number,
                    )
                    session.add(current)
                else:
                    current.title = value.title
                    current.description = value.description
                    current.completion_criteria = value.completion_criteria
                    current.priority = value.priority
                    current.required = value.required
                    replace_jsonb_field(current, "dependency_ids_json", value.dependency_ids)
                    current.profile_id = value.profile_id.value
                    current.updated_revision = revision_number
                    current.version += 1
                    current.updated_at = utc_now()
                persisted.append(current)
            for omitted in existing.values():
                if omitted.id in proposed_ids or omitted.status == "completed":
                    continue
                if omitted.status == "running":
                    raise AgentTaskConflict("plan_supersedes_running_todo", "A plan revision cannot supersede running work")
                omitted.status = "skipped"
                omitted.required = False
                omitted.terminal_reason = f"superseded_by_plan_revision:{revision_number}"
                omitted.updated_revision = revision_number
                omitted.version += 1
                omitted.updated_at = utc_now()
                persisted.append(omitted)
            await session.flush()
            task.total_todos = len(persisted)
            task.current_phase = "planned"
            task.version += 1
            await _append_event(
                session, task, "plan.superseded" if reason == "course_correction" else "plan.revised",
                agent_run_id=agent_run_id,
                payload={"revision": revision_number, "todo_count": len(persisted), "content_hash": revision.content_hash, "reason": reason},
            )
        await session.refresh(revision)
        return revision, persisted


async def schedule_ready_todos(task_id: str, *, limit: int) -> list[AgentTaskTodo]:
    """Atomically project dependency-ready todos and claim a bounded batch."""
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            todos = list((await session.execute(
                select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id).with_for_update()
            )).scalars().all())
            by_id = {todo.id: todo for todo in todos}
            terminal_success = {todo.id for todo in todos if todo.status in {"completed", "skipped"}}
            changed: list[AgentTaskTodo] = []
            for todo in todos:
                dependencies = [str(value) for value in (todo.dependency_ids_json or [])]
                if todo.status == "pending" and all(value in terminal_success for value in dependencies):
                    todo.status = "ready"
                    todo.version += 1
                    todo.updated_at = utc_now()
                    changed.append(todo)
                elif todo.status == "pending" and any(by_id.get(value) and by_id[value].status in {"failed", "cancelled", "blocked"} for value in dependencies):
                    todo.status = "blocked"
                    todo.terminal_reason = "dependency_failed"
                    todo.version += 1
                    todo.updated_at = utc_now()
                    changed.append(todo)
            # A scheduler replay can occur after todos were atomically claimed
            # but before the graph checkpoint committed the dispatch result.
            # Re-emit only attempts that have not started a subagent execution.
            claimed = [todo for todo in todos if todo.status == "running" and not todo.current_subagent_run_id]
            ready = sorted(
                [*claimed, *(todo for todo in todos if todo.status == "ready")],
                key=lambda value: (-value.priority, value.created_at, value.id),
            )[:max(1, limit)]
            for todo in ready:
                if todo.status == "ready":
                    todo.status = "running"
                    todo.attempt += 1
                todo.progress = max(1, todo.progress)
                todo.version += 1
                todo.updated_at = utc_now()
                await _append_event(session, task, "todo.started", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"attempt": todo.attempt, "profile_id": todo.profile_id})
            completed = sum(1 for todo in todos if todo.status == "completed")
            task.completed_todos = completed
            task.total_todos = len(todos)
            task.progress = int((completed * 100) / len(todos)) if todos else 0
            task.current_phase = "dispatching" if ready else "controlling"
            task.version += 1
            for todo in changed:
                await _append_event(session, task, f"todo.{todo.status}", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"version": todo.version})
        for todo in ready:
            await session.refresh(todo)
        return ready


async def block_todos(task_id: str, todo_ids: Iterable[str], *, reason: str) -> list[AgentTaskTodo]:
    ids = {str(value) for value in todo_ids if value}
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            rows = list((await session.execute(select(AgentTaskTodo).where(
                AgentTaskTodo.task_id == task_id,
                AgentTaskTodo.id.in_(ids),
            ).with_for_update())).scalars().all())
            for todo in rows:
                if todo.status in {"pending", "ready", "running"}:
                    todo.status = "blocked" if todo.required else "skipped"
                    todo.terminal_reason = reason
                    todo.current_subagent_run_id = None
                    todo.version += 1
                    todo.updated_at = utc_now()
                    await _append_event(session, task, f"todo.{todo.status}", agent_run_id=task.active_run_id, todo_id=todo.id, payload={"reason": reason})
            task.version += 1
        return rows


async def record_subagent_started(
    *, task_id: str, agent_run_id: str, todo_id: str, profile_id: str,
    plan_revision: int, attempt: int, timeout_ms: int, tool_policy_hash: str,
) -> tuple[AgentTaskSubagentRun, bool]:
    execution_key = canonical_hash({
        "task_id": task_id, "todo_id": todo_id, "profile_id": profile_id,
        "plan_revision": plan_revision, "attempt": attempt,
    })
    async with async_session_maker() as session:
        async with session.begin():
            existing = (await session.execute(select(AgentTaskSubagentRun).where(AgentTaskSubagentRun.execution_key == execution_key))).scalar_one_or_none()
            if existing:
                return existing, True
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            row = AgentTaskSubagentRun(
                task_id=task_id,
                agent_run_id=agent_run_id,
                todo_id=todo_id,
                execution_key=execution_key,
                profile_id=profile_id,
                plan_revision=plan_revision,
                attempt=attempt,
                status="running",
                tool_policy_hash=tool_policy_hash,
                timeout_ms=timeout_ms,
                started_at=utc_now(),
            )
            session.add(row)
            await session.flush()
            todo = await session.get(AgentTaskTodo, (task_id, todo_id))
            if todo is not None:
                todo.current_subagent_run_id = row.id
            budgets = normalize_budget_state(task.budgets_json, (task.config_json or {}).get("limits") or {})
            budgets["lifetime_usage"]["subagent_attempts"] = int(budgets["lifetime_usage"].get("subagent_attempts") or 0) + 1
            replace_jsonb_field(task, "budgets_json", budgets)
            await _append_event(session, task, "subagent.started", agent_run_id=agent_run_id, todo_id=todo_id, subagent_run_id=row.id, payload={"profile_id": profile_id, "attempt": attempt})
        await session.refresh(row)
        return row, False


async def record_subagent_result(
    *, task_id: str, todo_id: str, subagent_run_id: str, status: str,
    summary: str, artifact_ids: list[str], usage: Dict[str, Any], error: Optional[Dict[str, Any]], retryable: bool,
) -> AgentTaskTodo:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            todo = (await session.execute(select(AgentTaskTodo).where(
                AgentTaskTodo.task_id == task_id, AgentTaskTodo.id == todo_id
            ).with_for_update())).scalar_one()
            subagent = await session.get(AgentTaskSubagentRun, subagent_run_id)
            if subagent is None or subagent.task_id != task_id:
                raise AgentTaskConflict("subagent_run_mismatch", "Subagent execution does not belong to task")
            if subagent.completed_at is not None and subagent.status in {"completed", "failed", "timed_out", "cancelled"}:
                return todo
            subagent.status = status
            subagent.completed_at = utc_now()
            replace_jsonb_field(subagent, "usage_json", usage)
            replace_jsonb_field(subagent, "output_artifact_ids_json", artifact_ids)
            if error is not None:
                replace_jsonb_field(subagent, "error_json", error)
            todo.result_summary = summary[:12_000]
            replace_jsonb_field(todo, "artifact_ids_json", list(dict.fromkeys([*(todo.artifact_ids_json or []), *artifact_ids])))
            if status == "completed":
                todo.status = "completed"
                todo.progress = 100
            elif status == "cancelled":
                todo.status = "cancelled"
                todo.terminal_reason = "task_cancelled"
            elif retryable and todo.attempt < todo.max_attempts:
                todo.status = "ready"
                todo.terminal_reason = None
            else:
                todo.status = "failed"
                todo.terminal_reason = str((error or {}).get("code") or status)
            todo.current_subagent_run_id = None
            todo.version += 1
            todo.updated_at = utc_now()
            todos = list((await session.execute(select(AgentTaskTodo).where(AgentTaskTodo.task_id == task_id))).scalars().all())
            task.completed_todos = sum(1 for value in todos if value.status == "completed")
            task.total_todos = len(todos)
            task.progress = int((task.completed_todos * 100) / len(todos)) if todos else 0
            task.version += 1
            await _append_event(session, task, f"subagent.{status}", agent_run_id=subagent.agent_run_id, todo_id=todo.id, subagent_run_id=subagent.id, payload={"todo_status": todo.status, "retryable": retryable, "artifact_ids": artifact_ids})
            await _append_event(session, task, f"todo.{todo.status}", agent_run_id=subagent.agent_run_id, todo_id=todo.id, payload={"attempt": todo.attempt, "progress": todo.progress})
        await session.refresh(todo)
        return todo


async def register_artifact(metadata: AgentTaskArtifact) -> tuple[AgentTaskArtifact, bool]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == metadata.task_id).with_for_update())).scalar_one()
            duplicate = (await session.execute(select(AgentTaskArtifact).where(
                AgentTaskArtifact.agent_run_id == metadata.agent_run_id,
                AgentTaskArtifact.ownership_key == metadata.ownership_key,
                AgentTaskArtifact.sha256 == metadata.sha256,
                AgentTaskArtifact.kind == metadata.kind,
                AgentTaskArtifact.validity == "valid",
            ))).scalar_one_or_none()
            if duplicate:
                return duplicate, True
            limits = (task.config_json or {}).get("limits") or {}
            artifact_count = int((await session.execute(select(func.count(AgentTaskArtifact.id)).where(
                AgentTaskArtifact.task_id == task.id,
                AgentTaskArtifact.validity != "deleted",
            ))).scalar_one())
            budgets = normalize_budget_state(task.budgets_json, limits)
            if artifact_count >= int(limits.get("max_artifacts", 200)):
                raise AgentTaskConflict("artifact_count_budget_exhausted", "Task artifact count limit reached")
            if metadata.byte_size > int(limits.get("max_single_artifact_bytes", 10_485_760)):
                raise AgentTaskConflict("artifact_size_budget_exhausted", "Task artifact exceeds its configured size limit")
            if int(budgets["lifetime_usage"].get("artifact_bytes") or 0) + metadata.byte_size > int(limits.get("max_artifact_bytes", 104_857_600)):
                raise AgentTaskConflict("artifact_bytes_budget_exhausted", "Task artifact byte budget exhausted")
            session.add(metadata)
            await session.flush()
            budgets["lifetime_usage"]["artifact_bytes"] = int(budgets["lifetime_usage"].get("artifact_bytes") or 0) + metadata.byte_size
            replace_jsonb_field(task, "budgets_json", budgets)
            await _append_event(session, task, "artifact.created", agent_run_id=metadata.agent_run_id, todo_id=metadata.todo_id, subagent_run_id=metadata.subagent_run_id, artifact_id=metadata.id, payload={"kind": metadata.kind, "byte_size": metadata.byte_size, "sha256": metadata.sha256})
        await session.refresh(metadata)
        return metadata, False


async def complete_task(task_id: str, *, status: str, reason: Optional[str] = None, final_artifact_id: Optional[str] = None) -> AgentTask:
    if status not in TERMINAL_TASK_STATUSES:
        raise ValueError("task completion requires a terminal status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = status
            task.current_phase = status
            task.terminal_reason = reason
            task.completed_at = utc_now()
            task.expires_at = None
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, f"task.{status}", agent_run_id=task.active_run_id, artifact_id=final_artifact_id, payload={"reason": reason, "version": task.version})
        await session.refresh(task)
        return task


async def finalize_task_run(
    task_id: str,
    run_id: str,
    *,
    run_status: str,
    task_status: str,
    metrics: Dict[str, Any],
    error: Optional[Dict[str, Any]],
    debug_trace: Dict[str, Any],
    terminal_reason: Optional[str],
    terminal_event: Any,
    final_artifact_id: Optional[str] = None,
    completed_at: Any = None,
) -> AgentTask:
    """Atomically commit terminal run, task, and both product journals."""
    if task_status not in TERMINAL_TASK_STATUSES:
        raise ValueError("task finalization requires a terminal task status")
    if run_status not in {
        AgentRunStatus.COMPLETED.value,
        AgentRunStatus.FAILED.value,
        AgentRunStatus.CANCELLED.value,
    }:
        raise ValueError("task finalization requires a terminal run status")
    if not bool(getattr(terminal_event, "terminal", False)):
        raise ValueError("task finalization requires a terminal run event")

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one()
            run = (await session.execute(
                select(AgentRun).where(AgentRun.id == run_id, AgentRun.task_id == task_id).with_for_update()
            )).scalar_one()
            existing_terminal = (await session.execute(
                select(AgentRunEvent).where(
                    AgentRunEvent.agent_run_id == run_id,
                    AgentRunEvent.terminal.is_(True),
                )
            )).scalar_one_or_none()
            if existing_terminal is not None and existing_terminal.event_id != terminal_event.event_id:
                raise AgentTaskConflict("task_terminal_conflict", "A different terminal run event already exists")
            if task.status in TERMINAL_TASK_STATUSES and task.status != task_status:
                raise AgentTaskConflict("task_terminal_conflict", "The task already has a different terminal status")
            if run.status in TERMINAL_TASK_RUN_STATUSES and run.status != run_status:
                raise AgentTaskConflict("task_terminal_conflict", "The run already has a different terminal status")
            if (
                existing_terminal is not None
                and task.status == task_status
                and run.status == run_status
            ):
                return task

            completed_at = completed_at or utc_now()
            run.status = run_status
            run.completed_at = completed_at
            replace_jsonb_field(run, "metrics_json", metrics)
            replace_jsonb_field(run, "error_json", error or {})
            replace_jsonb_field(run, "debug_trace_json", debug_trace)
            run_metadata = dict(run.run_metadata_json or {})
            projection = dict(run_metadata.get("projection") or {})
            if projection.get("runtime_result"):
                projection.update({
                    "status": "applied",
                    "reconciliation_status": "projected",
                    "final_artifact_id": final_artifact_id,
                })
                run_metadata["projection"] = projection
                replace_jsonb_field(run, "run_metadata_json", run_metadata)

            task.status = task_status
            task.current_phase = task_status
            task.terminal_reason = terminal_reason
            task.completed_at = completed_at
            task.expires_at = None
            task.lease_owner = None
            task.lease_expires_at = None
            task.heartbeat_at = None
            task.version += 1

            pending_cancel_commands = list((await session.execute(
                select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task_id,
                    AgentTaskCommand.action == "cancel",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update()
            )).scalars().all())
            for command in pending_cancel_commands:
                command.status = "completed"
                command.result_json = {
                    "status": task_status,
                    "task_id": task_id,
                    "run_id": run_id,
                    "runtime_confirmation": "confirmed" if task_status == AgentTaskStatus.CANCELLED.value else "terminal_before_cancellation",
                }
                command.completed_at = completed_at

            if existing_terminal is None:
                latest_sequence = (await session.execute(
                    select(func.coalesce(func.max(AgentRunEvent.sequence), 0)).where(
                        AgentRunEvent.agent_run_id == run_id
                    )
                )).scalar_one()
                source_metadata = dict(terminal_event.source_metadata or {})
                source_metadata.setdefault("source_sequence", int(terminal_event.sequence))
                session.add(AgentRunEvent(
                    agent_run_id=run_id,
                    event_id=str(terminal_event.event_id),
                    sequence=int(latest_sequence) + 1,
                    attempt=int(terminal_event.attempt),
                    kind=str(terminal_event.kind),
                    occurred_at=parse_datetime_utc(terminal_event.occurred_at),
                    payload_json=dict(terminal_event.payload or {}),
                    trace_id=terminal_event.trace_id,
                    terminal=True,
                    source_metadata_json=source_metadata,
                ))
            await _append_event(
                session,
                task,
                f"task.{task_status}",
                agent_run_id=run_id,
                artifact_id=final_artifact_id,
                payload={"reason": terminal_reason, "version": task.version},
            )
        await session.refresh(task)
        return task


async def set_task_runtime_status(task_id: str, status: str, *, phase: Optional[str] = None, reason: Optional[str] = None) -> AgentTask:
    if status not in {value.value for value in AgentTaskStatus}:
        raise ValueError("unknown task status")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            task.status = status
            task.current_phase = phase or status
            task.terminal_reason = reason
            task.updated_at = utc_now()
            task.version += 1
            if status == AgentTaskStatus.PAUSED.value:
                task.paused_at = utc_now()
                task.expires_at = utc_now() + timedelta(days=7)
            elif status == AgentTaskStatus.AWAITING_APPROVAL.value:
                task.expires_at = utc_now() + timedelta(days=7)
            if status != AgentTaskStatus.RUNNING.value:
                task.lease_owner = None
                task.lease_expires_at = None
            await _append_event(session, task, f"task.{status}", agent_run_id=task.active_run_id, payload={"phase": task.current_phase, "reason": reason, "version": task.version})
        await session.refresh(task)
        return task


async def mark_runtime_projection_recovery_required(
    task_id: str,
    run_id: str,
    *,
    projection: Dict[str, Any],
    error: Dict[str, Any],
) -> AgentTask:
    """Atomically expose a runtime-complete/product-unprojected task state."""

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(
                AgentTask.id == task_id,
            ).with_for_update())).scalar_one()
            run = (await session.execute(select(AgentRun).where(
                AgentRun.id == run_id,
                AgentRun.task_id == task_id,
            ).with_for_update())).scalar_one()
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            projection_value = {
                **dict(projection),
                "status": "pending",
                "reconciliation_status": "failed",
                "projection_error": dict(error),
            }
            metadata = dict(run.run_metadata_json or {})
            metadata["projection"] = projection_value
            replace_jsonb_field(run, "run_metadata_json", metadata)
            run.status = AgentRunStatus.RECOVERY_REQUIRED.value
            run.error_json = dict(error)
            task.status = AgentTaskStatus.RECOVERY_REQUIRED.value
            task.current_phase = "runtime_projection_recovery_required"
            task.terminal_reason = "runtime_task_projection_conflict"
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = utc_now()
            task.version += 1
            await _append_event(
                session,
                task,
                "task.runtime_projection_failed",
                agent_run_id=run_id,
                payload={
                    "code": error.get("code"),
                    "retryable": bool(error.get("retryable")),
                    "delta_event_id": projection_value.get("delta_event_id"),
                    "operation_id": projection_value.get("operation_id"),
                },
            )
        await session.refresh(task)
        return task


async def finalize_reconciled_runtime_task(
    task_id: str,
    run_id: str,
    *,
    delta_event_id: str,
    payload_sha256: str,
    runtime_status: str,
    result: Dict[str, Any],
    final_artifact_id: Optional[str] = None,
) -> AgentTask:
    """Finalize only after proving the task delta committed with matching content."""

    run_status = {
        "completed": AgentRunStatus.COMPLETED.value,
        "failed": AgentRunStatus.FAILED.value,
        "cancelled": AgentRunStatus.CANCELLED.value,
        "canceled": AgentRunStatus.CANCELLED.value,
    }.get(runtime_status)
    task_status = {
        AgentRunStatus.COMPLETED.value: AgentTaskStatus.COMPLETED.value,
        AgentRunStatus.FAILED.value: AgentTaskStatus.FAILED.value,
        AgentRunStatus.CANCELLED.value: AgentTaskStatus.CANCELLED.value,
    }.get(run_status or "")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            run = (await session.execute(select(AgentRun).where(
                AgentRun.id == run_id, AgentRun.task_id == task_id,
            ).with_for_update())).scalar_one()
            ledger = (await session.execute(select(AgentTaskRuntimeDelta).where(
                AgentTaskRuntimeDelta.agent_run_id == run_id,
                AgentTaskRuntimeDelta.event_id == delta_event_id,
            ).with_for_update())).scalar_one_or_none()
            if ledger is None or ledger.payload_sha256 != payload_sha256:
                raise AgentTaskConflict("runtime_delta_not_applied", "Runtime delta ledger does not match reconciliation")
            metadata = dict(run.run_metadata_json or {})
            projection = dict(metadata.get("projection") or {})
            if (
                projection.get("status") == "applied"
                and projection.get("reconciliation_status") == "projected"
                and task.status in TERMINAL_TASK_STATUSES
                and run.status in TERMINAL_TASK_RUN_STATUSES
            ):
                return task
            projection.update({
                "status": "applied",
                "reconciliation_status": "projected",
                "delta_event_id": delta_event_id,
                "final_artifact_id": final_artifact_id,
            })
            projection.pop("projection_error", None)
            metadata["projection"] = projection
            replace_jsonb_field(run, "run_metadata_json", metadata)
            if run_status is not None and task_status is not None:
                completed_at = utc_now()
                run.status = run_status
                run.completed_at = completed_at
                replace_jsonb_field(run, "error_json", dict(result.get("agent_error") or result.get("error") or {}))
                task.status = task_status
                task.current_phase = task_status
                task.terminal_reason = str((result.get("agent_error") or {}).get("code") or runtime_status)
                task.completed_at = completed_at
                task.expires_at = None
                task.lease_owner = None
                task.lease_expires_at = None
                task.version += 1
                terminal_kind = "run.cancelled" if run_status == "cancelled" else "run.failed" if run_status == "failed" else "run.completed"
                existing_terminal = (await session.execute(select(AgentRunEvent).where(
                    AgentRunEvent.agent_run_id == run_id,
                    AgentRunEvent.terminal.is_(True),
                ))).scalar_one_or_none()
                if existing_terminal is None:
                    sequence = int((await session.execute(select(func.coalesce(func.max(AgentRunEvent.sequence), 0)).where(
                        AgentRunEvent.agent_run_id == run_id,
                    ))).scalar_one()) + 1
                    session.add(AgentRunEvent(
                        agent_run_id=run_id, event_id=delta_event_id, sequence=sequence,
                        attempt=max(1, int(run.task_attempt or 1)), kind=terminal_kind,
                        occurred_at=completed_at, payload_json=dict(result), terminal=True,
                        source_metadata_json={"source": "runtime_delta_reconciliation"},
                    ))
                await _append_event(
                    session, task, f"task.{task_status}", agent_run_id=run_id,
                    artifact_id=final_artifact_id,
                    payload={"reason": task.terminal_reason, "version": task.version, "reconciled": True},
                )
            else:
                await _append_event(
                    session, task, "task.runtime_projection_reconciled", agent_run_id=run_id,
                    payload={"delta_event_id": delta_event_id},
                )
        await session.refresh(task)
        return task


async def respond_to_result_review(
    task_id: str,
    *,
    run_id: str,
    interrupt_id: str,
    expected_version: int,
    decision: str,
    followup_input: Optional[str],
    idempotency_key: str,
) -> tuple[AgentTask, bool]:
    """Resolve a product-owned incomplete-result review atomically.

    This deliberately does not call a runtime resume operation. A retry closes
    the provisional run and queues a linked product run.
    """

    if decision not in {"accept", "retry_with_input"}:
        raise AgentTaskConflict("result_review_decision_invalid", "Unsupported result review decision")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one_or_none()
            if task is None:
                raise AgentTaskConflict("task_not_found", "Agent task not found")
            run = (await session.execute(
                select(AgentRun).where(AgentRun.id == run_id, AgentRun.task_id == task_id).with_for_update()
            )).scalar_one_or_none()
            if run is None:
                raise AgentTaskConflict("task_run_missing", "Agent task run not found")
            pending = dict(run.pending_interrupt_json or {})
            previous = pending.get("decision") if isinstance(pending.get("decision"), dict) else {}
            if pending.get("status") != "pending":
                if previous.get("idempotency_key") == idempotency_key:
                    return task, True
                raise AgentTaskConflict("result_review_already_resolved", "Result review is already resolved", current_version=task.version)
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            if task.status != AgentTaskStatus.AWAITING_APPROVAL.value or task.active_run_id != run.id:
                raise AgentTaskConflict("result_review_not_pending", "Task is not awaiting this result review", current_version=task.version)
            if pending.get("response_operation") != "task.result_review.respond" or pending.get("interrupt_id") != interrupt_id:
                raise AgentTaskConflict("result_review_identity_mismatch", "Result review identity does not match", current_version=task.version)

            now = utc_now()
            pending["status"] = "resolved"
            pending["resolved_at"] = now.isoformat()
            pending["decision"] = {
                "action": decision,
                "followup_input": followup_input,
                "idempotency_key": idempotency_key,
            }
            replace_jsonb_field(run, "pending_interrupt_json", pending)
            run.status = AgentRunStatus.COMPLETED.value
            run.completed_at = now
            if isinstance(run.debug_trace_json, dict):
                replace_jsonb_field(run, "debug_trace_json", append_runtime_event_to_debug_payload(
                    run.debug_trace_json,
                    "task.result_review.responded",
                    attributes={
                        "askpdf.task.id": task.id,
                        "askpdf.run.id": run.id,
                        "askpdf.result.outcome": "completed_with_warnings",
                    },
                    output_data={
                        "decision": decision,
                        "linked_retry": decision == "retry_with_input",
                        "provisional_artifact_id": pending.get("provisional_artifact_id"),
                    },
                    run_status=AgentRunStatus.COMPLETED.value,
                    completed_at=now,
                ))
            existing_terminal = (await session.execute(
                select(AgentRunEvent).where(
                    AgentRunEvent.agent_run_id == run.id,
                    AgentRunEvent.terminal.is_(True),
                )
            )).scalar_one_or_none()
            if existing_terminal is None:
                latest_sequence = int((await session.execute(
                    select(func.coalesce(func.max(AgentRunEvent.sequence), 0)).where(
                        AgentRunEvent.agent_run_id == run.id
                    )
                )).scalar_one())
                session.add(AgentRunEvent(
                    agent_run_id=run.id,
                    event_id=f"result-review:{run.id}:{interrupt_id}:completed",
                    sequence=latest_sequence + 1,
                    attempt=max(1, int(run.task_attempt or 1)),
                    kind="run.completed",
                    occurred_at=now,
                    payload_json={
                        "status": "completed",
                        "result_outcome": "completed_with_warnings",
                        "review_decision": decision,
                        "provisional_artifact_id": pending.get("provisional_artifact_id"),
                    },
                    terminal=True,
                    source_metadata_json={"framework": "product", "source_event": "task.result_review.respond"},
                ))

            if decision == "accept":
                active_corrections = list((await session.execute(select(AgentTaskCommand).where(
                    AgentTaskCommand.task_id == task.id,
                    AgentTaskCommand.action == "steer",
                    AgentTaskCommand.status == "accepted",
                ).with_for_update())).scalars().all())
                accepted_unresolved: list[str] = []
                for command in active_corrections:
                    command_result = dict(command.result_json or {})
                    correction = dict(command_result.get("correction") or {})
                    correction_id = str(correction.get("correction_id") or correction.get("id") or "")
                    correction.update({
                        "status": "accepted_unresolved",
                        "accepted_unresolved_at": now.isoformat(),
                        "review_action_version": task.version + 1,
                    })
                    command_result.update({
                        "correction": correction,
                        "delivery_state": "accepted_unresolved",
                        "review_interrupt_id": interrupt_id,
                    })
                    replace_jsonb_field(command, "result_json", command_result)
                    command.status = "completed"
                    command.completed_at = now
                    command.result_version = task.version + 1
                    accepted_unresolved.append(correction_id)
                if accepted_unresolved:
                    await _append_event(
                        session, task, "task.course_correction_accepted_unresolved",
                        agent_run_id=run.id,
                        payload={
                            "correction_ids": accepted_unresolved,
                            "interrupt_id": interrupt_id,
                            "action_version": task.version + 1,
                        },
                    )
                task.status = AgentTaskStatus.COMPLETED.value
                task.current_phase = AgentTaskStatus.COMPLETED.value
                task.terminal_reason = "completed_with_warnings"
                task.completed_at = now
                task.expires_at = None
                event_kind = "task.result_review_accepted"
            else:
                config = dict(task.config_json or {})
                review_context = list(config.get("result_review_context") or [])
                review_context.append({
                    "source_run_id": run.id,
                    "source_artifact_id": pending.get("provisional_artifact_id"),
                    "followup_input": followup_input,
                    "review_round": pending.get("review_round"),
                })
                config["result_review_context"] = review_context[-5:]
                replace_jsonb_field(task, "config_json", config)
                todos = list((await session.execute(
                    select(AgentTaskTodo).where(
                        AgentTaskTodo.task_id == task.id,
                        AgentTaskTodo.status.in_(["failed", "blocked", "cancelled"]),
                    ).with_for_update()
                )).scalars().all())
                for todo in todos:
                    todo.status = "pending"
                    todo.current_subagent_run_id = None
                    todo.terminal_reason = None
                    todo.progress = 0
                    todo.version += 1
                    todo.updated_at = now
                task.status = AgentTaskStatus.QUEUED.value
                task.current_phase = "result_review_retry_queued"
                task.terminal_reason = None
                task.queued_at = now
                task.completed_at = None
                task.expires_at = now + timedelta(hours=24)
                event_kind = "task.result_review_retry_queued"
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(
                session, task, event_kind, agent_run_id=run.id,
                artifact_id=pending.get("provisional_artifact_id"),
                payload={
                    "interrupt_id": interrupt_id,
                    "decision": decision,
                    "version": task.version,
                    "linked_retry": decision == "retry_with_input",
                },
            )
        await session.refresh(task)
        return task, False


async def respond_to_budget_review(
    task_id: str,
    *,
    run_id: str,
    interrupt_id: str,
    expected_version: int,
    decision: str,
    guidance: Optional[str],
    idempotency_key: str,
) -> tuple[AgentTask, bool, bool]:
    """Resolve a repeatable budget boundary for checkpoint or linked continuation."""

    if decision not in {"continue", "accept_partial", "steer"}:
        raise AgentTaskConflict("budget_review_decision_invalid", "Unsupported budget review decision")
    if decision == "steer" and not str(guidance or "").strip():
        raise AgentTaskConflict("budget_review_guidance_required", "Steering guidance is required")
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            run = (await session.execute(select(AgentRun).where(AgentRun.id == run_id, AgentRun.task_id == task_id).with_for_update())).scalar_one_or_none()
            if task is None or run is None:
                raise AgentTaskConflict("task_run_missing", "Agent task run not found")
            pending = dict(run.pending_interrupt_json or {})
            previous = pending.get("decision") if isinstance(pending.get("decision"), dict) else {}
            if pending.get("status") != "pending":
                if previous.get("idempotency_key") == idempotency_key:
                    return task, True, continuation_is_linked(run)
                raise AgentTaskConflict("budget_review_already_resolved", "Budget review is already resolved", current_version=task.version)
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            if task.status != AgentTaskStatus.AWAITING_APPROVAL.value or task.active_run_id != run.id:
                raise AgentTaskConflict("budget_review_not_pending", "Task is not awaiting this budget review", current_version=task.version)
            if pending.get("response_operation") != "task.budget_review.respond" or pending.get("interrupt_id") != interrupt_id:
                raise AgentTaskConflict("budget_review_identity_mismatch", "Budget review identity does not match", current_version=task.version)

            now = utc_now()
            pending["status"] = "resolved"
            pending["resolved_at"] = now.isoformat()
            linked_run = continuation_is_linked(run)
            pending["decision"] = {
                "action": decision,
                "idempotency_key": idempotency_key,
                "guidance_delivery": "course_correction_command" if decision == "steer" else None,
            }
            replace_jsonb_field(run, "pending_interrupt_json", pending)
            if decision == "accept_partial":
                if not str(pending.get("provisional_answer") or "").strip():
                    raise AgentTaskConflict("budget_partial_answer_unavailable", "No provisional answer is available to accept")
                run.status = AgentRunStatus.COMPLETED.value
                run.completed_at = now
                task.status = AgentTaskStatus.COMPLETED.value
                task.current_phase = AgentTaskStatus.COMPLETED.value
                task.terminal_reason = "completed_with_warnings"
                task.completed_at = now
                task.expires_at = None
                event_kind = "task.budget_review_partial_accepted"
            else:
                budget = normalize_budget_state(task.budgets_json, (task.config_json or {}).get("limits") or {})
                replace_jsonb_field(task, "budgets_json", reset_tranche(budget))
                if guidance:
                    correction_id = str(uuid.uuid4())
                    observed_plan_revision = int((await session.execute(
                        select(func.coalesce(func.max(AgentTaskPlanRevision.revision), 0)).where(
                            AgentTaskPlanRevision.task_id == task.id
                        )
                    )).scalar_one())
                    correction_command = AgentTaskCommand(
                        task_id=task.id,
                        action="steer",
                        idempotency_key=f"budget-review:{idempotency_key}",
                        expected_version=expected_version,
                        status="accepted",
                    )
                    session.add(correction_command)
                    await session.flush()
                    correction = {
                        "id": correction_id,
                        "correction_id": correction_id,
                        "command_id": correction_command.id,
                        "operation_id": correction_command.id,
                        "instruction": " ".join(guidance.split()).strip(),
                        "scope": "remaining_work",
                        "status": "pending",
                        "source": "budget_review",
                        "source_run_id": run.id,
                        "observed_task_version": expected_version + 1,
                        "observed_plan_revision": observed_plan_revision,
                        "submitted_at": now.isoformat(),
                    }
                    replace_jsonb_field(correction_command, "result_json", {
                        "correction": correction,
                        "delivery_mode": "linked_run" if linked_run else "same_run_safe_boundary",
                        "delivery_state": "accepted",
                        "source_run_id": run.id,
                    })
                task.status = AgentTaskStatus.QUEUED.value
                task.current_phase = (
                    "budget_correction_delivery_pending"
                    if decision == "steer" and not linked_run
                    else "budget_continuation_queued"
                )
                task.terminal_reason = None
                task.queued_at = now
                task.completed_at = None
                task.expires_at = now + timedelta(hours=24)
                if linked_run:
                    run.status = AgentRunStatus.COMPLETED.value
                    run.completed_at = now
                else:
                    run.status = AgentRunStatus.RUNNING.value
                event_kind = "task.budget_review_steered" if decision == "steer" else "task.budget_review_continued"
            if decision == "accept_partial" or linked_run:
                latest_sequence = int((await session.execute(
                    select(func.coalesce(func.max(AgentRunEvent.sequence), 0)).where(AgentRunEvent.agent_run_id == run.id)
                )).scalar_one())
                terminal_exists = (await session.execute(select(AgentRunEvent.id).where(
                    AgentRunEvent.agent_run_id == run.id, AgentRunEvent.terminal.is_(True),
                ))).scalar_one_or_none()
                if terminal_exists is None:
                    session.add(AgentRunEvent(
                        agent_run_id=run.id,
                        event_id=f"budget-review:{run.id}:{interrupt_id}:completed",
                        sequence=latest_sequence + 1,
                        attempt=max(1, int(run.task_attempt or 1)),
                        kind="run.completed", occurred_at=now, terminal=True,
                        payload_json={
                            "status": "completed", "result_outcome": "completed_with_warnings",
                            "review_decision": decision, "linked_continuation": linked_run,
                            "provisional_answer": pending.get("provisional_answer"),
                        },
                        source_metadata_json={"framework": "product", "source_event": "task.budget_review.respond"},
                    ))
            if isinstance(run.debug_trace_json, dict):
                replace_jsonb_field(run, "debug_trace_json", append_runtime_event_to_debug_payload(
                    run.debug_trace_json,
                    "intervention.responded",
                    attributes={
                        "askpdf.task.id": task.id, "askpdf.run.id": run.id,
                        "askpdf.intervention.kind": "budget_review",
                    },
                    output_data={"decision": decision, "linked_run": linked_run, "guidance_provided": bool(guidance)},
                    run_status=run.status,
                    completed_at=run.completed_at,
                ))
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, event_kind, agent_run_id=run.id, artifact_id=pending.get("provisional_artifact_id"), payload={
                "interrupt_id": interrupt_id, "decision": decision, "guidance": guidance,
                "linked_run": linked_run and decision != "accept_partial", "version": task.version,
            })
        await session.refresh(task)
        return task, False, linked_run and decision != "accept_partial"


async def create_budget_review(
    task_id: str,
    *,
    run_id: str,
    provisional_answer: str,
    warnings: list[Dict[str, Any]] | None = None,
    gaps: list[str] | None = None,
) -> tuple[AgentTask, Dict[str, Any]]:
    """Create one durable review for the currently exhausted tranche."""

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            run = (await session.execute(select(AgentRun).where(AgentRun.id == run_id, AgentRun.task_id == task_id).with_for_update())).scalar_one()
            existing = dict(run.pending_interrupt_json or {})
            if existing.get("status") == "pending" and existing.get("response_operation") == "task.budget_review.respond":
                return task, existing
            budget = normalize_budget_state(task.budgets_json, (task.config_json or {}).get("limits") or {})
            boundary = dict(budget.get("boundary") or {})
            interrupt_id = str(uuid.uuid4())
            pending = {
                "interrupt_id": interrupt_id,
                "type": "budget_review",
                "response_operation": "task.budget_review.respond",
                "status": "pending",
                "title": "Research budget reached",
                "allowed_actions": ["continue", "accept_partial", "steer"],
                "boundary_strategy": "safe_atomic_boundary",
                "continuation_semantics": "linked_run",
                "preserves_run_id": False,
                "artifact_inheritance": "valid_artifacts",
                "safe_boundary_latency": "after_active_workers",
                "provisional_answer": str(provisional_answer or "").strip(),
                "warnings": list(warnings or []),
                "gaps": list(gaps or []),
                "usage": {
                    "tranche_index": budget.get("tranche_index"),
                    "tranche_limits": budget.get("tranche_limits"),
                    "tranche_usage": budget.get("tranche_usage"),
                    "lifetime_usage": budget.get("lifetime_usage"),
                    "exhausted_dimensions": boundary.get("dimensions") or exhausted_dimensions(budget),
                },
                "created_at": utc_now().isoformat(),
            }
            replace_jsonb_field(run, "pending_interrupt_json", pending)
            run.status = AgentRunStatus.AWAITING_HUMAN.value
            if isinstance(run.debug_trace_json, dict):
                debug_payload = append_runtime_event_to_debug_payload(
                    run.debug_trace_json, "provisional_synthesis.completed",
                    attributes={"askpdf.task.id": task.id, "askpdf.run.id": run.id},
                    output_data={"usable_output": bool(pending["provisional_answer"]), "gaps": pending["gaps"], "warnings": pending["warnings"]},
                    run_status=AgentRunStatus.AWAITING_HUMAN.value,
                )
                replace_jsonb_field(run, "debug_trace_json", append_runtime_event_to_debug_payload(
                    debug_payload, "budget.boundary_requested",
                    attributes={"askpdf.task.id": task.id, "askpdf.run.id": run.id},
                    output_data={"usage": pending["usage"], "continuation_semantics": pending["continuation_semantics"]},
                    run_status=AgentRunStatus.AWAITING_HUMAN.value,
                ))
            task.status = AgentTaskStatus.AWAITING_APPROVAL.value
            task.current_phase = "budget_review"
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(
                session, task, "task.budget_review_requested", agent_run_id=run.id,
                causal_key=f"run:{run.id}:budget-review:{budget.get('tranche_index')}", payload={
                "interrupt_id": interrupt_id, "usage": pending["usage"],
                "accept_partial_enabled": bool(pending["provisional_answer"]), "version": task.version,
                },
            )
        await session.refresh(task)
        return task, pending


async def submit_course_correction(
    task_id: str,
    *,
    run_id: str,
    expected_version: int,
    instruction: str,
    scope: str,
    idempotency_key: str,
) -> tuple[AgentTask, AgentTaskCommand, bool, Dict[str, Any]]:
    """Persist user-authored steering for the next safe orchestration boundary."""

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one_or_none()
            run = (await session.execute(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )).scalar_one_or_none()
            if task is None or run is None or run.task_id != task_id:
                raise AgentTaskConflict("task_run_missing", "Agent task run not found")
            duplicate = (await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.idempotency_key == idempotency_key,
            ))).scalar_one_or_none()
            if duplicate is not None:
                result = dict(duplicate.result_json or {})
                return task, duplicate, True, dict(result.get("correction") or {})
            if task.version != expected_version:
                raise AgentTaskConflict("task_version_conflict", "Task version is stale", current_version=task.version)
            if task.deletion_requested_at is not None or task.status not in {
                AgentTaskStatus.RUNNING.value,
                AgentTaskStatus.QUEUED.value,
                AgentTaskStatus.PAUSED.value,
                AgentTaskStatus.AWAITING_APPROVAL.value,
            }:
                raise AgentTaskConflict("course_correction_unavailable", "Course correction is unavailable for this task state", current_version=task.version)
            if not supports_course_correction(run):
                raise AgentTaskConflict("runtime_capability_unsupported", "The selected runtime does not support course correction", current_version=task.version)
            normalized_instruction = " ".join(instruction.split()).strip()
            if not normalized_instruction:
                raise AgentTaskConflict("course_correction_instruction_required", "Course correction instruction is required")
            observed_plan_revision = int((await session.execute(
                select(func.coalesce(func.max(AgentTaskPlanRevision.revision), 0)).where(
                    AgentTaskPlanRevision.task_id == task.id
                )
            )).scalar_one())
            delivery_mode = (
                "same_run_safe_boundary"
                if not continuation_is_linked(run) and run.status not in TERMINAL_TASK_RUN_STATUSES
                else "linked_run"
            )
            correction_id = str(uuid.uuid4())
            command = AgentTaskCommand(
                task_id=task.id,
                action="steer",
                idempotency_key=idempotency_key,
                expected_version=expected_version,
                status="accepted",
            )
            session.add(command)
            await session.flush()
            correction = {
                "id": correction_id,
                "correction_id": correction_id,
                "command_id": command.id,
                "operation_id": command.id,
                "instruction": normalized_instruction,
                "scope": scope,
                "status": "pending",
                "source": "user",
                "source_run_id": run.id,
                "observed_task_version": expected_version + 1,
                "observed_plan_revision": observed_plan_revision,
                "idempotency_key": idempotency_key,
                "submitted_at": utc_now().isoformat(),
            }
            replace_jsonb_field(command, "result_json", {
                "correction": correction,
                "delivery_mode": delivery_mode,
                "delivery_state": "accepted",
                "source_run_id": run.id,
            })
            task.version += 1
            command.result_version = task.version
            await _append_event(session, task, "task.course_correction_submitted", agent_run_id=run.id, payload={
                **correction,
                "delivery_mode": delivery_mode,
                "delivery_state": "accepted",
                "version": task.version,
            })
        await session.refresh(task)
        await session.refresh(command)
        return task, command, False, correction


async def get_course_correction_command(
    task_id: str,
    *,
    idempotency_key: str,
) -> Optional[AgentTaskCommand]:
    async with async_session_maker() as session:
        return (await session.execute(select(AgentTaskCommand).where(
            AgentTaskCommand.task_id == task_id,
            AgentTaskCommand.action == "steer",
            AgentTaskCommand.idempotency_key == idempotency_key,
        ))).scalar_one_or_none()


async def pending_course_corrections(
    task_id: str,
    *,
    delivery_mode: Optional[str] = None,
) -> list[Dict[str, Any]]:
    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == task_id).with_for_update()
            )).scalar_one_or_none()
        commands = list((await session.execute(
            select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).order_by(AgentTaskCommand.created_at, AgentTaskCommand.id)
        )).scalars().all())
        values = [
            {
                **dict((command.result_json or {}).get("correction") or {}),
                "command_id": command.id,
                "operation_id": command.id,
                "delivery_mode": (command.result_json or {}).get("delivery_mode"),
                "delivery_state": (command.result_json or {}).get("delivery_state") or "accepted",
            }
            for command in commands
        ]
        return [value for value in values if delivery_mode is None or value.get("delivery_mode") == delivery_mode]


async def mark_course_corrections_runtime_applied(
    task_id: str, correction_ids: Iterable[str], *, plan_revision: int,
) -> None:
    """Record prompt/plan incorporation without claiming result satisfaction."""

    selected = {str(value) for value in correction_ids}
    if not selected:
        return
    async with async_session_maker() as session:
        async with session.begin():
            task = await session.get(AgentTask, task_id, with_for_update=True)
            if task is None:
                return
            commands = list((await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).with_for_update())).scalars().all())
            acknowledged: list[str] = []
            for command in commands:
                result = dict(command.result_json or {})
                correction = dict(result.get("correction") or {})
                correction_id = str(correction.get("correction_id") or correction.get("id") or "")
                if correction_id not in selected:
                    continue
                if (
                    result.get("delivery_state") == "incorporated"
                    and int(result.get("runtime_plan_revision") or 0) == plan_revision
                ):
                    continue
                correction.update({"status": "incorporated", "runtime_plan_revision": plan_revision})
                result.update({
                    "correction": correction,
                    "delivery_state": "incorporated",
                    "runtime_plan_revision": plan_revision,
                })
                replace_jsonb_field(command, "result_json", result)
                acknowledged.append(correction_id)
            if acknowledged:
                await _append_event(
                    session, task, "task.course_correction_incorporated",
                    agent_run_id=task.active_run_id,
                    payload={"correction_ids": sorted(acknowledged), "runtime_plan_revision": plan_revision},
                )


async def mark_course_correction_delivered(command_id: str, *, receipt: Dict[str, Any]) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            command = await session.get(AgentTaskCommand, command_id, with_for_update=True)
            if command is None or command.action != "steer" or command.status != "accepted":
                return
            result = dict(command.result_json or {})
            result.update({"delivery_state": "delivered", "runtime_receipt": dict(receipt)})
            replace_jsonb_field(command, "result_json", result)
            correction = dict(result.get("correction") or {})
            if correction.get("source") == "budget_review":
                task = await session.get(AgentTask, command.task_id, with_for_update=True)
                if task is not None and task.current_phase == "budget_correction_delivery_pending":
                    task.current_phase = "budget_continuation_queued"


async def reject_course_correction(command_id: str, *, error: Dict[str, Any]) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            command = await session.get(AgentTaskCommand, command_id, with_for_update=True)
            if command is None or command.action != "steer" or command.status != "accepted":
                return
            task = (await session.execute(
                select(AgentTask).where(AgentTask.id == command.task_id).with_for_update()
            )).scalar_one_or_none()
            result = dict(command.result_json or {})
            result.update({"delivery_state": "rejected", "error": dict(error)})
            replace_jsonb_field(command, "result_json", result)
            command.status = "rejected"
            command.completed_at = utc_now()
            if task is not None:
                correction = dict(result.get("correction") or {})
                if (
                    correction.get("source") == "budget_review"
                    and task.current_phase == "budget_correction_delivery_pending"
                ):
                    source_run = await session.get(
                        AgentRun,
                        str(result.get("source_run_id") or correction.get("source_run_id") or ""),
                        with_for_update=True,
                    )
                    pending = dict(source_run.pending_interrupt_json or {}) if source_run is not None else {}
                    if source_run is not None and pending.get("type") == "budget_review":
                        pending.update({"status": "pending", "delivery_error": dict(error)})
                        pending.pop("decision", None)
                        replace_jsonb_field(source_run, "pending_interrupt_json", pending)
                        source_run.status = AgentRunStatus.AWAITING_HUMAN.value
                    task.status = AgentTaskStatus.AWAITING_APPROVAL.value
                    task.current_phase = "budget_review"
                    task.version += 1
                await _append_event(
                    session,
                    task,
                    "task.course_correction_rejected",
                    agent_run_id=result.get("source_run_id"),
                    payload={
                        "command_id": command.id,
                        "correction_id": correction.get("correction_id") or correction.get("id"),
                        "error": dict(error),
                    },
                )


async def set_course_correction_delivery_mode(
    command_id: str,
    *,
    delivery_mode: str,
    receipt: Optional[Dict[str, Any]] = None,
) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            command = await session.get(AgentTaskCommand, command_id, with_for_update=True)
            if command is None or command.action != "steer" or command.status != "accepted":
                return
            result = dict(command.result_json or {})
            result.update({"delivery_mode": delivery_mode, "delivery_state": "accepted"})
            if receipt is not None:
                result["runtime_receipt"] = dict(receipt)
            replace_jsonb_field(command, "result_json", result)


async def list_pending_course_correction_commands(*, limit: int = 100) -> list[AgentTaskCommand]:
    async with async_session_maker() as session:
        return list((await session.execute(
            select(AgentTaskCommand).where(
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).order_by(AgentTaskCommand.created_at, AgentTaskCommand.id).limit(max(1, min(limit, 500)))
        )).scalars().all())


async def list_course_corrections(task_id: str, *, limit: int = 100) -> list[Dict[str, Any]]:
    """Return bounded product-owned redirect lifecycle projections in submission order."""

    async with async_session_maker() as session:
        commands = list((await session.execute(select(AgentTaskCommand).where(
            AgentTaskCommand.task_id == task_id,
            AgentTaskCommand.action == "steer",
        ).order_by(AgentTaskCommand.created_at, AgentTaskCommand.id).limit(max(1, min(limit, 200))))).scalars().all())
    values: list[Dict[str, Any]] = []
    for command in commands:
        result = dict(command.result_json or {})
        correction = dict(result.get("correction") or {})
        values.append({
            "command_id": command.id,
            "correction_id": correction.get("correction_id") or correction.get("id"),
            "instruction": correction.get("instruction"),
            "status": correction.get("status") or command.status,
            "delivery_mode": result.get("delivery_mode"),
            "delivery_state": result.get("delivery_state") or command.status,
            "linked_run_id": result.get("linked_run_id"),
            "runtime_outcome": dict(result.get("runtime_outcome") or {}),
            "submitted_at": correction.get("submitted_at"),
        })
    return values


async def complete_linked_course_corrections(
    task_id: str,
    *,
    source_run_id: str,
    linked_run_id: str,
) -> None:
    """Record linked-run delivery without claiming result coverage."""
    async with async_session_maker() as session:
        async with session.begin():
            task = await session.get(AgentTask, task_id, with_for_update=True)
            commands = list((await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).with_for_update())).scalars().all())
            linked: list[str] = []
            for command in commands:
                result = dict(command.result_json or {})
                if (
                    str(result.get("source_run_id") or "") != source_run_id
                    and str(result.get("delivery_mode") or "") != "linked_run"
                ):
                    continue
                correction = dict(result.get("correction") or {})
                correction_id = str(correction.get("correction_id") or correction.get("id") or "")
                correction.update({"status": "linked", "linked_run_id": linked_run_id})
                result.update({
                    "correction": correction, "delivery_mode": "linked_run",
                    "delivery_state": "linked", "linked_run_id": linked_run_id,
                })
                replace_jsonb_field(command, "result_json", result)
                linked.append(correction_id)
            if task is not None and linked:
                await _append_event(
                    session, task, "task.course_correction_linked", agent_run_id=linked_run_id,
                    payload={"correction_ids": linked, "source_run_id": source_run_id, "linked_run_id": linked_run_id},
                )


async def queue_linked_course_correction(task_id: str, *, run_id: str) -> AgentTask:
    """Queue exactly one linked run after preserving the source run's terminal outcome."""

    async with async_session_maker() as session:
        async with session.begin():
            task = (await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())).scalar_one()
            run = (await session.execute(select(AgentRun).where(AgentRun.id == run_id, AgentRun.task_id == task_id).with_for_update())).scalar_one()
            commands = list((await session.execute(select(AgentTaskCommand).where(
                AgentTaskCommand.task_id == task_id,
                AgentTaskCommand.action == "steer",
                AgentTaskCommand.status == "accepted",
            ).with_for_update())).scalars().all())
            corrections = [dict((value.result_json or {}).get("correction") or {}) for value in commands]
            if not corrections:
                return task
            if task.deletion_requested_at is not None or task.status in {AgentTaskStatus.CANCELLING.value, AgentTaskStatus.CANCELLED.value}:
                for command in commands:
                    result = dict(command.result_json or {})
                    result.update({"delivery_state": "rejected", "error": {"code": "course_correction_cancelled"}})
                    replace_jsonb_field(command, "result_json", result)
                    command.status = "rejected"
                    command.completed_at = utc_now()
                return task
            if run.status not in TERMINAL_TASK_RUN_STATUSES:
                raise AgentTaskConflict("course_correction_source_run_active", "Linked correction requires a terminal source run")
            if (
                task.active_run_id == run.id
                and task.status == AgentTaskStatus.QUEUED.value
                and task.current_phase == "course_correction_queued"
            ):
                return task
            now = utc_now()
            if isinstance(run.debug_trace_json, dict):
                replace_jsonb_field(run, "debug_trace_json", append_runtime_event_to_debug_payload(
                    run.debug_trace_json, "linked_run.created",
                    attributes={"askpdf.task.id": task.id, "askpdf.run.id": run.id},
                    output_data={"parent_run_id": run.id, "correction_ids": [value.get("id") for value in corrections]},
                    run_status=run.status, completed_at=run.completed_at,
                ))
            task.status = AgentTaskStatus.QUEUED.value
            task.current_phase = "course_correction_queued"
            task.queued_at = now
            task.lease_owner = None
            task.lease_expires_at = None
            task.version += 1
            await _append_event(session, task, "linked_run.created", agent_run_id=run.id, payload={
                "parent_run_id": run.id, "correction_ids": [value.get("id") for value in corrections], "version": task.version,
            })
        await session.refresh(task)
        return task


async def expire_stale_tasks(*, limit: int = 100) -> int:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            rows = list((await session.execute(
                select(AgentTask)
                .where(
                    AgentTask.status.in_([
                        AgentTaskStatus.CREATED.value,
                        AgentTaskStatus.QUEUED.value,
                        AgentTaskStatus.PAUSED.value,
                        AgentTaskStatus.AWAITING_APPROVAL.value,
                    ]),
                    AgentTask.expires_at.is_not(None),
                    AgentTask.expires_at < now,
                )
                .with_for_update(skip_locked=True)
                .limit(max(1, min(limit, 1000)))
            )).scalars().all())
            for task in rows:
                task.status = AgentTaskStatus.EXPIRED.value
                task.current_phase = "expired"
                task.terminal_reason = "idle_or_approval_expired"
                task.completed_at = now
                task.lease_owner = None
                task.lease_expires_at = None
                task.version += 1
                if task.active_run_id:
                    run = await session.get(AgentRun, task.active_run_id)
                    if run is not None and run.status in ACTIVE_TASK_RUN_STATUSES:
                        run.status = AgentRunStatus.EXPIRED.value
                        run.completed_at = now
                        pending = dict(run.pending_interrupt_json or {})
                        if pending:
                            pending["status"] = "expired"
                            pending["decision"] = {"action": "expire", "reason": task.terminal_reason}
                            replace_jsonb_field(run, "pending_interrupt_json", pending)
                        replace_jsonb_field(run, "error_json", {
                            "code": "agent_task_expired", "raw_message": task.terminal_reason, "retryable": False,
                        })
                await _append_event(session, task, "task.expired", agent_run_id=task.active_run_id, payload={"reason": task.terminal_reason})
            return len(rows)
