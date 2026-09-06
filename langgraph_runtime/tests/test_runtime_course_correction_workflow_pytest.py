from __future__ import annotations

import pytest
from types import SimpleNamespace

from langgraph_runtime.workflows.deep_research_execution import runtime_execution_services_factory
from langgraph_runtime.workflows.deep_research_nodes import deep_task_scheduler


@pytest.mark.asyncio
async def test_scheduler_stops_new_dispatch_when_runtime_correction_is_pending() -> None:
    async def corrections():
        return [{
            "correction_id": "correction-1",
            "operation_id": "operation-1",
            "instruction": "Change the remaining research scope.",
            "scope": "remaining_work",
            "status": "accepted",
        }]

    state = {
        "agent_task_id": "task-1",
        "agent_run_id": "run-1",
        "task_limits": {"max_concurrency": 2, "max_fanout": 2},
        "task_todos": [{
            "id": "todo-1",
            "title": "Pending work",
            "description": "Do remaining work",
            "completion_criteria": "Done",
            "status": "pending",
            "priority": 1,
            "required": True,
            "dependency_ids": [],
            "profile_id": "document_researcher",
            "attempt": 0,
            "max_attempts": 2,
            "progress": 0,
            "artifact_ids": [],
            "version": 1,
        }],
    }
    result = await deep_task_scheduler(state, {"configurable": {
        "deep_research_services_factory": runtime_execution_services_factory,
        "cancellation_checker": lambda: False,
        "course_correction_reader": corrections,
    }})

    assert result["task_work_items"] == []
    assert result["task_course_corrections"][0]["id"] == "correction-1"
    assert result["task_todos"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_runtime_replan_preserves_completed_todos_and_artifacts() -> None:
    completed = {
        "id": "done-1", "title": "Completed", "description": "Original work",
        "completion_criteria": "Done", "status": "completed", "priority": 10,
        "required": True, "dependency_ids": [], "profile_id": "document_researcher",
        "attempt": 1, "max_attempts": 2, "progress": 100,
        "result_summary": "Verified result", "artifact_ids": ["artifact-1"], "version": 2,
    }
    services = runtime_execution_services_factory(
        {"task_plan_revision": 1, "task_limits": {}, "task_todos": [completed]},
        {"cancellation_checker": lambda: False},
    )
    proposed = SimpleNamespace(
        id="done-1",
        dependency_ids=[],
        model_dump=lambda **_kwargs: {**completed, "description": "Attempted rewrite"},
    )

    _, todos = await services.persist_plan(
        "task-1", SimpleNamespace(todos=[proposed]), reason="course_correction",
    )

    assert todos[0].status == "completed"
    assert todos[0].description == "Original work"
    assert todos[0].artifact_ids_json == ["artifact-1"]
