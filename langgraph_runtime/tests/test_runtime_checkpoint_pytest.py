from __future__ import annotations

import os
import uuid
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_runtime.checkpointing import open_agent_checkpointer


class _CheckpointState(TypedDict, total=False):
    question: str
    decision: str


def _approval_node(state: _CheckpointState) -> _CheckpointState:
    decision = interrupt({"kind": "approval", "question": state["question"]})
    return {"decision": str(decision)}


def _compile_approval_graph(checkpointer):
    graph = StateGraph(_CheckpointState)
    graph.add_node("approval", _approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST") != "1",
    reason="set ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST=1 to run the runtime Postgres checkpoint test",
)
async def test_runtime_graph_resumes_after_postgres_checkpointer_reopen():
    thread_id = f"runtime-checkpoint-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    async with open_agent_checkpointer(setup=False) as first_checkpointer:
        first_graph = _compile_approval_graph(first_checkpointer)
        paused = await first_graph.ainvoke({"question": "Continue?"}, config=config)

    assert paused["__interrupt__"]
    assert paused["__interrupt__"][0].value == {"kind": "approval", "question": "Continue?"}

    async with open_agent_checkpointer(setup=False) as second_checkpointer:
        second_graph = _compile_approval_graph(second_checkpointer)
        resumed = await second_graph.ainvoke(Command(resume="approved"), config=config)
        snapshot = await second_graph.aget_state(config)
        await second_checkpointer.adelete_thread(thread_id)

    assert resumed["decision"] == "approved"
    assert snapshot.values["decision"] == "approved"
    assert not snapshot.next
