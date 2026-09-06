import inspect

import pytest


@pytest.mark.asyncio
async def test_workflow_tool_invocation_dispatches_by_mcp_tool_name(monkeypatch):
    from langgraph_runtime.workflows import runtime_invocation

    calls = []

    class FakeExecutor:
        async def ainvoke(self, value, config=None):
            calls.append((value, config))
            return {"content": "mcp-result"}

    monkeypatch.setattr(
        runtime_invocation,
        "resolve_tool_executor",
        lambda tool_name, *, caller_node, config: (
            calls.append((tool_name, caller_node, config)) or FakeExecutor()
        ),
    )

    result = await runtime_invocation.invoke_tool_for_node(
        "search_documents",
        {"query": "question"},
        state={},
        config={},
        node="retrieval_worker",
        started=0.0,
    )

    assert result == {"content": "mcp-result"}
    assert calls[0][0] == "search_documents"
    assert calls[1][0] == {"query": "question"}
    assert "tool" not in inspect.signature(runtime_invocation.invoke_tool_for_node).parameters
