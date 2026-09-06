import inspect

import pytest

from app.mcp.registry import MCP_TOOL_DEFINITIONS, enabled_definitions


def test_mcp_runner_includes_all_framework_neutral_tests():
    from scripts.run_tests import MCP_TEST_FILES

    assert "test_mcp_framework_neutral.py" in MCP_TEST_FILES
    assert set(MCP_TEST_FILES) >= {
        "test_mcp_context.py",
        "test_mcp_transport.py",
        "test_mcp_contracts.py",
        "test_mcp_compatibility.py",
        "test_mcp_tool_adapter.py",
        "test_mcp_framework_neutral.py",
    }


def test_migrated_mcp_handlers_are_framework_neutral():
    migrated = {
        "get_thread_shape",
        "search_documents",
        "search_document_by_id",
        "search_thread_conversation_history",
        "search_durable_memory",
        "search_thread_events",
        "search_web",
        "wikipedia",
        "wikidata",
        "arxiv",
        "pub_med",
        "pubmed",
        "semanticscholar",
        "semantic_scholar",
        "stack_exchange",
        "yahoo_finance_news",
    }
    for name in migrated:
        source = inspect.getsource(MCP_TOOL_DEFINITIONS[name].handler)
        assert ".ainvoke(" not in source
        assert "RunnableConfig" not in source


def test_registry_definitions_are_typed_and_contract_backed():
    definitions = enabled_definitions()
    assert definitions
    for name, definition in definitions.items():
        assert definition.name == name
        assert definition.registry_contract_id
        assert definition.contract_version
        assert definition.server_name
        assert callable(definition.handler)


def test_logical_server_groups_have_unique_known_tools():
    from app.mcp.registry import logical_server_groups

    groups = logical_server_groups()
    tools = [tool for group in groups.values() for tool in group.tool_names]
    assert {"first_party_context", "first_party_research"} <= set(groups)
    assert len(tools) == len(set(tools))
    assert set(tools) <= set(MCP_TOOL_DEFINITIONS)


@pytest.mark.asyncio
async def test_durable_memory_budget_is_clamped_to_contract_limit(monkeypatch):
    from app.models.memory_limits import MAX_MEMORY_CONTEXT_CHARS
    from app.tools.context import ToolInvocationContext
    from app.tools.retrieval_memory import search_durable_memory
    from app.tools.contracts import DocumentSearchRequest

    captured = {}

    async def fake_build(**_kwargs):
        return object(), object(), object()

    async def fake_search(_context, request, **_kwargs):
        captured["char_budget"] = request.char_budget
        return {"memories": [], "scopes": [], "scope_policy": {}}

    monkeypatch.setattr("app.services.memory_tool_service.build_memory_tool_context", fake_build)
    monkeypatch.setattr("app.services.memory_tool_service.search_memory_tool", fake_search)

    result = await search_durable_memory(
        DocumentSearchRequest(query="remember this", max_results=10),
        ToolInvocationContext(thread_id="thread-1", context_window=20_000),
    )

    assert captured["char_budget"] == MAX_MEMORY_CONTEXT_CHARS
    assert result.ok is True


@pytest.mark.asyncio
async def test_memory_prepare_change_mcp_enforces_conversation_review_policy(monkeypatch):
    from app.models.memory_tools import MemoryPrepareChangeInput
    from app.tools.context import ToolInvocationContext
    from app.tools.memory_manager import memory_prepare_change

    prepare = pytest.importorskip("app.tools.memory_manager")
    called = False

    async def fake_prepare(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"operations": []}

    monkeypatch.setattr(prepare, "prepare_memory_change", fake_prepare)
    context = ToolInvocationContext(extensions={
        "memory_tool_context": {"selected_scope_type": "thread", "selected_scope_id": "thread-1"},
        "scope_ids": ["project-1"],
        "curator_mode": "conversation_review",
    })
    result = await memory_prepare_change(
        MemoryPrepareChangeInput(intents=[{
            "action": "create", "scope_type": "project", "content": "invalid",
            "override_target_ids": ["project-1"],
        }]),
        context,
    )
    assert result.ok is False
    assert called is False


@pytest.mark.asyncio
async def test_memory_web_mcp_enforces_run_budget(monkeypatch):
    from app.tools.context import ToolInvocationContext
    from app.tools.memory_manager import InternetSearchRequest, internet_search

    search = pytest.importorskip("app.tools.memory_manager")
    called = False

    async def fake_search(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"status": "ok", "sources": []}

    monkeypatch.setattr(search, "search_internet", fake_search)
    result = await internet_search(
        InternetSearchRequest(query="latest result"),
        ToolInvocationContext(extensions={
            "memory_tool_context": {"selected_scope_type": "thread", "selected_scope_id": "thread-1"},
            "capabilities": ["web:search"],
            "web_search_mode": "allow",
            "web_call_count": 2,
            "web_call_limit": 2,
        }),
    )
    assert result.ok is True
    assert '"status": "limit_reached"' in result.content
    assert called is False
