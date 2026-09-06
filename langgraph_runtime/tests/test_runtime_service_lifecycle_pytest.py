from __future__ import annotations

from fastapi.testclient import TestClient
import httpx
import time
import pytest
from runtime_protocol.contracts import AgentRuntimeResult
from runtime_protocol.protocol import RUNTIME_MINIMUM_COMPATIBLE_VERSION, RUNTIME_PROTOCOL_VERSION
from langgraph_runtime.execution_store import ExecutionStore

from langgraph_runtime.api import create_app
from langgraph_runtime.dependencies import probe_mcp, probe_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 406, 422, 500])
async def test_mcp_readiness_rejects_http_errors(status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(status)))
    try:
        result = await probe_mcp("http://mcp/internal/mcp/", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": False, "http_status": status, "reason": "unexpected_status"}


@pytest.mark.asyncio
async def test_mcp_readiness_requires_a_valid_tools_list():
    response = {"jsonrpc": "2.0", "id": "runtime-readiness", "result": {"tools": [{"name": "get_thread_shape"}]}}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response)))
    try:
        result = await probe_mcp("http://mcp/internal/mcp/", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": True, "http_status": 200, "protocol": "mcp", "capability_ids": ["get_thread_shape"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 406, 422, 500])
async def test_provider_readiness_rejects_http_errors(status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(status)))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": False, "http_status": status, "reason": "unexpected_status"}


@pytest.mark.asyncio
async def test_provider_readiness_requires_a_models_list():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": True, "http_status": 200, "capability_ids": []}


def test_runtime_healthz_is_liveness_only(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "langgraph-runtime"}


def test_runtime_readyz_is_structured_when_optional_probes_are_unconfigured(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["checkpoint_store"]["backend"] == "memory"
    assert payload["checks"]["execution_store"]["status"] == "ok"
    assert "mcp" not in payload["checks"]
    assert "provider" not in payload["checks"]
    assert "DATABASE_URL" not in response.text


def test_runtime_startup_and_dependency_endpoints_are_separate(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        assert client.get("/startupz").json() == {"status": "ok"}
        dependency_response = client.get("/v1/dependencies")
    dependencies = dependency_response.json()["result"]["dependencies"]
    assert dependencies["mcp"]["state"] == "not_configured"
    assert dependencies["provider"]["state"] == "not_configured"


def test_runtime_accepts_missing_outer_protocol_metadata(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.post(
            "/v1/capabilities",
            json={
                "definition": {
                    "definition_id": "router_rag_agent",
                    "framework": "langgraph",
                    "builder_id": "langgraph_graph",
                },
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["protocol_version"] == RUNTIME_PROTOCOL_VERSION
    assert payload["minimum_compatible_version"] == RUNTIME_MINIMUM_COMPATIBLE_VERSION
    assert "capabilities" in payload["result"]


def test_runtime_definition_errors_are_versioned_for_control_plane_parsing(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app(require_auth=False)) as client:
        response = client.post(
            "/v1/resolve",
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "minimum_compatible_version": RUNTIME_MINIMUM_COMPATIBLE_VERSION,
            },
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["protocol_version"] == RUNTIME_PROTOCOL_VERSION
    assert payload["minimum_compatible_version"] == RUNTIME_MINIMUM_COMPATIBLE_VERSION
    assert payload["error"]["code"] == "runtime_definition_invalid"
    assert "detail" not in payload


def test_dependency_outage_marks_readiness_unavailable_but_blocks_required_run(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "http://unavailable/mcp")
    monkeypatch.setenv("MCP_TRANSPORT", "loopback_http")
    monkeypatch.setenv("LLM_API_URL", "")

    async def unavailable_probe(*_args, **_kwargs):
        return {"ok": False, "reason": "ConnectError"}

    monkeypatch.setattr("langgraph_runtime.dependencies.probe_mcp", unavailable_probe)
    payload = {
        "protocol_version": RUNTIME_PROTOCOL_VERSION,
        "minimum_compatible_version": RUNTIME_MINIMUM_COMPATIBLE_VERSION,
        "operation_id": "dependency-blocked:start",
        "request": {
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
            "minimum_compatible_version": RUNTIME_MINIMUM_COMPATIBLE_VERSION,
            "run_id": "dependency-blocked",
            "thread_id": "thread-1",
            "definition_id": "router_rag_agent",
            "framework": "langgraph",
            "builder_id": "langgraph_graph",
            "input": {
                "question": "hello",
                "mcp_allowed_tool_ids": ["search_documents"],
            },
            "options": {},
        },
        "context": {"resolved_spec": {"config": {"allowed_tool_ids": ["document_evidence"]}}},
        "definition": {
            "definition_id": "router_rag_agent",
            "framework": "langgraph",
            "builder_id": "langgraph_graph",
            "capabilities": {},
            "definition_metadata": {},
        },
    }
    with TestClient(create_app(require_auth=False)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/startupz").status_code == 200
        readiness = client.get("/readyz")
        assert readiness.status_code == 503
        assert readiness.json()["checks"]["configured_dependencies"]["dependency"] == "mcp"
        response = client.post("/v1/runs/start", json=payload)
        cancel_response = client.post(
            "/v1/runs/dependency-blocked/cancel",
            json={
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "minimum_compatible_version": RUNTIME_MINIMUM_COMPATIBLE_VERSION,
                "request": payload["request"],
            },
        )
        assert cancel_response.status_code == 404
        assert cancel_response.json()["error"]["code"] == "runtime_run_not_found"
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "runtime_dependency_unavailable"
    assert error["retryable"] is True
    assert error["details"]["dependency"] == "mcp"
    assert "http://unavailable" not in response.text


def test_recovery_loop_reclaims_a_lease_after_restart(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_TRANSPORT", "")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    monkeypatch.setenv("AGENT_RUNTIME_RECOVERY_LOOP_ENABLED", "true")
    monkeypatch.setenv("AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS", "1")

    class FakeAdapter:
        async def prepare_execution_context(self, context):
            return context

        async def start(self, request, *, context, event_sink=None):
            return AgentRuntimeResult(status="completed", output={"answer": "recovered"})

    monkeypatch.setattr("langgraph_runtime.adapter.LangGraphRuntimeAdapter", FakeAdapter)
    store = ExecutionStore(database_url="")
    request = {
        "protocol_version": RUNTIME_PROTOCOL_VERSION,
        "minimum_compatible_version": RUNTIME_MINIMUM_COMPATIBLE_VERSION,
        "run_id": "restart-recovery",
        "thread_id": "thread-1",
        "definition_id": "router_rag_agent",
        "framework": "langgraph",
        "builder_id": "langgraph_graph",
        "input": {"question": "hello"},
        "options": {},
    }

    async def seed():
        await store.create(
            "restart-recovery",
            "start",
            request,
            {"request": request, "context": {}},
            operation_id="restart-recovery:start",
        )
        await store.claim("restart-recovery", owner_id="old-worker", lease_seconds=1)

    import asyncio
    asyncio.run(seed())
    with TestClient(create_app(execution_store=store, require_auth=False)):
        time.sleep(2.2)
        record = store._records["restart-recovery"]
        assert record.status == "completed"
