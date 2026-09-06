"""Fixtures shared by LangGraph-runtime-owned tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def configure_langgraph_runtime_limits():
    """Configure runtime limits only for tests that execute runtime code."""
    from langgraph_runtime.models.llm import configure_runtime_limits

    configure_runtime_limits(dict(os.environ))
    yield


@pytest.fixture(scope="session", autouse=True)
def configure_runtime_test_environment():
    """Provide the runtime-only limits required by direct component tests."""
    defaults = {
        "AGENT_RUNTIME_LEASE_SECONDS": "120",
        "AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS": "30",
        "AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS": "300",
        "AGENT_RUNTIME_READ_TIMEOUT_SECONDS": "600",
        "AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS": "3",
        "AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS": "1",
        "AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS": "30",
        "AGENT_RUNTIME_SHUTDOWN_GRACE_SECONDS": "30",
        "AGENT_RUNTIME_TERMINAL_CONFIRM_TIMEOUT_SECONDS": "30",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS": "1",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_BYTES": "8192",
        "AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS": "30",
        "AGENT_RUNTIME_DEPENDENCY_TIMEOUT_SECONDS": "5",
        "AGENT_RUNTIME_DEPENDENCY_STALE_SECONDS": "60",
        "AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS": "30",
        "AGENT_RUNTIME_RECOVERY_BATCH_SIZE": "10",
        "AGENT_RUNTIME_RECOVERY_LOOP_ENABLED": "false",
        "MCP_REQUEST_TIMEOUT_SECONDS": "30",
        "MCP_OTEL_ENABLED": "false",
        "LLM_AUTH_MODE": "none",
        "LLM_KEYLESS_PROVIDER": "local",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("LANGGRAPH_RUNTIME_BINDING_SECRET", "test-langgraph-runtime-binding-secret-32-characters")
    os.environ.setdefault("DEFAULT_TOKEN_BUDGET", "8192")
    os.environ.setdefault("REPLANS_LIMIT", "10")
    os.environ.setdefault("MAX_CUSTOM_INSTRUCTIONS_CHARS", "2000")
    os.environ.setdefault("MAX_SYSTEM_ROLE_CHARS", "500")
