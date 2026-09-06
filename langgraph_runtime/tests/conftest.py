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
    os.environ.setdefault("LANGGRAPH_RUNTIME_BINDING_SECRET", "test-langgraph-runtime-binding-secret-32-characters")
    os.environ.setdefault("DEFAULT_TOKEN_BUDGET", "8192")
    os.environ.setdefault("REPLANS_LIMIT", "10")
    os.environ.setdefault("MAX_CUSTOM_INSTRUCTIONS_CHARS", "2000")
    os.environ.setdefault("MAX_SYSTEM_ROLE_CHARS", "500")
