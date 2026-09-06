#!/usr/bin/env python3
"""Docker-native test runner for askPDF.

This script is intended to run inside the rag-service image. It owns test
database lifecycle, test grouping, and standalone verification checks so host
platforms and CI can use the same entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg


APP_DIR = Path("/app")
REPO_DIR = Path(os.environ.get("ASKPDF_REPO_DIR", "/workspace"))

UNIT_TEST_FILES = [
    "test_control_plane_import_boundary_pytest.py",
    "test_builder_provider_pytest.py",
    "test_hermes_builder_provider_pytest.py",
    "test_hermes_compose_profile_pytest.py",
    "test_hermes_execution_store_pytest.py",
    "test_hermes_profile_pytest.py",
    "test_hermes_runtime_adapter_pytest.py",
    "test_external_hermes_runtime_smoke_pytest.py",
    "test_real_hermes_container_smoke_pytest.py",
    "test_agent_prompt_behavior.py",
    "test_agent_retry_behavior.py",
    "test_agent_tool_contract_pytest.py",
    "test_dimension_mismatch_scenarios.py",
    "test_external_research_tools.py",
    "test_runtime_execution_store_pytest.py",
    "test_runtime_http_adapter_pytest.py",
    "test_langgraph_boundary_gaps_pytest.py",
    "test_runtime_contracts_pytest.py",
    "test_runtime_hardening_pytest.py",
    "test_runtime_configuration_pytest.py",
    "test_runtime_events_pytest.py",
    "test_runtime_capability_gate_pytest.py",
    "test_runtime_capability_resolver_pytest.py",
    "test_langgraph_capabilities_pytest.py",
    "test_provider_clients.py",
    "test_first_party_tool_contracts.py",
    "test_llm_server_client_pytest.py",
    "test_memory_retrieval_policy_pytest.py",
    "test_memory_hardening_pytest.py",
    "test_memory_review_service_pytest.py",
    "test_message_api_pytest.py",
    "test_model_aware_collections.py",
    "test_model_registry_edge_cases.py",
    "test_modular_visualization_pytest.py",
    "test_parsing_pytest.py",
    "test_production_edge_cases.py",
    "test_temporal_metadata_retrieval.py",
    "test_time_utils.py",
    "test_tool_registry_contracts.py",
    "test_http_client_lifecycle.py",
    "test_workflow_budget.py",
]

MCP_TEST_FILES = [
    "test_hermes_runtime_mcp_contract_pytest.py",
    "test_mcp_context.py",
    "test_mcp_transport.py",
    "test_mcp_contracts.py",
    "test_mcp_compatibility.py",
    "test_mcp_tool_adapter.py",
    "test_mcp_framework_neutral.py",
]

LANGGRAPH_RUNTIME_TEST_TARGETS = [
    "/workspace/langgraph_runtime/tests/test_mcp_runtime_invocation_pytest.py",
]

DB_TEST_FILES = [
    "test_database_connection_pytest.py",
    "test_models_sqlmodel_pytest.py",
    "test_project_memory_repository_pytest.py",
    "test_memory_manager_engine_pytest.py",
    "test_project_lifecycle_service_pytest.py",
    "test_thread_repository_pytest.py",
    "test_file_repository_pytest.py",
    "test_message_repository_pytest.py",
    "test_thread_file_repository_pytest.py",
    "test_stats_repository_pytest.py",
    "test_repository_transactions_pytest.py",
    "test_agent_task_runtime_projection_pytest.py",
    "test_jsonb_operations_pytest.py",
    "test_thread_fork_service_pytest.py",
]

API_TEST_FILES = [
    "test_api_endpoints_pytest.py",
    "test_api_integration_pytest.py",
    "test_project_lifecycle_api_pytest.py",
]

INTEGRATION_TEST_FILES = [
    "test_agent_workflows_pytest.py",
    "test_api_integration_pytest.py",
    "test_model_aware_integration.py",
]

SCHEMA_TEST_FILES = [
    "test_schema_guardrails.py",
    "test_migration_smoke_pytest.py",
]

AGENT_CHECKPOINT_TEST_TARGETS = [
    "/app/tests/test_runtime_checkpoint_pytest.py::test_runtime_graph_resumes_after_postgres_checkpointer_reopen",
]

DIAGNOSTIC_PACKAGES = (
    "pydantic",
    "langchain-core",
    "langchain-community",
    "ddgs",
    "pytest",
    "pytest-asyncio",
    "sqlalchemy",
    "asyncpg",
    "httpx",
)


def _print_dependency_versions() -> None:
    versions = []
    for package in DIAGNOSTIC_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        versions.append(f"{package}={version}")
    print("Test dependency versions: " + ", ".join(versions), flush=True)


def _test_path(name: str) -> str:
    return str(APP_DIR / "tests" / name)


def _postgres_driver_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(_postgres_driver_url(url))
    return urlunsplit((parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment))


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


async def _create_database(admin_url: str, db_name: str) -> None:
    conn = await asyncpg.connect(admin_url)
    try:
        await conn.execute(f"CREATE DATABASE {_quote_ident(db_name)}")
    finally:
        await conn.close()


async def _drop_database(admin_url: str, db_name: str) -> None:
    conn = await asyncpg.connect(admin_url)
    try:
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1
              AND pid <> pg_backend_pid()
            """,
            db_name,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {_quote_ident(db_name)}")
    finally:
        await conn.close()


async def _setup_agent_checkpointer(database_url: str) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(_postgres_driver_url(database_url)) as checkpointer:
        await checkpointer.setup()


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=APP_DIR, env=env, check=True)


def _run_standalone(pdf_path: str | None = None) -> None:
    env = os.environ.copy()
    # Keep the backend import root and include the repository root for the
    # root-level Hermes gateway used by the Hermes runtime integration proof.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_DIR), str(APP_DIR)]))

    if pdf_path:
        candidate = Path(pdf_path)
        if not candidate.is_absolute():
            candidate = REPO_DIR / pdf_path
        if not candidate.exists():
            raise FileNotFoundError(f"PDF file not found: {candidate}")
        _run(["python", str(APP_DIR / "tests" / "test_parsing_service.py"), "--pdf", str(candidate)], env=env)
        return

    script = REPO_DIR / "test_proactive_collections.py"
    if not script.exists():
        raise FileNotFoundError(f"Standalone test script not found: {script}")
    _run(["python", str(script)], env=env)


def _pytest_targets(args: argparse.Namespace) -> list[str]:
    if args.file:
        target = args.file
        if target.startswith("rag_service/tests/"):
            target = target.removeprefix("rag_service/tests/")
        elif target.startswith("/app/tests/"):
            target = target.removeprefix("/app/tests/")
        target = _test_path(target)
        if args.test:
            target = f"{target}::{args.test}"
        return [target]

    if args.test:
        raise SystemExit("Error: --test requires --file to be specified")

    group = args.group
    if args.standalone or args.pdf:
        group = "standalone"
    elif args.unit:
        group = "unit"
    elif args.db or args.db_tests or args.db_only:
        group = "db"
    elif args.integration:
        group = "integration"
    elif args.agent_checkpoint:
        group = "agent-checkpoint"
    elif args.api:
        group = "api"
    elif args.schema:
        group = "schema"
    elif args.mcp:
        group = "mcp"
    elif args.langgraph_runtime:
        group = "langgraph-runtime"
    elif args.all or args.all_tests:
        group = "all"

    if group == "unit":
        return [_test_path(name) for name in UNIT_TEST_FILES]
    if group == "db":
        return [_test_path(name) for name in DB_TEST_FILES]
    if group == "api":
        return [_test_path(name) for name in API_TEST_FILES]
    if group == "integration":
        return [_test_path(name) for name in INTEGRATION_TEST_FILES]
    if group == "agent-checkpoint":
        return AGENT_CHECKPOINT_TEST_TARGETS
    if group == "schema":
        return [_test_path(name) for name in SCHEMA_TEST_FILES]
    if group == "mcp":
        return [_test_path(name) for name in MCP_TEST_FILES]
    if group == "langgraph-runtime":
        return LANGGRAPH_RUNTIME_TEST_TARGETS
    if group == "standalone":
        return []
    if group == "all":
        return [str(APP_DIR / "tests")]

    raise SystemExit(f"Unknown test group: {group}")


def _should_run_standalone(args: argparse.Namespace) -> bool:
    if args.pdf:
        return True
    if args.group == "standalone" or args.standalone:
        return True
    if args.file or args.test:
        return False
    if (
        args.unit
        or args.db
        or args.db_tests
        or args.db_only
        or args.integration
        or args.agent_checkpoint
        or args.api
        or args.schema
        or args.mcp
        or args.langgraph_runtime
    ):
        return False
    return args.group == "all" or args.all or args.all_tests


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run askPDF tests inside Docker.")
    parser.add_argument(
        "--group",
        choices=[
            "unit",
            "db",
            "api",
            "integration",
            "agent-checkpoint",
            "schema",
            "mcp",
            "langgraph-runtime",
            "standalone",
            "all",
        ],
        default=os.environ.get("TEST_GROUP", "all"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--test")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--strict-warnings", action="store_true")
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--pdf")
    parser.add_argument("--db", action="store_true")
    parser.add_argument("--db-tests", action="store_true")
    parser.add_argument("--db-only", action="store_true")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--agent-checkpoint", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--langgraph-runtime", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--all-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    _print_dependency_versions()
    targets = _pytest_targets(args)
    agent_checkpoint_run = args.agent_checkpoint or args.group == "agent-checkpoint"

    base_database_url = os.environ.get("DATABASE_URL")
    if not base_database_url:
        raise SystemExit("DATABASE_URL environment variable is required")

    test_run_id = uuid.uuid4().hex
    test_db_name = f"test_askpdf_{test_run_id}"
    admin_url = _with_database(base_database_url, "postgres")
    test_db_url = _with_database(base_database_url, test_db_name).replace("postgresql://", "postgresql+asyncpg://", 1)
    data_dir = f"/data/test_{test_run_id}"

    print(f"Test run ID: {test_run_id}", flush=True)
    print(f"Test PostgreSQL database: {test_db_name}", flush=True)
    print(f"Test data directory: {data_dir}", flush=True)

    asyncio.run(_create_database(admin_url, test_db_name))
    if agent_checkpoint_run:
        print("Preparing LangGraph Postgres checkpointer schema", flush=True)
        asyncio.run(_setup_agent_checkpointer(test_db_url))

    env = os.environ.copy()
    env["DATABASE_URL"] = test_db_url
    env["TEST_DATABASE_URL"] = test_db_url
    env["DATA_DIR"] = data_dir
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_DIR), str(APP_DIR)]))
    if agent_checkpoint_run:
        env["ASKPDF_AGENT_CHECKPOINTER"] = "postgres"
        env["AGENT_CHECKPOINT_DATABASE_URL"] = test_db_url
        env["ASKPDF_AGENT_CHECKPOINTER_SETUP"] = "false"
        env["ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST"] = "1"
    else:
        env["ASKPDF_AGENT_CHECKPOINTER"] = "memory"
        env.pop("AGENT_CHECKPOINT_DATABASE_URL", None)
        env.pop("ASKPDF_AGENT_CHECKPOINTER_SETUP", None)
        env.pop("ASKPDF_RUN_POSTGRES_CHECKPOINT_TEST", None)

    # Runtime state has its own schema authority. Prepare it for every test
    # group because the runtime store is exercised by both unit and integration
    # tests, while remaining isolated from application Alembic revisions.
    env["AGENT_RUNTIME_EXECUTION_DATABASE_URL"] = test_db_url
    env["RUN_RUNTIME_DB_MIGRATIONS"] = "true"
    _run(["python", "-m", "langgraph_runtime.migrate"], env=env)

    try:
        if targets:
            command = ["pytest", *targets]
            if args.verbose:
                command.append("-v")
            if args.coverage:
                command.extend(["--cov=app", "--cov-report=term-missing"])
            if args.strict_warnings:
                command.extend([
                    "-W", "error::RuntimeWarning",
                    "-W", "error::pydantic.warnings.PydanticDeprecatedSince20",
                    "-W", "error::pytest.PytestUnraisableExceptionWarning",
                    "-o", "asyncio_debug=true",
                ])
                env["PYTHONASYNCIODEBUG"] = "1"
            _run(command, env=env)

        if _should_run_standalone(args):
            _run_standalone(args.pdf)
    finally:
        print(f"Cleaning up test database: {test_db_name}", flush=True)
        asyncio.run(_drop_database(admin_url, test_db_name))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
