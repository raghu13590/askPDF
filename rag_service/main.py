"""
main.py - FastAPI entrypoint for the Processing Service (Modular version)

This module handles:
- Service initialization and lifespan
- CORS configuration
- Inclusion of modular API routes
"""

import logging
import os
import asyncio
import time
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging - LOG_LEVEL must be explicitly set
_log_level_str = os.environ.get("LOG_LEVEL")
if _log_level_str is None:
    raise RuntimeError("LOG_LEVEL environment variable is required")
log_level = _log_level_str.upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logging.getLogger("app").setLevel(getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Import modular components after logging is configured so app.* loggers emit in Docker.
from app.api.threads import router as threads_router
from app.api.projects import router as projects_router
from app.api.memories import router as memories_router
from app.api.memory_manager import router as memory_manager_router
from app.api.files import router as files_router
from app.api.messages import router as messages_router
from app.api.models import router as models_router
from app.api.agent_workflows import router as agent_workflows_router
from app.api.agent_tasks import router as agent_tasks_router
from app.api.tools import router as tools_router
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.execution_stream import drain_retained_executions
from app.db import ensure_default_project
from app.db.connection_sqlmodel import close_db
from app.db.vector import close_vector_db, get_vector_db
from app.services.memory_service import (
    retry_pending_memory_indexes,
)
from app.services.memory_repair_scheduler import shutdown_memory_repairs
from app.services.embedding_materialization_service import embedding_job_worker
from app.services.agent_task_runtime import run_task_worker
from app.mcp.server import get_http_app
from app.runtime.hermes_profile import HERMES_BASE_TOOL_IDS, HERMES_EXTERNAL_TOOL_IDS
from app.http_clients import close_http_clients, init_http_clients
from app.runtime.registry import get_runtime_registry
from app.runtime.hermes_config import hermes_runtime_enabled, validate_hermes_model_compatibility
from runtime_protocol.configuration import validate_runtime_environment


AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS = 30
RETAINED_EXECUTION_SHUTDOWN_GRACE_SECONDS = 30
MCP_HTTP_APP = get_http_app()


async def _probe_runtime_readiness() -> None:
    """Refresh product readiness without making startup depend on runtimes."""
    registry = get_runtime_registry()
    results: dict[str, dict[str, object]] = {}

    async def probe(adapter: object) -> None:
        identity = registry.deployment_id(adapter)  # type: ignore[arg-type]
        try:
            capabilities = await adapter.deployment_capabilities()  # type: ignore[attr-defined]
            results[identity] = {
                "status": "ready",
                "protocol_version": capabilities.protocol_version,
                "minimum_compatible_version": capabilities.minimum_compatible_version,
            }
        except Exception as exc:
            results[identity] = {
                "status": "unavailable",
                "reason": type(exc).__name__,
            }

    adapters = [
        adapter for adapter in registry.adapters()
        if getattr(adapter, "framework", "") != "hermes" or hermes_runtime_enabled()
    ]
    await asyncio.gather(*(probe(adapter) for adapter in adapters))
    app.state.runtime_readiness = {
        "checked_at": time.time(),
        "runtimes": results,
        "ready": bool(results) and all(item["status"] == "ready" for item in results.values()),
    }


async def _runtime_readiness_loop(stop: asyncio.Event) -> None:
    interval = max(1.0, float(os.getenv("AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS", "30")))
    while not stop.is_set():
        await _probe_runtime_readiness()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
HERMES_OFFLINE_MCP_APP = get_http_app(
    allowed_tools=frozenset(HERMES_BASE_TOOL_IDS), require_execution_token=True,
)
HERMES_EXTERNAL_MCP_APP = get_http_app(
    allowed_tools=frozenset(HERMES_BASE_TOOL_IDS + HERMES_EXTERNAL_TOOL_IDS), require_execution_token=True,
)


def _record_agent_task_worker_completion(app: FastAPI, task: asyncio.Task) -> None:
    """Make an unexpected integrated-worker exit immediately observable."""
    if getattr(app.state, "agent_task_worker_status", None) == "stopping":
        app.state.agent_task_worker_status = "stopped"
        return
    app.state.agent_task_worker_status = "failed"
    if task.cancelled():
        logger.critical("Integrated agent task worker was cancelled unexpectedly")
        return
    error = task.exception()
    if error is None:
        logger.critical("Integrated agent task worker exited unexpectedly without an error")
    else:
        logger.critical(
            "Integrated agent task worker exited unexpectedly",
            exc_info=(type(error), error, error.__traceback__),
        )


async def _memory_maintenance_loop(stop_event: asyncio.Event) -> None:
    """Incrementally retry pending and failed memory indexes."""

    interval = max(30, int(os.environ.get("MEMORY_MAINTENANCE_INTERVAL_SECONDS", "300")))
    batch_size = max(1, min(500, int(os.environ.get("MEMORY_MAINTENANCE_BATCH_SIZE", "100"))))
    while not stop_event.is_set():
        try:
            await retry_pending_memory_indexes(limit=batch_size)
        except Exception:
            logger.exception("Incremental memory maintenance failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service lifespan management.
    Performs startup tasks like database initialization.
    """
    logger.info("--- RAG Service Starting ---")
    memory_maintenance_stop = None
    memory_maintenance_task = None
    embedding_job_stop = None
    embedding_job_task = None
    agent_task_worker_stop = None
    agent_task_worker = None
    runtime_readiness_stop = None
    runtime_readiness_task = None
    mcp_lifespans = []
    try:
        validate_runtime_environment(service="control_plane")
        if hermes_runtime_enabled():
            validate_hermes_model_compatibility()
        get_runtime_registry().initialize()
        app.state.runtime_readiness = {"checked_at": None, "runtimes": {}, "ready": False}
        # Keep cleanup active from the first allocation onward.  In
        # particular, database or MCP startup failures must not strand the
        # application-scoped HTTP clients initialized above them.
        await init_http_clients()
        await ensure_default_project()
        await AgentWorkflowRepository().seed_builtin_workflows()
        logger.info("Database migrations already applied; application data initialization complete.")

        try:
            logger.info("Initializing Weaviate collections...")
            await get_vector_db().ensure_collections()
            logger.info("Weaviate collection initialization complete.")
        except Exception:
            logger.exception("Failed to initialize Weaviate collections")

        memory_maintenance_stop = asyncio.Event()
        memory_maintenance_task = asyncio.create_task(
            _memory_maintenance_loop(memory_maintenance_stop)
        )
        embedding_job_stop = asyncio.Event()
        embedding_job_task = asyncio.create_task(embedding_job_worker(embedding_job_stop))
        agent_task_worker_stop = asyncio.Event()
        app.state.agent_task_worker_status = "running"
        agent_task_worker = asyncio.create_task(
            run_task_worker(stop_event=agent_task_worker_stop),
            name="agent-task-worker",
        )
        agent_task_worker.add_done_callback(
            lambda task: _record_agent_task_worker_completion(app, task)
        )
        runtime_readiness_stop = asyncio.Event()
        runtime_readiness_task = asyncio.create_task(_runtime_readiness_loop(runtime_readiness_stop))
        # The SDK streamable-HTTP session manager is single-use. Rebuild the
        # mounted app for every FastAPI lifespan so TestClient restarts,
        # reloads, and application shutdown/startup cycles get a fresh manager.
        global MCP_HTTP_APP, HERMES_OFFLINE_MCP_APP, HERMES_EXTERNAL_MCP_APP
        MCP_HTTP_APP = get_http_app()
        HERMES_OFFLINE_MCP_APP = get_http_app(
            allowed_tools=frozenset(HERMES_BASE_TOOL_IDS), require_execution_token=True,
        )
        HERMES_EXTERNAL_MCP_APP = get_http_app(
            allowed_tools=frozenset(HERMES_BASE_TOOL_IDS + HERMES_EXTERNAL_TOOL_IDS), require_execution_token=True,
        )
        apps_by_route = {
            "internal-mcp": MCP_HTTP_APP,
            "internal-hermes-mcp-offline": HERMES_OFFLINE_MCP_APP,
            "internal-hermes-mcp-external": HERMES_EXTERNAL_MCP_APP,
        }
        for route in app.router.routes:
            mounted = apps_by_route.get(getattr(route, "name", None))
            if mounted is not None:
                route.app = mounted
        for mounted in apps_by_route.values():
            manager = mounted.router.lifespan_context(mounted)
            await manager.__aenter__()
            mcp_lifespans.append(manager)
        yield
    finally:
        logger.info("--- RAG Service Shutting Down ---")
        try:
            await drain_retained_executions(RETAINED_EXECUTION_SHUTDOWN_GRACE_SECONDS)
        except Exception:
            logger.exception("Error draining retained agent executions")
        for mcp_lifespan in reversed(mcp_lifespans):
            try:
                await mcp_lifespan.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error during MCP lifespan shutdown")
        if agent_task_worker is not None and agent_task_worker_stop is not None:
            app.state.agent_task_worker_status = "stopping"
            agent_task_worker_stop.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(agent_task_worker),
                    timeout=AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent task worker exceeded %ss shutdown grace; cancelling active execution",
                    AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS,
                )
                agent_task_worker.cancel()
                with suppress(asyncio.CancelledError):
                    await agent_task_worker
            except Exception:
                logger.exception("Agent task worker exited unexpectedly")
        if runtime_readiness_task is not None and runtime_readiness_stop is not None:
            runtime_readiness_stop.set()
            await runtime_readiness_task
        if memory_maintenance_task is not None and memory_maintenance_stop is not None:
            memory_maintenance_stop.set()
            await memory_maintenance_task
        if embedding_job_task is not None and embedding_job_stop is not None:
            embedding_job_stop.set()
            await embedding_job_task
        try:
            await shutdown_memory_repairs()
        except Exception:
            logger.exception("Error during memory repair shutdown")
        try:
            logger.info("Closing database connections...")
            await close_db()
            logger.info("Database connections closed.")
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
        try:
            logger.info("Closing Weaviate client connection...")
            close_vector_db()
            logger.info("Weaviate client connection closed.")
        except Exception as e:
            logger.error(f"Error during Weaviate shutdown: {e}")
        try:
            await close_http_clients()
        except Exception:
            logger.exception("Error during HTTP client shutdown")

app = FastAPI(
    title="RAG Service",
    description="Modular Retrieval-Augmented Generation Service for AskPDF",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/internal/hermes-mcp/preflight", include_in_schema=False)
async def hermes_mcp_preflight(
    execution_context: str | None = Header(default=None, alias="X-AskPDF-Execution-Context"),
    expected_run_id: str | None = Header(default=None, alias="X-AskPDF-Expected-Run-Id"),
    expected_thread_id: str | None = Header(default=None, alias="X-AskPDF-Expected-Thread-Id"),
    expected_task_id: str | None = Header(default=None, alias="X-AskPDF-Expected-Task-Id"),
):
    """Validate a run-scoped MCP context without invoking or auditing a tool."""
    from app.mcp.execution_context_token import (
        ExecutionContextTokenError,
        decode_execution_context_token,
        validate_execution_context_identity,
    )

    if not execution_context:
        logger.warning("Hermes MCP preflight rejected reason=missing")
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"})
    if not expected_run_id or not expected_thread_id or not expected_task_id:
        logger.warning("Hermes MCP preflight rejected reason=identity_mismatch fields=expected_identity")
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"})
    try:
        context = decode_execution_context_token(execution_context)
    except ExecutionContextTokenError as exc:
        logger.warning("Hermes MCP preflight rejected reason=%s", exc.reason)
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"}) from exc
    try:
        validate_execution_context_identity(
            context,
            run_id=expected_run_id,
            thread_id=expected_thread_id,
            task_id=expected_task_id,
        )
    except ExecutionContextTokenError as exc:
        logger.warning("Hermes MCP preflight rejected reason=%s", exc.reason)
        raise HTTPException(status_code=401, detail={"code": "mcp_execution_context_rejected"}) from exc
    return {"status": "ok", "run_id": context.run_id}

# CORS Middleware for cross-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular routes
app.include_router(threads_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(memories_router, prefix="/api")
app.include_router(memory_manager_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(agent_workflows_router, prefix="/api")
app.include_router(agent_tasks_router, prefix="/api")
app.include_router(tools_router, prefix="/api")

@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    worker_status = getattr(app.state, "agent_task_worker_status", "not_started")
    payload = {
        "status": "ok",
        "service": "rag-service",
        "version": "2.0.0",
        "mode": "modular",
        "agent_task_worker": worker_status,
    }
    return payload


@app.get("/ready")
async def product_readiness():
    """Readiness for product traffic, including mandatory external runtimes."""
    worker_status = getattr(app.state, "agent_task_worker_status", "not_started")
    runtime_readiness = getattr(app.state, "runtime_readiness", {})
    checked_at = runtime_readiness.get("checked_at")
    try:
        freshness_window = max(
            5.0,
            3.0 * float(os.getenv("AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS", "30")),
        )
    except (TypeError, ValueError):
        freshness_window = 5.0
    runtime_fresh = (
        isinstance(checked_at, (int, float))
        and time.time() - float(checked_at) <= freshness_window
    )
    ready = (
        worker_status == "running"
        and bool(runtime_readiness.get("ready"))
        and runtime_fresh
    )
    payload = {
        "status": "ok" if ready else "unavailable",
        "service": "rag-service",
        "agent_task_worker": worker_status,
        "runtime_readiness": {**runtime_readiness, "fresh": runtime_fresh},
    }
    return payload if payload["status"] == "ok" else JSONResponse(status_code=503, content=payload)


app.mount("/internal/mcp/", MCP_HTTP_APP, name="internal-mcp")
app.mount("/internal/hermes-mcp/offline/", HERMES_OFFLINE_MCP_APP, name="internal-hermes-mcp-offline")
app.mount("/internal/hermes-mcp/external/", HERMES_EXTERNAL_MCP_APP, name="internal-hermes-mcp-external")


# Mount static files last to avoid shadowing API routes.
app.mount("/files", StaticFiles(directory="/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
