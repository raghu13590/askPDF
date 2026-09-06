"""
conftest.py - Pytest configuration and fixtures for database tests.

This module provides shared fixtures for PostgreSQL database testing,
including connection management, session handling, and test data.
"""

import os
import asyncio
import uuid
from typing import AsyncGenerator, Generator
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from httpx import ASGITransport, AsyncClient


os.environ.setdefault("HERMES_API_TOKEN", "test-hermes-api-token-32-characters")

from app.db.models_sqlmodel import (
    Project, Thread, File, ThreadFile,
    ChatTurn, ProcessStatus, MessageRole, AgentRuntimeOperation
)


collect_ignore = [
    "test_modular_visualization.py",
    "test_parsing_service.py",
]


_askpdf_completed_test_calls = 0


def pytest_configure(config: pytest.Config) -> None:
    """Initialize the opt-in proof-suite execution counter."""
    global _askpdf_completed_test_calls
    _askpdf_completed_test_calls = 0


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count tests that reached and completed their call phase."""
    global _askpdf_completed_test_calls
    if report.when == "call" and not report.skipped:
        _askpdf_completed_test_calls += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail an explicitly guarded proof suite when every collected test skipped."""
    enabled = os.getenv("ASKPDF_FAIL_IF_ALL_SKIPPED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    completed = _askpdf_completed_test_calls
    if enabled and session.testscollected > 0 and completed == 0 and exitstatus == 0:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "proof suite failed: every collected test was skipped")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# Faker instance for generating test data
fake = Faker()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Get the test database URL from environment or use default with random database name.
    
    In Docker: Use the postgres service
    Locally: Use localhost postgres
    """
    test_url = os.getenv("TEST_DATABASE_URL")
    if test_url:
        # The Docker test runner creates an isolated database before pytest.
        # Direct pytest invocations (for example, `docker compose run
        # rag-service pytest ...`) do not go through that runner, so make the
        # configured test database available here as well.  This is limited to
        # TEST_DATABASE_URL and never touches the application DATABASE_URL.
        asyncio.run(_ensure_test_database(test_url))
        return test_url
    
    # Otherwise, generate a random database name for local testing
    base_url = "postgresql+asyncpg://postgres:postgres@localhost:5432"
    random_db_name = f"test_askpdf_{uuid.uuid4().hex[:12]}"
    test_url = f"{base_url}/{random_db_name}"
    asyncio.run(_ensure_test_database(test_url))
    return test_url


def _admin_database_url(database_url: str) -> tuple[str, str]:
    normalized = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    parts = urlsplit(normalized)
    database_name = parts.path.lstrip("/")
    if not database_name:
        raise RuntimeError(f"TEST_DATABASE_URL must include a database name: {database_url}")
    admin_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    return admin_url, database_name


async def _ensure_test_database(database_url: str) -> None:
    """Create TEST_DATABASE_URL's database when a direct pytest run needs it.

    `scripts/run_tests.py` owns isolated database lifecycle in Docker.  This
    fallback only creates a missing named database and intentionally does not
    drop it, because a direct invocation may reuse the Compose PostgreSQL
    service across multiple test commands.
    """
    admin_url, database_name = _admin_database_url(database_url)
    conn = await asyncpg.connect(admin_url, timeout=10)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            database_name,
        )
        if exists:
            return
        try:
            await conn.execute(f'CREATE DATABASE "{database_name.replace(chr(34), chr(34) * 2)}"')
        except asyncpg.exceptions.DuplicateDatabaseError:
            # Another test process may have created it between the check and
            # CREATE DATABASE.  It is safe to continue in that case.
            return
    finally:
        await conn.close()


def _build_test_engine(test_database_url: str):
    return create_async_engine(
        test_database_url,
        poolclass=NullPool,
        echo=False,
        future=True
    )


async def _create_test_schema(engine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def _drain_test_loop_tasks() -> None:
    """Allow already-scheduled cleanup callbacks to finish.

    Do not cancel arbitrary tasks here.  A blanket cancellation can interrupt
    an asyncpg command while it is scheduling its own cancellation coroutine,
    which is then destroyed with ``Connection._cancel`` still unawaited.
    Application-owned tasks must be cancelled and awaited by their owning
    lifespan/fixture before this helper runs.
    """
    current = asyncio.current_task()
    for _ in range(6):
        pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
        if not pending:
            return
        await asyncio.sleep(0)


async def _drop_test_schema(engine):
    # Do not call SQLAlchemy's process-global close_all_sessions() here.  The
    # sync TestClient runs on its AnyIO portal loop, while other pytest
    # fixtures may have created sessions on their own loop; the global session
    # registry can therefore attempt to close a transaction from the wrong
    # loop and raise IllegalStateChangeError.  Application repositories own
    # their sessions with async context managers, and the TestClient lifespan
    # has already stopped MCP/workflow tasks before this teardown runs.
    # asyncpg schedules connection-cancel work as a task. Give the owning
    # portal loop a few turns after application/MCP shutdown to run that task
    # before DDL and engine disposal close its loop.
    for _ in range(3):
        await asyncio.sleep(0)
    await _drain_test_loop_tasks()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    for _ in range(3):
        await asyncio.sleep(0)
    await engine.dispose()
    # asyncpg may schedule connection-cancellation cleanup from the dispose
    # itself. Drain the owning loop after disposal, not only before DDL.
    await _drain_test_loop_tasks()


def _build_session_maker(engine):
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )


def _patch_app_session_makers(monkeypatch, session_maker):
    from app.db import connection_sqlmodel
    from app.db.repositories import (
        file_repo_sqlmodel,
        memory_repo_sqlmodel,
        message_repo_sqlmodel,
        project_repo_sqlmodel,
        project_file_repo_sqlmodel,
        stats_repo_sqlmodel,
        thread_file_repo_sqlmodel,
        thread_repo_sqlmodel,
    )
    from app.services import (
        agent_task_repository,
        embedding_materialization_service,
        effective_memory_service,
        memory_manager_engine,
        memory_manager_service,
        memory_representation_service,
        memory_service,
        memory_tool_service,
        memory_workspace_service,
        project_lifecycle_service,
        thread_management_service,
    )
    from app.agent_workflows import (
        chat_cancellation,
        repository as agent_workflow_repository,
    )

    for module in (
        connection_sqlmodel,
        file_repo_sqlmodel,
        memory_repo_sqlmodel,
        message_repo_sqlmodel,
        project_repo_sqlmodel,
        project_file_repo_sqlmodel,
        stats_repo_sqlmodel,
        thread_file_repo_sqlmodel,
        thread_repo_sqlmodel,
        thread_management_service,
        effective_memory_service,
        memory_manager_engine,
        memory_manager_service,
        memory_representation_service,
        memory_service,
        memory_tool_service,
        memory_workspace_service,
        project_lifecycle_service,
        embedding_materialization_service,
        agent_workflow_repository,
        chat_cancellation,
        agent_task_repository,
    ):
        monkeypatch.setattr(module, "async_session_maker", session_maker)

    # Rebuild repository singletons so endpoint tests cannot reuse app-DB sessions.
    import app.db as db_api

    for attr in ("_thread_repo", "_file_repo", "_message_repo", "_thread_file_repo", "_stats_repo", "_agent_workflow_repo", "_project_repo", "_project_file_repo", "_memory_repo"):
        monkeypatch.setattr(db_api, attr, None)


@pytest.fixture(scope="function")
def api_client(test_database_url, monkeypatch) -> Generator:
    """Create a sync FastAPI test client wired to the isolated test database."""
    from fastapi.testclient import TestClient

    engine = _build_test_engine(test_database_url)
    session_maker = _build_session_maker(engine)
    _patch_app_session_makers(monkeypatch, session_maker)

    import main as main_module

    async def idle_task_worker(*, stop_event, **_kwargs):
        await stop_event.wait()

    monkeypatch.setattr(main_module, "run_task_worker", idle_task_worker)
    # The isolated fixture owns test schema creation. Production startup must
    # not call SQLModel.metadata.create_all; migrations are the schema authority.
    asyncio.run(_create_test_schema(engine))
    app = main_module.app

    try:
        with TestClient(app) as test_client:
            yield test_client
            test_client.portal.call(_drop_test_schema, engine)
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def async_api_client(test_database_url, monkeypatch) -> AsyncGenerator[AsyncClient, None]:
    """Create an async FastAPI test client wired to the isolated test database."""
    engine = _build_test_engine(test_database_url)
    session_maker = _build_session_maker(engine)
    await _create_test_schema(engine)
    _patch_app_session_makers(monkeypatch, session_maker)

    import main as main_module

    async def idle_task_worker(*, stop_event, **_kwargs):
        await stop_event.wait()

    monkeypatch.setattr(main_module, "run_task_worker", idle_task_worker)
    app = main_module.app

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        await _drop_test_schema(engine)


@pytest_asyncio.fixture(scope="function")
async def engine(test_database_url: str):
    """
    Create a test database engine.
    
    This fixture is function-scoped to create tables for each test.
    Uses NullPool to avoid connection conflicts between concurrent tests.
    """
    
    engine = _build_test_engine(test_database_url)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine

    # Sessions are owned and closed by their async context managers.  Do not
    # call SQLAlchemy's process-global close_all_sessions() here: pytest-
    # asyncio creates a new loop per test, so the global registry can retain
    # loop-bound sessions and schedule asyncpg cancellation work on a loop
    # that is already being torn down.
    for _ in range(3):
        await asyncio.sleep(0)
    await _drain_test_loop_tasks()

    # Drop all tables after test
    for _ in range(3):
        await asyncio.sleep(0)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    for _ in range(3):
        await asyncio.sleep(0)
    
    await engine.dispose()
    await _drain_test_loop_tasks()


@pytest_asyncio.fixture(scope="function")
async def test_session_maker(engine, monkeypatch):
    """Bind every application repository and service to the current test loop."""

    session_maker = _build_session_maker(engine)
    _patch_app_session_makers(monkeypatch, session_maker)
    return session_maker


@pytest_asyncio.fixture(scope="function")
async def session(engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create a test database session with transaction rollback.
    
    Each test gets a clean session that rolls back at the end,
    ensuring tests don't affect each other.
    """
    
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    
    session = async_session()
    try:
        await session.begin()
        yield session
    finally:
        # Close explicitly on the fixture's owning loop.  Relying only on the
        # async context manager during generator teardown can leave an
        # asyncpg cancellation task behind when pytest is rotating loops.
        if session.in_transaction():
            await session.rollback()
        await session.close()


# Test data fixtures for Thread model
@pytest.fixture
def thread_data():
    """Generate sample thread data."""
    return {
        "name": fake.sentence(nb_words=4),
        "embedding_model": "BAAI/bge-m3",
        "settings": {"replans": 10, "token_budget": 8192}
    }


@pytest_asyncio.fixture
async def sample_thread(session, thread_data):
    """Create a sample thread in the database."""
    
    import uuid
    project = Project(
        id=str(uuid.uuid4()),
        name="Sample Project",
        embedding_model=thread_data["embedding_model"],
    )
    session.add(project)
    thread = Thread(
        id=str(uuid.uuid4()),
        name=thread_data["name"],
        project_id=project.id,
        embedding_model=thread_data["embedding_model"],
        settings=thread_data["settings"],
        created_at=datetime.utcnow()
    )
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    return thread


@pytest_asyncio.fixture
async def test_model_project(session):
    """Create the project required by tests that use the synthetic test model."""

    project = Project(
        id=str(uuid.uuid4()),
        name="Test Model Project",
        embedding_model="test-model",
    )
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


# Test data fixtures for File model
@pytest.fixture
def file_data():
    """Generate sample file data."""
    return {
        "file_hash": fake.sha256(),
        "file_name": f"{fake.word()}.pdf",
        "file_path": f"/data/{fake.word()}.pdf",
        "source_type": "pdf"
    }


@pytest_asyncio.fixture
async def sample_file(session, file_data):
    """Create a sample file in the database."""
    
    file = File(**file_data)
    session.add(file)
    await session.commit()
    await session.refresh(file)
    return file


# Test data fixtures for ChatTurn-backed message compatibility
@pytest.fixture
def message_data(sample_thread):
    """Generate sample chat turn data."""
    return {
        "thread_id": sample_thread.id,
        "role": MessageRole.USER,
        "content": fake.paragraph(nb_sentences=3),
        "context_compact": fake.sentence(),
        "reasoning": None,
        "reasoning_available": False,
        "reasoning_format": "none",
        "web_sources": []
    }


@pytest_asyncio.fixture
async def sample_message(session, message_data):
    """Create a sample chat turn in the database."""
    
    import uuid
    message = ChatTurn(
        id=str(uuid.uuid4()),
        thread_id=message_data["thread_id"],
        status="completed",
        payload={
            "question": message_data["content"],
            "rewritten_question": message_data["context_compact"],
            "answer": None,
            "reasoning": message_data["reasoning"],
            "reasoning_available": message_data["reasoning_available"],
            "reasoning_format": message_data["reasoning_format"],
            "web_sources": message_data["web_sources"],
            "metadata": {},
        },
        created_at=datetime.utcnow()
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


# Test data fixtures for ThreadFile association
@pytest_asyncio.fixture
async def sample_thread_file(session, sample_thread, sample_file):
    """Create a sample thread-file association."""
    
    thread_file = ThreadFile(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        added_at=datetime.utcnow()
    )
    session.add(thread_file)
    await session.commit()
    await session.refresh(thread_file)
    return thread_file


@pytest.fixture
def annotation_data():
    """Generate sample annotation data."""
    return {
        "annotations": [
            {
                "page": 1,
                "bbox": [100, 200, 300, 400],
                "text": fake.sentence(),
                "label": "important"
            }
        ]
    }


@pytest_asyncio.fixture
async def sample_annotation(session, sample_thread, sample_file, annotation_data):
    """Create a sample thread-file association with annotations."""
    annotation = ThreadFile(
        thread_id=sample_thread.id,
        file_hash=sample_file.file_hash,
        added_at=datetime.utcnow(),
        annotations=annotation_data["annotations"],
        annotations_updated_at=datetime.utcnow()
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return annotation


# Test data fixtures for thread stats fields
@pytest_asyncio.fixture
async def sample_thread_stats(session, sample_thread):
    """Populate sample thread stats fields."""
    sample_thread.total_qa_pairs = 5
    sample_thread.total_qa_chars = 1000
    sample_thread.avg_qa_chars = 200.0
    sample_thread.last_qa_at = datetime.utcnow()
    sample_thread.documents_meta = {}
    sample_thread.stats_last_updated_at = datetime.utcnow()
    session.add(sample_thread)
    await session.commit()
    await session.refresh(sample_thread)
    return sample_thread


# Fixture for multiple threads
@pytest_asyncio.fixture
async def multiple_threads(session, thread_data):
    """Create multiple sample threads."""
    
    import uuid
    project = Project(
        id=str(uuid.uuid4()),
        name="Multiple Threads Project",
        embedding_model=thread_data["embedding_model"],
    )
    session.add(project)
    threads = []
    for i in range(3):
        thread = Thread(
            id=str(uuid.uuid4()),
            name=f"{thread_data['name']} {i}",
            project_id=project.id,
            embedding_model=thread_data["embedding_model"],
            settings=thread_data["settings"],
            created_at=datetime.utcnow()
        )
        session.add(thread)
        threads.append(thread)
    
    await session.commit()
    for thread in threads:
        await session.refresh(thread)
    
    return threads


# Fixture for multiple chat turns
@pytest_asyncio.fixture
async def multiple_messages(session, sample_thread):
    """Create multiple sample chat turns in a thread."""
    
    import uuid
    messages = []
    for i in range(5):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        is_user = role == MessageRole.USER
        message = ChatTurn(
            id=str(uuid.uuid4()),
            thread_id=sample_thread.id,
            status="completed",
            payload={
                "question": fake.paragraph(nb_sentences=2) if is_user else "",
                "answer": fake.paragraph(nb_sentences=2) if not is_user else None,
                "metadata": {},
            },
            created_at=datetime.utcnow()
        )
        session.add(message)
        messages.append(message)
    
    await session.commit()
    for message in messages:
        await session.refresh(message)
    
    return messages


# Fixture for file status JSON
@pytest.fixture
def file_status_data():
    """Generate sample file status data."""
    return {
        "parsing": {
            "status": ProcessStatus.COMPLETED.value,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat()
        },
        "indexing": {
            "status": ProcessStatus.COMPLETED.value,
            "chunk_count": 100,
            "total_chars": 50000
        }
    }


# Fixture for parsed sentences JSON
@pytest.fixture
def parsed_sentences_data():
    """Generate sample parsed sentences data."""
    return {
        "sentences": [
            {
                "id": "1",
                "text": fake.sentence(),
                "page": 1,
                "bbox": [0, 0, 100, 20]
            }
        ]
    }
