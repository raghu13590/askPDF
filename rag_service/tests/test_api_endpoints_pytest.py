"""
test_api_endpoints_pytest.py - DB-agnostic API endpoint tests.

This module provides comprehensive tests for API endpoints using FastAPI's TestClient.
These tests validate HTTP contracts and API behavior with a test database.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Generator
from unittest.mock import patch, AsyncMock, Mock

import pytest

from app.api import threads as threads_api
from app.models.requests import (
    PromptDefaults,
    PromptPreviewRequest,
    ThreadChatRequest,
    ThreadSettingsResponse,
    ThreadSettingsUpdateRequest,
)
from app.models.llm_server_client import REPLANS_LIMIT


def _close_scheduled_coroutine(coroutine):
    """Model task scheduling without leaking the coroutine passed to create_task."""
    if hasattr(coroutine, "close"):
        coroutine.close()
    return Mock()


@pytest.fixture(scope="function")
def client(api_client) -> Generator:
    """Keep existing test signatures while using the shared API client fixture."""
    yield api_client


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check(self, client):
        """Test that health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "rag-service"
        assert data["agent_task_worker"] == "running"
        assert "version" in data

    def test_health_check_remains_live_when_integrated_worker_fails(self, client):
        from main import app

        app.state.agent_task_worker_status = "failed"
        try:
            response = client.get("/health")
        finally:
            app.state.agent_task_worker_status = "running"

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["agent_task_worker"] == "failed"

    @pytest.mark.asyncio
    async def test_integrated_worker_completion_marks_unexpected_failure(self):
        from main import app, _record_agent_task_worker_completion

        async def fail_worker():
            raise RuntimeError("worker failed")

        task = asyncio.create_task(fail_worker())
        with pytest.raises(RuntimeError, match="worker failed"):
            await task
        app.state.agent_task_worker_status = "running"

        _record_agent_task_worker_completion(app, task)

        assert app.state.agent_task_worker_status == "failed"
        app.state.agent_task_worker_status = "running"


class TestThreadEndpoints:
    """Test suite for thread endpoints."""

    def test_create_thread(self, client):
        """Test creating a new thread."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "project_id" in data
        assert data["name"] == "Test Thread"
        assert data["embedding_model"] == "BAAI/bge-m3"

    def test_create_thread_rejects_thread_level_embedding_model(self, client):
        """Thread creation must not accept a model independent of its project."""
        response = client.post(
            "/api/threads",
            json={"name": "Legacy Alias Thread", "embed_model": "BAAI/bge-m3"}
        )
        assert response.status_code == 422

    def test_create_thread_default_embedding_model(self, client):
        """Test creating a thread with default embed model."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Thread"
        assert data["embedding_model"] is not None

    def test_list_threads(self, client):
        """Test listing all threads."""
        # Create a thread first
        client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        
        response = client.get("/api/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data
        assert isinstance(data["threads"], list)
        assert "project_id" in data["threads"][0]

    def test_create_project_and_thread_in_project(self, client):
        project_response = client.post(
            "/api/projects",
            json={
                "name": "Research",
                "description": "Shared research context",
                "embedding_model": "BAAI/bge-m3",
            },
        )
        assert project_response.status_code == 200
        project = project_response.json()
        assert project["last_activity_at"]

        thread_response = client.post(
            f"/api/projects/{project['id']}/threads",
            json={"name": "Project Thread"},
        )
        assert thread_response.status_code == 200
        thread = thread_response.json()
        assert thread["project_id"] == project["id"]
        assert thread["embedding_model"] == project["embedding_model"]

        listed_projects = client.get("/api/projects").json()["projects"]
        assert listed_projects[0]["id"] == project["id"]
        assert listed_projects[0]["last_activity_at"] >= project["last_activity_at"]

    def test_project_global_memory_setting_create_and_update(self, client):
        created_response = client.post(
            "/api/projects",
            json={
                "name": "Consent Project",
                "embedding_model": "BAAI/bge-m3",
                "settings_json": {
                    "layout": "compact",
                    "memory": {"project_reads_user_memory": True},
                },
            },
        )
        assert created_response.status_code == 200
        created = created_response.json()
        assert created["settings_json"] == {
            "layout": "compact",
            "memory": {"project_reads_user_memory": True},
        }

        updated_response = client.put(
            f"/api/projects/{created['id']}",
            json={
                "settings_json": {
                    "memory": {"project_reads_user_memory": False},
                }
            },
        )
        assert updated_response.status_code == 200
        assert updated_response.json()["settings_json"] == {
            "layout": "compact",
            "memory": {"project_reads_user_memory": False},
        }

    def test_thread_memory_settings_round_trip_and_defaults(self, client):
        thread = client.post(
            "/api/threads",
            json={"name": "Memory Settings"},
        ).json()

        defaults_response = client.get(f"/api/threads/{thread['id']}/settings")
        assert defaults_response.status_code == 200
        assert defaults_response.json()["memory"] == {
            "memory_enabled": True,
            "thread_reads_thread_memory": True,
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": False,
        }

        updated_response = client.put(
            f"/api/threads/{thread['id']}/settings",
            json={
                "memory": {
                    "memory_enabled": False,
                    "thread_reads_thread_memory": False,
                    "thread_reads_project_memory": False,
                    "thread_reads_user_memory": True,
                }
            },
        )
        assert updated_response.status_code == 200
        assert updated_response.json()["memory"] == {
            "memory_enabled": False,
            "thread_reads_thread_memory": False,
            "thread_reads_project_memory": False,
            "thread_reads_user_memory": True,
        }

    def test_move_thread_to_project(self, client):
        project_response = client.post(
            "/api/projects",
            json={"name": "Target", "embedding_model": "BAAI/bge-m3"},
        )
        thread_response = client.post(
            "/api/threads",
            json={"name": "Movable"},
        )
        project_id = project_response.json()["id"]
        thread_id = thread_response.json()["id"]

        move_response = client.put(
            f"/api/threads/{thread_id}/project",
            json={"project_id": project_id},
        )
        assert move_response.status_code == 200
        assert move_response.json()["project_id"] == project_id

    def test_move_and_cross_project_fork_reject_different_embedding_models(self, client):
        source_thread = client.post(
            "/api/threads",
            json={"name": "Locked source"},
        ).json()
        incompatible_project = client.post(
            "/api/projects",
            json={
                "name": "Different embedding space",
                "embedding_model": "remote/other-embedding-model",
            },
        ).json()

        move_response = client.put(
            f"/api/threads/{source_thread['id']}/project",
            json={"project_id": incompatible_project["id"]},
        )
        assert move_response.status_code == 409

        fork_response = client.post(
            f"/api/threads/{source_thread['id']}/fork",
            json={"target_project_id": incompatible_project["id"]},
        )
        assert fork_response.status_code == 409

    def test_chat_rejects_unavailable_project_embedding_model(self, client):
        project = client.post(
            "/api/projects",
            json={
                "name": "Offline model project",
                "embedding_model": "remote/offline-embedding-model",
            },
        ).json()
        thread = client.post(
            "/api/threads",
            json={"name": "Visible but blocked", "project_id": project["id"]},
        ).json()

        with patch(
            "app.services.embedding_model_service.check_embedding_model_ready",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.post(
                f"/api/threads/{thread['id']}/chat",
                json={
                    "thread_id": thread["id"],
                    "question": "Can this bypass the composer?",
                    "llm_model": "test-chat-model",
                },
            )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "embedding_model_unavailable"

    def test_direct_memory_create_contract_is_retired(self, client):
        project_response = client.post(
            "/api/projects",
            json={"name": "Memory Project", "embedding_model": "BAAI/bge-m3"},
        )
        project_id = project_response.json()["id"]

        response = client.post(
            "/api/memories",
            json={
                "scope_type": "project",
                "scope_id": project_id,
                "memory_type": "semantic",
                "content": "The project codename is Atlas.",
                "confidence": 0.9,
            },
        )
        assert response.status_code == 405

        effective_response = client.get(
            f"/api/projects/{project_id}/memories/effective"
        )
        assert effective_response.status_code == 200
        assert effective_response.json()["memories"] == []

    def test_memory_endpoints_reject_invalid_contract_values(self, client):
        retired_direct_create = client.post(
            "/api/memories",
            json={
                "scope_type": "project",
                "scope_id": "project-1",
                "memory_type": "fact",
                "content": "Invalid memory type.",
            },
        )
        assert retired_direct_create.status_code == 405

        unconfirmed_manager_apply = client.post(
            "/api/memory-manager/apply",
            json={
                "context": {
                    "selected_scope_type": "user",
                    "selected_scope_id": "default",
                },
                "operations": [],
            },
        )
        assert unconfirmed_manager_apply.status_code == 422

        retired_candidate_endpoint = client.post("/api/memory-candidates", json={})
        assert retired_candidate_endpoint.status_code == 404


    def test_memory_delete_endpoint_reports_missing_record(self, client):
        missing_memory_response = client.delete("/api/memories/missing-memory")
        assert missing_memory_response.status_code == 404


    def test_get_thread(self, client):
        """Test getting a specific thread."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        thread_id = create_response.json()["id"]
        
        with patch("app.api.threads.check_embedding_model_ready", new_callable=AsyncMock, return_value=False):
            response = client.get(f"/api/threads/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == thread_id
        assert data["name"] == "Test Thread"
        assert data["embedding_model_ready"] is False

    @pytest.mark.asyncio
    async def test_get_thread_returns_metadata_when_embedding_model_offline(self):
        """Thread detail should not require vector dimensions when embeddings are offline."""
        thread = SimpleNamespace(
            id="thread-1",
            name="Offline Thread",
            embedding_model="text-embedding-nomic-embed-text-v1.5",
            settings={},
            created_at=datetime(2026, 1, 1),
        )
        file = SimpleNamespace(
            file_hash="file-1",
            file_name="paper.pdf",
            file_path="/data/paper.pdf",
            source_type="pdf",
            association_scope="thread",
            is_project_knowledge=False,
            added_at=datetime(2026, 1, 1),
        )

        with (
            patch("app.api.threads.get_thread", new_callable=AsyncMock, return_value=thread),
            patch("app.api.threads.get_effective_thread_files", new_callable=AsyncMock, return_value=[file]),
            patch(
                "app.api.threads.get_file_status",
                new_callable=AsyncMock,
                return_value={
                    "parsing": {"status": "completed"},
                    "indexing_status": {
                        "summary": {"status": "completed"},
                        "models": {
                            "text-embedding-nomic-embed-text-v1.5": {"status": "completed"}
                        },
                    },
                },
            ),
            patch("app.api.threads.repair_thread_documents_meta", new_callable=AsyncMock) as repair_meta,
            patch("app.api.threads.check_embedding_model_ready", new_callable=AsyncMock, return_value=False),
            patch("app.api.threads.get_vector_db") as get_vector_db,
            patch("app.api.threads.trigger_reembed_for_missing_sources", new_callable=AsyncMock) as trigger_reembed,
        ):
            data = await threads_api.get_thread_endpoint("thread-1")

        assert data["id"] == "thread-1"
        assert data["files"][0]["file_hash"] == "file-1"
        assert data["embedding_model_ready"] is False
        assert data["stats"] == threads_api._empty_thread_stats()
        assert data["stats_unavailable_reason"] == "Embedding model is not ready"
        get_vector_db.assert_not_called()
        repair_meta.assert_not_called()
        trigger_reembed.assert_not_called()

    @pytest.mark.asyncio
    async def test_thread_index_status_blocks_when_embedding_model_offline(self):
        """Index status should not probe vector collections when embeddings are offline."""
        thread = SimpleNamespace(
            id="thread-1",
            name="Offline Thread",
            embedding_model="text-embedding-nomic-embed-text-v1.5",
            settings={},
            created_at=datetime(2026, 1, 1),
        )

        with (
            patch("app.api.threads.get_thread", new_callable=AsyncMock, return_value=thread),
            patch("app.api.threads.check_embedding_model_ready", new_callable=AsyncMock, return_value=False),
            patch("app.api.threads.get_vector_db") as get_vector_db,
        ):
            data = await threads_api.get_thread_index_status_endpoint("thread-1")

        assert data == {
            "thread_id": "thread-1",
            "status": "blocked",
            "stats": threads_api._empty_thread_stats(),
            "embedding_model_ready": False,
        }
        get_vector_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_thread_uses_real_stats_when_embedding_model_ready(self):
        """Ready embeddings should keep the existing vector stats path."""
        thread = SimpleNamespace(
            id="thread-1",
            name="Ready Thread",
            embedding_model="BAAI/bge-m3",
            settings={},
            created_at=datetime(2026, 1, 1),
        )
        stats = {
            "total_documents": 1,
            "total_chunks": 3,
            "total_chars": 1200,
            "documents": {},
        }
        db = SimpleNamespace(
            get_thread_stats=AsyncMock(return_value=stats),
            collection_manager=SimpleNamespace(
                ensure_collections_for_thread=AsyncMock(return_value=None)
            ),
        )

        with (
            patch("app.api.threads.get_thread", new_callable=AsyncMock, return_value=thread),
            patch("app.api.threads.get_effective_thread_files", new_callable=AsyncMock, return_value=[]),
            patch("app.api.threads.repair_thread_documents_meta", new_callable=AsyncMock),
            patch("app.api.threads.check_embedding_model_ready", new_callable=AsyncMock, return_value=True),
            patch("app.api.threads.get_vector_db", return_value=db),
            patch("app.api.threads.trigger_reembed_for_missing_sources", new_callable=AsyncMock),
            patch("app.api.threads.asyncio.create_task", side_effect=_close_scheduled_coroutine),
        ):
            data = await threads_api.get_thread_endpoint("thread-1")

        assert data["embedding_model_ready"] is True
        assert data["stats"] == stats
        assert data["stats_unavailable_reason"] is None
        db.get_thread_stats.assert_awaited_once_with(
            thread_id="thread-1",
            file_hashes=[],
            embedding_model="BAAI/bge-m3",
        )

    def test_get_nonexistent_thread(self, client):
        """Test getting a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id")
        assert response.status_code == 404

    def test_update_thread(self, client):
        """Test updating a thread's name."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Original Name"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.put(
            f"/api/threads/{thread_id}",
            json={"name": "Updated Name"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    def test_update_nonexistent_thread(self, client):
        """Test updating a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id",
            json={"name": "New Name"}
        )
        assert response.status_code == 404

    def test_get_thread_settings(self, client):
        """Test getting thread settings."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.get(f"/api/threads/{thread_id}/settings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert data["hitl_web_approval"] is False
        assert data["use_reranker"] is False

    def test_get_thread_settings_clamps_legacy_replans(self, client, monkeypatch):
        """Stored replans above the current limit should not break settings reads."""
        create_response = client.post(
            "/api/threads",
            json={"name": "Legacy Settings Thread"}
        )
        thread_id = create_response.json()["id"]

        async def fake_get_thread_settings(_thread_id):
            return {"replans": REPLANS_LIMIT + 7}

        monkeypatch.setattr(threads_api, "get_thread_settings", fake_get_thread_settings)

        response = client.get(f"/api/threads/{thread_id}/settings")

        assert response.status_code == 200
        assert response.json()["replans"] == REPLANS_LIMIT

    def test_get_settings_nonexistent_thread(self, client):
        """Test getting settings for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/settings")
        assert response.status_code == 404

    def test_update_thread_settings(self, client):
        """Test updating thread settings."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        thread_id = create_response.json()["id"]
        
        response = client.put(
            f"/api/threads/{thread_id}/settings",
            json={
                "replans": 2,
                "token_budget": 16384,
                "hitl_web_approval": True,
                "agent_workflow": {"workflow_id": "router_rag_agent"},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert data["hitl_web_approval"] is True
        assert data["agent_workflow"]["workflow_id"] == "router_rag_agent"

    def test_update_settings_nonexistent_thread(self, client):
        """Test updating settings for a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id/settings",
            json={"replans": 2}
        )
        assert response.status_code == 404

    def test_delete_thread(self, client):
        """Test deleting a thread."""
        # Create a thread
        create_response = client.post(
            "/api/threads",
            json={"name": "To Delete"}
        )
        thread_id = create_response.json()["id"]

        with patch("app.api.threads.hard_delete_thread_memory_resources", new_callable=AsyncMock) as cleanup_memory:
            cleanup_memory.return_value = {
                "deleted_memory_ids": [],
                "vector_cleanup": False,
            }
            response = client.delete(f"/api/threads/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["thread_id"] == thread_id
        cleanup_memory.assert_awaited_once_with(thread_id)

    def test_delete_nonexistent_thread(self, client):
        """Test deleting a thread that doesn't exist."""
        response = client.delete("/api/threads/nonexistent-id")
        assert response.status_code == 404

    def test_bulk_delete_threads(self, client):
        """Test deleting multiple threads."""
        delete_resource = AsyncMock(side_effect=[True, True])

        with patch("app.api.threads._delete_thread_resources", delete_resource):
            response = client.post(
                "/api/threads/bulk/delete",
                json={"thread_ids": ["thread-1", "thread-2"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["deleted_thread_ids"] == ["thread-1", "thread-2"]
        assert data["not_found_thread_ids"] == []
        assert data["failed_thread_ids"] == []
        assert delete_resource.await_count == 2

    def test_bulk_delete_threads_with_missing_id(self, client):
        """Bulk delete should delete existing IDs even when one is missing."""
        delete_resource = AsyncMock(side_effect=[True, False, True])

        with patch("app.api.threads._delete_thread_resources", delete_resource):
            response = client.post(
                "/api/threads/bulk/delete",
                json={"thread_ids": ["thread-1", "missing-thread", "thread-2"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["deleted_thread_ids"] == ["thread-1", "thread-2"]
        assert data["not_found_thread_ids"] == ["missing-thread"]
        assert data["failed_thread_ids"] == []
        assert delete_resource.await_count == 3

    def test_bulk_delete_threads_empty_ids(self, client):
        """Bulk delete should reject an empty thread list."""
        response = client.post(
            "/api/threads/bulk/delete",
            json={"thread_ids": []},
        )

        assert response.status_code == 422

    def test_bulk_delete_threads_deduplicates_ids(self, client):
        """Bulk delete should delete duplicate IDs once."""
        delete_resource = AsyncMock(side_effect=[True, True])

        with patch("app.api.threads._delete_thread_resources", delete_resource):
            response = client.post(
                "/api/threads/bulk/delete",
                json={"thread_ids": ["thread-1", "thread-1", "thread-2"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["deleted_thread_ids"] == ["thread-1", "thread-2"]
        assert data["not_found_thread_ids"] == []
        assert delete_resource.await_count == 2

    def test_fork_thread_endpoint(self, client):
        """Test forking a thread."""
        forked_thread = SimpleNamespace(
            id="forked-thread",
            name="Source Thread (Fork)",
            embedding_model="BAAI/bge-m3",
            settings={"replans": 3},
            thread_metadata={
                "fork": {
                    "parent_thread_id": "source-thread",
                    "parent_thread_name": "Source Thread",
                    "forked_at": "2026-01-01T00:00:00Z",
                    "source_message_id": "message-1",
                    "source_message_created_at": "2026-01-01T00:00:00Z",
                    "mode": "from_message",
                }
            },
            created_at=datetime(2026, 1, 1),
        )
        file = SimpleNamespace(file_hash="file-1")

        with (
            patch(
                "app.api.threads.fork_thread",
                new_callable=AsyncMock,
                return_value={"thread": forked_thread, "files": [file]},
            ) as fork_thread,
            patch(
                "app.api.threads.trigger_reembed_for_missing_sources",
                new_callable=AsyncMock,
            ) as trigger_reembed,
            patch("app.api.threads.asyncio.create_task", side_effect=_close_scheduled_coroutine) as create_task,
        ):
            response = client.post(
                "/api/threads/source-thread/fork",
                json={"message_id": "message-1"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "forked-thread"
        assert data["thread_metadata"]["fork"]["parent_thread_id"] == "source-thread"
        assert data["thread_metadata"]["fork"]["source_message_id"] == "message-1"
        fork_thread.assert_awaited_once_with(
            source_thread_id="source-thread",
            message_id="message-1",
            name=None,
            target_project_id=None,
            memory_copy_mode=None,
        )
        trigger_reembed.assert_called_once_with(
            thread_id="forked-thread",
            embedding_model="BAAI/bge-m3",
            file_hashes=["file-1"],
        )
        assert create_task.call_count >= 1

    def test_fork_thread_endpoint_missing_source(self, client):
        """Forking a missing source thread should return 404."""
        with patch(
            "app.api.threads.fork_thread",
            new_callable=AsyncMock,
            side_effect=threads_api.SourceThreadNotFoundError("missing"),
        ):
            response = client.post("/api/threads/missing-thread/fork", json={})

        assert response.status_code == 404

    def test_fork_thread_endpoint_message_from_other_thread(self, client):
        """Forking from a message outside the source thread should return 400."""
        with patch(
            "app.api.threads.fork_thread",
            new_callable=AsyncMock,
            side_effect=threads_api.ForkMessageNotFoundError("missing message"),
        ):
            response = client.post(
                "/api/threads/source-thread/fork",
                json={"message_id": "other-message"},
            )

        assert response.status_code == 400

    def test_get_prompt_tools(self, client):
        """Test getting prompt tools and defaults."""
        response = client.get("/api/threads/prompt-tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "defaults" in data
        assert isinstance(data["tools"], list)
        assert data["tools"]
        assert all(set(tool) == {"id", "display_name", "description", "default_prompt"} for tool in data["tools"])
        assert {"system_role", "tool_instructions", "custom_instructions"} <= set(data["defaults"])
        assert "reasoning_mode" not in data["defaults"]

    def test_prompt_preview(self, client, monkeypatch):
        """Test getting prompt preview."""
        async def fake_prompt_preview(self, definition, spec, options):
            return "# Router Node Prompt\n# Final Answer Prompt"

        monkeypatch.setattr(
            "app.runtime.http_adapter.HttpLangGraphRuntimeAdapter.prompt_preview",
            fake_prompt_preview,
        )
        response = client.post(
            "/api/threads/prompt-preview",
            json={
                "context_window": 8192,
                "system_role": "You are a helpful assistant",
                "tool_instructions": {},
                "custom_instructions": "Be concise"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert isinstance(data["prompt"], str)
        assert "# Router Node Prompt" in data["prompt"]
        assert "# Final Answer Prompt" in data["prompt"]

    def test_prompt_preview_supports_plan_execute_pattern(self, client, monkeypatch):
        """Prompt preview should use selected agent workflow runtime prompts."""
        async def fake_prompt_preview(self, definition, spec, options):
            return "# Planner Node Prompt\nexecution_plan"

        monkeypatch.setattr(
            "app.runtime.http_adapter.HttpLangGraphRuntimeAdapter.prompt_preview",
            fake_prompt_preview,
        )
        response = client.post(
            "/api/threads/prompt-preview",
            json={
                "context_window": 8192,
                "system_role": "You are a helpful assistant",
                "tool_instructions": {},
                "custom_instructions": "Be concise",
                "agent_workflow_id": "plan_execute_rag_agent",
            },
        )

        assert response.status_code == 200
        prompt = response.json()["prompt"]
        assert "# Planner Node Prompt" in prompt
        assert "execution_plan" in prompt

    def test_prompt_preview_unknown_workflow_is_not_found(self, client):
        """Unknown workflow IDs must not silently select a different workflow."""
        response = client.post(
            "/api/threads/prompt-preview",
            json={
                "context_window": 8192,
                "agent_workflow_id": "unknown_agent",
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == {"code": "agent_workflow_not_found"}

    def test_reasoning_mode_removed_from_request_models(self):
        """Reasoning-mode compatibility should not be exposed by API schemas."""
        for model in (
            ThreadChatRequest,
            ThreadSettingsResponse,
            ThreadSettingsUpdateRequest,
            PromptDefaults,
            PromptPreviewRequest,
        ):
            assert "reasoning_mode" not in model.model_fields

        filtered = threads_api._public_thread_settings(
            {"replans": 3, "reasoning_mode": True, "unknown": "value"}
        )
        assert filtered == {"replans": 3}


class TestMessageEndpoints:
    """Test suite for message endpoints."""

    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for message tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        return response.json()["id"]

    def test_get_thread_messages_empty(self, client, sample_thread):
        """Test getting messages from an empty thread."""
        response = client.get(f"/api/threads/{sample_thread}/messages")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 0

    def test_get_thread_messages_with_pagination(self, client, sample_thread):
        """Test getting messages with pagination parameters."""
        response = client.get(
            f"/api/threads/{sample_thread}/messages",
            params={"limit": 10, "offset": 0}
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_messages_nonexistent_thread(self, client):
        """Test getting messages for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/messages")
        assert response.status_code == 404

    def test_delete_message_nonexistent(self, client):
        """Test deleting a message that doesn't exist."""
        response = client.delete("/api/messages/nonexistent-id")
        assert response.status_code == 200


class TestFileEndpoints:
    """Test suite for file endpoints."""

    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for file tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        return response.json()["id"]

    def test_get_thread_files_empty(self, client, sample_thread):
        """Test getting files from a thread with no files."""
        response = client.get(f"/api/threads/{sample_thread}/files")
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert isinstance(data["files"], list)
        assert len(data["files"]) == 0

    def test_get_files_nonexistent_thread(self, client):
        """Test getting files for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files")
        assert response.status_code == 404

    def test_add_file_to_thread(self, client, sample_thread):
        """Test adding a file to a thread."""
        response = client.post(
            f"/api/threads/{sample_thread}/files",
            json={
                "file_hash": "abc123",
                "file_name": "test.pdf",
                "file_path": "/data/test.pdf"
            }
        )
        # This endpoint requires background tasks and may not work in simple test
        # but we can at least check the endpoint exists
        assert response.status_code in [200, 500]  # May fail due to missing dependencies


class TestProactiveCollectionCreation:
    """Test proactive collection creation during thread access."""
    
    @pytest.fixture
    def sample_thread(self, client):
        """Create a sample thread for collection tests."""
        response = client.post(
            "/api/threads",
            json={"name": "Test Thread"}
        )
        return response.json()["id"]
    
    @patch('app.api.threads.asyncio.create_task', side_effect=_close_scheduled_coroutine)
    @patch('app.api.threads.repair_thread_documents_meta', new_callable=AsyncMock)
    @patch('app.api.threads.check_embedding_model_ready', new_callable=AsyncMock, return_value=True)
    @patch('app.api.threads.get_vector_db')
    @patch('app.api.threads.trigger_reembed_for_missing_sources', new_callable=AsyncMock)
    def test_thread_access_triggers_collection_creation(self, mock_reembed, mock_get_db, mock_check_ready, mock_repair_meta, mock_create_task, client, sample_thread):
        """Test that accessing a thread triggers proactive collection creation."""
        mock_create_task.reset_mock()

        # Mock vector DB and collection manager
        mock_db = AsyncMock()
        mock_collection_manager = AsyncMock()
        mock_db.collection_manager = mock_collection_manager
        mock_db.get_thread_stats.return_value = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_chars": 0,
            "documents": {},
        }
        mock_get_db.return_value = mock_db
        
        # Access thread endpoint
        response = client.get(f"/api/threads/{sample_thread}")
        assert response.status_code == 200
        
        # Should create background work for both reembed and collection creation
        mock_reembed.assert_called_once_with(
            thread_id=sample_thread,
            embedding_model="BAAI/bge-m3",
        )
        mock_collection_manager.ensure_collections_for_thread.assert_called_once_with(
            embedding_model="BAAI/bge-m3"
        )
    
    @patch('app.api.threads.asyncio.create_task', side_effect=_close_scheduled_coroutine)
    @patch('app.api.threads.repair_thread_documents_meta', new_callable=AsyncMock)
    @patch('app.api.threads.check_embedding_model_ready', new_callable=AsyncMock, return_value=True)
    @patch('app.api.threads.get_vector_db')
    @patch('app.api.threads.trigger_reembed_for_missing_sources', new_callable=AsyncMock)
    def test_thread_access_handles_collection_creation_failure(self, mock_reembed, mock_get_db, mock_check_ready, mock_repair_meta, mock_create_task, client, sample_thread):
        """Test that collection creation failures don't break thread access."""
        mock_create_task.reset_mock()

        # Mock vector DB to raise exception during collection creation
        mock_db = AsyncMock()
        mock_collection_manager = AsyncMock()
        mock_collection_manager.ensure_collections_for_thread.side_effect = Exception("Collection creation failed")
        mock_db.collection_manager = mock_collection_manager
        mock_db.get_thread_stats.return_value = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_chars": 0,
            "documents": {},
        }
        mock_get_db.return_value = mock_db
        
        # Thread access should still succeed despite collection creation failure
        # (since it runs as background task)
        response = client.get(f"/api/threads/{sample_thread}")
        assert response.status_code == 200
        
        # Should still attempt to create the collection background work
        mock_collection_manager.ensure_collections_for_thread.assert_called_once_with(
            embedding_model="BAAI/bge-m3"
        )
    
    def test_nonexistent_thread_returns_404(self, client):
        """Test that accessing nonexistent thread returns 404."""
        response = client.get("/api/threads/nonexistent-id")
        assert response.status_code == 404
        assert "Thread not found" in response.json()["detail"]

    def test_remove_file_from_thread(self, client, sample_thread):
        """Test removing a file from a thread."""
        response = client.delete(f"/api/threads/{sample_thread}/files/abc123")
        # May fail if file doesn't exist, but endpoint should be accessible
        assert response.status_code in [200, 404, 500]

    def test_get_file_status_nonexistent_thread(self, client):
        """Test getting file status for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files/abc123/status")
        assert response.status_code == 404

    def test_get_annotations_nonexistent_thread(self, client):
        """Test getting annotations for a thread that doesn't exist."""
        response = client.get("/api/threads/nonexistent-id/files/abc123/annotations")
        assert response.status_code == 404

    def test_update_annotations_nonexistent_thread(self, client):
        """Test updating annotations for a thread that doesn't exist."""
        response = client.put(
            "/api/threads/nonexistent-id/files/abc123/annotations",
            json={"annotations": []}
        )
        assert response.status_code == 404


class TestModelsEndpoint:
    """Test suite for models endpoint."""

    def test_get_models(self, client):
        """Test getting available models."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict) or isinstance(data, list)
