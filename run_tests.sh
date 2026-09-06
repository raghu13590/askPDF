#!/bin/bash
# run_tests.sh - Docker-native test runner wrapper for askPDF.
#
# Usage:
#   ./run_tests.sh                          # Run frontend tests, all pytest tests, plus standalone checks
#   ./run_tests.sh --unit                   # Run unit and mock-based tests
#   ./run_tests.sh --db                     # Run PostgreSQL database tests
#   ./run_tests.sh --api                    # Run API endpoint tests
#   ./run_tests.sh --integration            # Run integration tests
#   ./run_tests.sh --agent-checkpoint       # Run Postgres checkpoint/resume hardening test
#   ./run_tests.sh --langgraph-runtime                # Run isolated LangGraph runtime integration checks
#   ./run_tests.sh --external-runtime                 # Alias for isolated external runtime checks
#   ./run_tests.sh --langgraph-runtime-real           # Run LangGraph runtime against a configured real provider
#   ./run_tests.sh --hermes-runtime                 # Run deterministic Hermes runtime proof
#   ./run_tests.sh --schema                 # Run schema validation tests
#   ./run_tests.sh --standalone             # Run standalone proactive collection script
#   ./run_tests.sh --frontend               # Run frontend tests only
#   ./run_tests.sh --api --strict-warnings   # Fail on coroutine and Pydantic warnings
#   ./run_tests.sh --file test_api_integration_pytest.py --test TestAPIIntegration::test_create_thread_endpoint
#
# Environment:
#   ASKPDF_TEST_PROJECT_NAME=askpdf-test    # Override isolated Compose project name
#   ASKPDF_KEEP_TEST_CONTAINERS=1           # Keep test containers/volumes for debugging

set -e

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed or not on PATH"
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker-compose)
else
    echo "Error: docker compose or docker-compose is not installed"
    exit 1
fi

TEST_PROJECT_NAME="${ASKPDF_TEST_PROJECT_NAME:-askpdf-test}"
COMPOSE_ARGS=(-p "$TEST_PROJECT_NAME" -f docker-compose.test.yml)
EXTERNAL_RUNTIME_PROJECT_NAME="${ASKPDF_RUNTIME_TEST_PROJECT_NAME:-${ASKPDF_EXTERNAL_RUNTIME_PROJECT_NAME:-askpdf-runtime-integration-test-$$}}"
EXTERNAL_RUNTIME_COMPOSE_ARGS=(-p "$EXTERNAL_RUNTIME_PROJECT_NAME" -f docker-compose.runtime-integration.yml)

args=("$@")
for arg in "${args[@]}"; do
    if [ "$arg" = "--langgraph-runtime" ] || [ "$arg" = "--external-runtime" ]; then
        RUN_LANGGRAPH_RUNTIME=1
    fi
    if [ "$arg" = "--langgraph-runtime-real" ]; then
        RUN_LANGGRAPH_RUNTIME=1
        RUN_LANGGRAPH_RUNTIME_REAL=1
    fi
    if [ "$arg" = "--hermes-runtime" ]; then
        RUN_HERMES_RUNTIME=1
    fi
done

cleanup() {
    if [ "${ASKPDF_KEEP_TEST_CONTAINERS:-}" = "1" ]; then
        echo "Keeping test containers and volumes for project '$TEST_PROJECT_NAME'"
        return
    fi

    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    if [ "${RUN_LANGGRAPH_RUNTIME:-0}" = "1" ] || [ "${RUN_HERMES_RUNTIME:-0}" = "1" ]; then
        "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" down --volumes --remove-orphans || true
    fi
}

trap cleanup EXIT

run_frontend_tests() {
    echo "Running frontend tests..."
    "${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm frontend-test-runner
}

external_runtime_diagnostics() {
    echo "External runtime failed; collecting bounded service diagnostics..." >&2
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" ps || true
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" logs --tail=200 \
        rag-service langgraph-runtime db-migrate runtime-db-migrate \
        postgresql runtime-checkpoint-db-init fake-llm weaviate || true
}

external_runtime_test() {
    if ! "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm --no-deps "$@"; then
        external_runtime_diagnostics
        return 1
    fi
}

wait_for_external_job() {
    local service="$1"
    local label="$2"
    for attempt in $(seq 1 120); do
        local container_id
        container_id=$("${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" ps --all -q "$service" 2>/dev/null || true)
        if [ -n "$container_id" ]; then
            local state
            state=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$container_id" 2>/dev/null || true)
            if [ "$state" = "exited 0" ]; then
                return 0
            fi
            if [[ "$state" = "exited "* ]] || [[ "$state" = "dead "* ]]; then
                echo "$label failed: $state" >&2
                external_runtime_diagnostics
                return 1
            fi
        fi
        sleep 1
    done
    echo "$label timed out after 120 seconds" >&2
    external_runtime_diagnostics
    return 1
}

if [ "${RUN_LANGGRAPH_RUNTIME:-0}" = "1" ]; then
    trap external_runtime_diagnostics ERR
    if [ "${RUN_LANGGRAPH_RUNTIME_REAL:-0}" = "1" ]; then
        if [ -z "${LLM_API_URL:-}" ]; then
            echo "--langgraph-runtime-real requires LLM_API_URL" >&2
            exit 1
        fi
        export EXTERNAL_RUNTIME_LLM_API_URL="$LLM_API_URL"
    else
        export EXTERNAL_RUNTIME_LLM_API_URL="http://fake-llm:9000/v1"
    fi
    echo "Starting isolated LangGraph runtime Compose environment '$EXTERNAL_RUNTIME_PROJECT_NAME'..."
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" build rag-service langgraph-runtime
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" up -d postgresql db-migrate runtime-checkpoint-db-init runtime-db-migrate weaviate fake-llm rag-service langgraph-runtime
    wait_for_external_job db-migrate "Product database migration"
    wait_for_external_job runtime-checkpoint-db-init "Runtime checkpoint database initialization"
    wait_for_external_job runtime-db-migrate "Runtime database migration"
    control_plane_ready=0
    for attempt in $(seq 1 120); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T rag-service python -c \
            'import json, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)); assert health["status"] == "ok"' 2>/dev/null; then
            control_plane_ready=1
            break
        fi
        sleep 1
    done
    if [ "$control_plane_ready" != "1" ]; then
        echo "Control plane readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    runtime_started=0
    for attempt in $(seq 1 120); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; startup=json.load(urllib.request.urlopen("http://127.0.0.1:8100/startupz", timeout=5)); assert startup["status"] == "ok"' 2>/dev/null; then
            runtime_started=1
            break
        fi
        sleep 1
    done
    if [ "$runtime_started" != "1" ]; then
        echo "LangGraph runtime startup readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    runtime_ready=0
    for attempt in $(seq 1 120); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, urllib.request; ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=5)); assert ready["status"] == "ok"' 2>/dev/null; then
            runtime_ready=1
            break
        fi
        sleep 1
    done
    if [ "$runtime_ready" != "1" ]; then
        echo "LangGraph runtime work readiness timed out after 120 seconds" >&2
        external_runtime_diagnostics
        exit 1
    fi
    echo "Verifying the immutable production control-plane image..."
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T rag-service python -c \
        'import importlib.util; from runtime_protocol.contracts import AgentDefinition; from app.runtime.registry import RuntimeRegistry; assert importlib.util.find_spec("langgraph") is None; registry=RuntimeRegistry(); registry.initialize(); definition=AgentDefinition(definition_id="router_rag_agent", framework="langgraph", builder_id="langgraph_graph"); adapter=registry.get(definition); assert adapter.__class__.__name__ == "HttpLangGraphRuntimeAdapter" and adapter.framework == "langgraph"'
    external_runtime_test runtime-test-runner
    external_runtime_test test-runner --file test_runtime_contracts_pytest.py
    external_runtime_test test-runner --file test_runtime_http_adapter_pytest.py
    if [ "${RUN_LANGGRAPH_RUNTIME_REAL:-0}" = "1" ]; then
        if [ -z "${EXTERNAL_RUNTIME_LLM_MODEL:-}" ]; then
            echo "--langgraph-runtime-real requires EXTERNAL_RUNTIME_LLM_MODEL" >&2
            exit 1
        fi
        external_runtime_test -e EXTERNAL_RUNTIME_SMOKE=true -e EXTERNAL_RUNTIME_LLM_MODEL="$EXTERNAL_RUNTIME_LLM_MODEL" test-runner --file test_external_runtime_smoke_pytest.py
    else
        external_runtime_test -e EXTERNAL_RUNTIME_SMOKE=true -e EXTERNAL_RUNTIME_LLM_MODEL=external_runtime-deterministic test-runner --file test_external_runtime_smoke_pytest.py
    fi
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_execution_pytest.py runtime-test-runner
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_lifecycle_pytest.py runtime-test-runner
    external_runtime_test test-runner --file test_agent_runtime_reconciliation_pytest.py
    external_runtime_test test-runner --file test_control_plane_import_boundary_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
        'import json, urllib.error, urllib.request
try: urllib.request.urlopen("http://127.0.0.1:8100/v1/dependencies", timeout=3); raise AssertionError("protected runtime endpoint admitted an anonymous request")
except urllib.error.HTTPError as exc: body=json.load(exc); assert exc.code == 401 and body["error"]["code"] == "runtime_unauthorized"'
    echo "Verifying dependency outage isolation and admission recovery..."
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" stop rag-service fake-llm
    dependencies_degraded=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, os, urllib.error, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); ready_status=200
try: json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3))
except urllib.error.HTTPError as exc: ready_status=exc.code
protected=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(protected, timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and ready_status == 503; assert dependencies["mcp"]["state"] in {"degraded", "unavailable"} and dependencies["provider"]["state"] in {"degraded", "unavailable"}'; then
            dependencies_degraded=1
            break
        fi
        sleep 1
    done
    if [ "$dependencies_degraded" != "1" ]; then
        echo "Runtime dependencies did not become unavailable and readiness did not fail closed" >&2
        exit 1
    fi
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
        'import json, os, urllib.error, urllib.request; payload={"operation_id":"external-runtime-dependency-outage:start","request":{"run_id":"external_runtime-dependency-outage","thread_id":"external_runtime-thread","definition_id":"router_rag_agent","framework":"langgraph","builder_id":"langgraph_graph","input":{"question":"test"},"options":{"llm_model":"external_runtime-deterministic","embedding_model":"external_runtime-deterministic-embedding"}},"context":{"embedding_model":"external_runtime-deterministic-embedding","resolved_spec":{"config":{"allowed_tool_ids":["document_evidence"]}}}}; request=urllib.request.Request("http://127.0.0.1:8100/v1/runs/start", data=json.dumps(payload).encode(), headers={"content-type":"application/json", "Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}, method="POST");
payload["protocol_version"] = "1.4"; payload["minimum_compatible_version"] = "1.4"; payload["request"]["protocol_version"] = "1.4"; payload["request"]["minimum_compatible_version"] = "1.4"; request.data = json.dumps(payload).encode()
try: urllib.request.urlopen(request, timeout=3); raise AssertionError("dependent run was admitted")
except urllib.error.HTTPError as exc: body=json.load(exc); assert exc.code == 503 and body["error"]["code"] == "runtime_dependency_unavailable" and body["error"]["retryable"] is True'
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" start fake-llm rag-service
    dependencies_available=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, os, urllib.request; request=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(request, timeout=3))["result"]["dependencies"]; assert dependencies["mcp"]["state"] == "available" and dependencies["provider"]["state"] == "available"'; then
            dependencies_available=1
            break
        fi
        sleep 1
    done
    if [ "$dependencies_available" != "1" ]; then
        echo "Runtime dependency monitor did not recover after services restarted" >&2
        exit 1
    fi
    echo "Restarting langgraph-runtime to verify readiness and checkpoint service continuity..."
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" stop langgraph-runtime
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" start langgraph-runtime
    runtime_ready=0
    for attempt in $(seq 1 45); do
        if "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" exec -T langgraph-runtime python -c \
            'import json, os, urllib.request; health=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=3)); startup=json.load(urllib.request.urlopen("http://127.0.0.1:8100/startupz", timeout=3)); ready=json.load(urllib.request.urlopen("http://127.0.0.1:8100/readyz", timeout=3)); request=urllib.request.Request("http://127.0.0.1:8100/v1/dependencies", headers={"Authorization": "Bearer " + os.environ["LANGGRAPH_RUNTIME_TOKEN"]}); dependencies=json.load(urllib.request.urlopen(request, timeout=3))["result"]["dependencies"]; assert health["status"] == "ok" and startup["status"] == "ok" and ready["status"] == "ok"; assert dependencies["mcp"]["state"] == "available" and dependencies["mcp"]["protocol"] == "mcp"; assert dependencies["provider"]["state"] == "available"; print(json.dumps({"health": health, "startup": startup, "ready": ready, "dependencies": dependencies}, sort_keys=True))'; then
            runtime_ready=1
            break
        fi
        sleep 2
    done
    if [ "$runtime_ready" -ne 1 ]; then
        echo "Runtime readiness did not recover after restart" >&2
        exit 1
    fi
    echo "Verifying execution recovery after restart and lease expiry..."
    external_runtime_test -e RUN_RUNTIME_DB_MIGRATIONS=true -e AGENT_RUNTIME_RECOVERY_LOOP_ENABLED=true -e RUNTIME_TEST_TARGET=/app/langgraph_runtime/tests/test_runtime_service_lifecycle_pytest.py runtime-test-runner --test test_recovery_loop_reclaims_a_lease_after_restart
    trap - ERR
    exit 0
fi

if [ "${RUN_HERMES_RUNTIME:-0}" = "1" ]; then
    HERMES_RUNTIME_RECOVERY_RUN_ID="${HERMES_RUNTIME_RECOVERY_RUN_ID:-hermes-runtime-recovery-$$}"
    export HERMES_RUNTIME_RECOVERY_RUN_ID
    export HERMES_RUNTIME_COMPOSE_PROFILES=hermes
    export HERMES_RUNTIME_INTEGRATION=true
    echo "Starting deterministic Hermes runtime Hermes runtime proof..."
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" build rag-service
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" up -d postgresql runtime-checkpoint-db-init weaviate db-migrate fake-llm rag-service hermes hermes-runtime
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_runtime_mcp_contract_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_builder_provider_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm test-runner --file test_hermes_execution_store_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm \
        -e HERMES_RUNTIME_SMOKE=true \
        -e HERMES_MODEL=hermes-runtime-deterministic-hermes \
        -e HERMES_RUNTIME_PRODUCT_DATABASE_URL=postgresql://postgres:postgres@postgresql:5432/askpdf \
        test-runner --file test_external_hermes_runtime_smoke_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm \
        -e HERMES_RUNTIME_REAL_SMOKE=true \
        -e HERMES_MODEL=hermes-runtime-deterministic-hermes \
        -e HERMES_RUNTIME_URL=http://hermes-runtime:8200 \
        test-runner --file test_real_hermes_container_smoke_pytest.py
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm \
        -e HERMES_RUNTIME_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e HERMES_RUNTIME_RECOVERY_RUN_ID="$HERMES_RUNTIME_RECOVERY_RUN_ID" \
        test-runner --file test_hermes_runtime_restart_pytest.py --test test_seed_restart_recovery_record
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" restart hermes-runtime
    "${DOCKER_COMPOSE[@]}" "${EXTERNAL_RUNTIME_COMPOSE_ARGS[@]}" run --rm \
        -e HERMES_RUNTIME_INTEGRATION=true \
        -e ASKPDF_FAIL_IF_ALL_SKIPPED=true \
        -e HERMES_RUNTIME_RECOVERY_RUN_ID="$HERMES_RUNTIME_RECOVERY_RUN_ID" \
        test-runner --file test_hermes_runtime_restart_pytest.py --test test_recovered_run_reconnects_without_another_upstream_start
    exit 0
fi

backend_args=()
run_frontend=0
frontend_only=0

if [ "$#" -eq 0 ]; then
    run_frontend=1
else
    for arg in "${args[@]}"; do
        case "$arg" in
            --frontend)
                run_frontend=1
                frontend_only=1
                ;;
            --all|--all-tests)
                run_frontend=1
                backend_args+=("$arg")
                ;;
            *)
                backend_args+=("$arg")
                ;;
        esac
    done
fi

if [ "$run_frontend" = "1" ]; then
    run_frontend_tests
fi

if [ "$frontend_only" = "1" ] && [ "${#backend_args[@]}" -eq 0 ]; then
    exit 0
fi

"${DOCKER_COMPOSE[@]}" "${COMPOSE_ARGS[@]}" run --rm --build test-runner "${backend_args[@]}"
