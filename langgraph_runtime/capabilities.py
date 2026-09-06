"""Exact LangGraph deployment and definition capability profiles."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Mapping

from runtime_protocol.contracts import (
    AgentDefinition,
    RuntimeCapabilitySemantics,
    RuntimeCancellationMode,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilities,
    RuntimeConfirmationMode,
    RuntimeFeatureId,
    RuntimeFeatureDescriptor,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeSupportLevel,
    RuntimeTerminalState,
    conditional,
    native,
    unsupported,
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def checkpoint_database_url(environ: Mapping[str, str] | None = None) -> str:
    values = environ or os.environ
    url = values.get("AGENT_CHECKPOINT_DATABASE_URL") or ""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    return url


@dataclass(frozen=True)
class LangGraphDeploymentProfile:
    runtime_mode: str
    checkpointer_backend: str
    checkpoint_available: bool
    durable_persistence: bool
    runtime_available: bool
    configuration_error: str | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "LangGraphDeploymentProfile":
        values = environ or os.environ
        runtime_mode = "external_service"

        backend = str(values.get("ASKPDF_AGENT_CHECKPOINTER") or "").strip().lower()
        if backend == "postgres":
            url_available = bool(checkpoint_database_url(values))
            saver_available = _module_available("langgraph.checkpoint.postgres.aio")
            available = url_available and saver_available
            error = None
            if not url_available:
                error = "LangGraph Postgres checkpointer requires a database URL"
            elif not saver_available:
                error = "LangGraph Postgres checkpointer is unavailable"
            return cls(
                runtime_mode=runtime_mode,
                checkpointer_backend=backend,
                checkpoint_available=available,
                durable_persistence=available,
                runtime_available=available,
                configuration_error=error,
            )
        return cls(
            runtime_mode=runtime_mode,
            checkpointer_backend=backend,
            checkpoint_available=False,
            durable_persistence=False,
            runtime_available=False,
            configuration_error=f"Unsupported ASKPDF_AGENT_CHECKPOINTER value: {backend!r}",
        )

    def deployment_metadata(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "checkpointer_backend": self.checkpointer_backend,
            "checkpoint_available": self.checkpoint_available,
            "durable_persistence": self.durable_persistence,
            "runtime_available": self.runtime_available,
            "configuration_error": self.configuration_error,
        }


def _feature(
    enabled: bool,
    *,
    semantics: RuntimeCapabilitySemantics,
    details: Mapping[str, Any] | None = None,
) -> RuntimeFeatureDescriptor:
    return RuntimeFeatureDescriptor(
        RuntimeSupportLevel.NATIVE if enabled else RuntimeSupportLevel.UNSUPPORTED,
        enabled,
        disabled_reason=None if enabled else RuntimeCapabilityDisabledReason.DEFINITION_CAPABILITY_UNAVAILABLE,
        semantics=semantics,
        details=dict(details or {}),
    )


def _deep_agents_features(definition: AgentDefinition) -> dict[RuntimeFeatureId, RuntimeFeatureDescriptor]:
    metadata = definition.definition_metadata
    node_types = {str(value) for value in metadata.get("graph_node_types", [])}
    tools = {str(value) for value in metadata.get("allowed_tool_ids", [])}
    profiles = {str(value) for value in metadata.get("task_profiles", [])}
    features = definition.capabilities
    is_deep = (
        definition.category == "deep"
        or definition.definition_id == "deep_research_agent"
        or "deep_research_subagent" in node_types
    )
    if not is_deep:
        return {}
    planning = bool(features.get("supports_replans")) or "deep_task_planner" in node_types
    parallel = bool(features.get("supports_parallel_dispatch")) or "parallel_dispatch" in node_types
    artifacts = bool(features.get("supports_artifacts"))
    subagents = "deep_research_subagent" in node_types and bool(profiles)
    memory = "durable_memory" in tools
    return {
        RuntimeFeatureId.PLANNING: _feature(planning, semantics=RuntimeCapabilitySemantics.DEFINITION_PLANNER_NODES),
        RuntimeFeatureId.PARALLEL_DISPATCH: _feature(parallel, semantics=RuntimeCapabilitySemantics.DEFINITION_PARALLEL_DISPATCH),
        RuntimeFeatureId.ARTIFACTS: _feature(artifacts, semantics=RuntimeCapabilitySemantics.DEFINITION_ARTIFACT_POLICY),
        RuntimeFeatureId.SUBAGENT_ORCHESTRATION: _feature(
            subagents,
            semantics=RuntimeCapabilitySemantics.PRODUCT_MANAGED_SUBAGENTS,
            details={"profiles": sorted(profiles)},
        ),
        RuntimeFeatureId.MEMORY: _feature(memory, semantics=RuntimeCapabilitySemantics.DEFINITION_TOOL_POLICY, details={"tool_id": "durable_memory"}),
        RuntimeFeatureId.TOOLS: _feature(bool(tools), semantics=RuntimeCapabilitySemantics.DEFINITION_TOOL_POLICY, details={"count": len(tools)}),
    }


def langgraph_definition_features(definition: AgentDefinition) -> dict[RuntimeFeatureId, RuntimeFeatureDescriptor]:
    """Return definition-owned Deep Agent features for central reconciliation."""

    return _deep_agents_features(definition)


def langgraph_deployment_capabilities(
    *,
    profile: LangGraphDeploymentProfile | None = None,
) -> RuntimeCapabilities:
    """Return deployment declarations without definition or run policy."""

    return langgraph_capabilities(None, profile=profile)


def langgraph_capabilities(
    definition: AgentDefinition | None,
    *,
    profile: LangGraphDeploymentProfile | None = None,
) -> RuntimeCapabilities:
    profile = profile or LangGraphDeploymentProfile.from_environment()
    checkpoint = profile.checkpoint_available
    deployment_reason = (
        RuntimeCapabilityDisabledReason.RUNTIME_CONFIGURATION_INVALID
        if profile.configuration_error
        else RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE
    )

    def enabled_descriptor(descriptor: RuntimeOperationDescriptor) -> RuntimeOperationDescriptor:
        if profile.runtime_available:
            return descriptor
        return RuntimeOperationDescriptor(
            descriptor.support, descriptor.owner, False,
            disabled_reason=deployment_reason,
            modes=descriptor.modes,
            semantics=descriptor.semantics,
            confirmation=descriptor.confirmation,
            terminal_states=descriptor.terminal_states,
            preserves_run_id=descriptor.preserves_run_id,
            preserves_session_id=descriptor.preserves_session_id,
            requires_runtime_binding=descriptor.requires_runtime_binding,
        )

    operations: dict[RuntimeOperationId, RuntimeOperationDescriptor] = {
        RuntimeOperationId.RUN_START: enabled_descriptor(native()),
        RuntimeOperationId.RUN_CANCEL: enabled_descriptor(native(
            modes=(RuntimeCancellationMode.INTERRUPT,),
            confirmation=RuntimeConfirmationMode.ASYNCHRONOUS,
            terminal_states=(RuntimeTerminalState.CANCELLED, RuntimeTerminalState.INTERRUPTED),
        )),
        RuntimeOperationId.RUN_RESUME: conditional(
            enabled=checkpoint,
            semantics=RuntimeCapabilitySemantics.RESUME_FROM_INTERRUPT,
            disabled_reason=None if checkpoint else RuntimeCapabilityDisabledReason.CHECKPOINT_STORE_UNAVAILABLE,
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_INSPECT_STATE: conditional(
            enabled=checkpoint,
            semantics=RuntimeCapabilitySemantics.CHECKPOINT_STATE_INSPECTION,
            disabled_reason=None if checkpoint else RuntimeCapabilityDisabledReason.CHECKPOINT_STORE_UNAVAILABLE,
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_CONTINUATION_CLEANUP: conditional(
            enabled=checkpoint,
            semantics=RuntimeCapabilitySemantics.CHECKPOINT_THREAD_CLEANUP,
            disabled_reason=None if checkpoint else RuntimeCapabilityDisabledReason.CHECKPOINT_STORE_UNAVAILABLE,
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_APPROVAL_RESPOND: unsupported(),
        RuntimeOperationId.RUN_STEER_LIVE: unsupported(),
        RuntimeOperationId.RUN_SEND_FOLLOWUP: unsupported(),
        RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT: unsupported(),
        RuntimeOperationId.RUN_REPLAY: unsupported(),
        RuntimeOperationId.RUN_FORK: unsupported(),
        RuntimeOperationId.SUBAGENT_LIST: unsupported(),
        RuntimeOperationId.SUBAGENT_SEND: unsupported(),
        RuntimeOperationId.SUBAGENT_CANCEL: unsupported(),
    }
    return RuntimeCapabilities(
        operations=operations,
        features=_deep_agents_features(definition) if definition is not None else {},
        deployment=profile.deployment_metadata(),
        behavior={
            "continuation_semantics": "same_run_safe_boundary",
            "usage_accounting_owner": "runtime",
            "preserves_run_id": True,
            "artifact_inheritance": "valid_artifacts",
            "supports_orchestration_delta": True,
            "required_input_fields": ["task_context", "resolved_spec", "embedding_model"],
            "supports_pause_resume": bool(checkpoint),
            "supports_course_correction": bool(checkpoint),
            "budget_boundary_owner": "runtime",
            "grounding_owner": "runtime",
        },
    )
