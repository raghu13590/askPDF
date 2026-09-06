"""Dependency-light startup configuration validation for agent runtimes.

The control plane and both external runtime services import this module.  It
intentionally knows only environment names and validation rules; framework
specific behavior remains in the owning runtime packages.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class RuntimeConfigurationError(RuntimeError):
    """Raised when startup configuration is missing or invalid."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid runtime configuration: " + "; ".join(errors))


_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_HERMES_PINNED_REVISION = "bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894"
LANGGRAPH_LIMIT_NAMES = (
    "DEFAULT_TOKEN_BUDGET",
    "REPLANS_LIMIT",
    "MAX_CUSTOM_INSTRUCTIONS_CHARS",
    "MAX_SYSTEM_ROLE_CHARS",
)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Validated environment values, retained as strings at the boundary."""

    values: Mapping[str, str]

    def get(self, name: str) -> str:
        return self.values[name]


def _raw(name: str, values: Mapping[str, str], seen: frozenset[str] = frozenset()) -> str | None:
    if name in seen:
        raise ValueError(f"{name} has a cyclic environment reference")
    value = values.get(name)
    if value is None:
        return None
    match = _REFERENCE.fullmatch(value.strip())
    if match:
        target = match.group(1)
        if not target.startswith("DEEP_AGENT_"):
            raise ValueError(f"{name} references unsupported environment variable {target}")
        return _raw(target, values, seen | {name})
    if value.strip().startswith("${"):
        raise ValueError(f"{name} has an unresolved environment reference")
    return value


def _required(name: str, values: Mapping[str, str], errors: list[str]) -> str | None:
    value = values.get(name)
    if value is None or not value.strip():
        errors.append(f"{name} is required")
        return None
    return value.strip()


def _positive_int(name: str, values: Mapping[str, str], errors: list[str]) -> int | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    try:
        parsed = int(value, 10)
    except ValueError:
        errors.append(f"{name} must be a positive integer")
        return None
    if parsed <= 0:
        errors.append(f"{name} must be a positive integer")
        return None
    return parsed


def parse_required_positive_int(name: str, value: str | None) -> int:
    """Parse one required positive integer without applying a fallback."""
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    normalized = value.strip().lower()
    if normalized in _TRUE or normalized in _FALSE:
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(normalized, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(name: str, values: Mapping[str, str], errors: list[str]) -> float | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        errors.append(f"{name} must be a positive number")
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        errors.append(f"{name} must be a positive number")
        return None
    return parsed


def parse_bounded_ratio(value: str | float, *, name: str, minimum: float = 0.0, maximum: float = 0.5) -> float:
    """Parse a finite inclusive ratio without silently clamping it."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite ratio between {minimum:g} and {maximum:g}") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be a finite ratio between {minimum:g} and {maximum:g}")
    return parsed


def _jitter_ratio(name: str, values: Mapping[str, str], errors: list[str]) -> float | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    try:
        parsed = parse_bounded_ratio(value, name=name)
    except ValueError as exc:
        errors.append(str(exc))
        return None
    return parsed


def _boolean(name: str, values: Mapping[str, str], errors: list[str]) -> bool | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    errors.append(f"{name} must be a boolean")
    return None


def _url(name: str, values: Mapping[str, str], errors: list[str]) -> str | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        errors.append(f"{name} must be an absolute http:// or https:// URL")
        return None
    return value


def _database_url(name: str, values: Mapping[str, str], errors: list[str]) -> str | None:
    value = _required(name, values, errors)
    if value is None:
        return None
    if urlparse(value).scheme not in {"postgres", "postgresql", "postgresql+asyncpg"}:
        errors.append(f"{name} must be a PostgreSQL database URL")
        return None
    return value


def _deep_agent_budgets(framework: str, values: Mapping[str, str], errors: list[str]) -> None:
    common = {
        "MAX_MODEL_CALLS": 1,
        "MAX_MODEL_TOKENS": 1,
        "MAX_TOOL_CALLS": 1,
        "MAX_ACTIVE_RUNTIME_MS": 1,
        "MAX_DURATION_MS": 1,
        "MAX_OUTPUT_CHARS": 1,
        "MAX_EVENT_COUNT": 1,
        "WAKE_LIMIT_SECONDS": 1,
    }
    if framework == "langgraph":
        common.update({
            "SUBAGENT_TIMEOUT_MS": 1,
            "DISPATCH_TIMEOUT_MS": 1,
            "WORKER_TIMEOUT_MS": 1,
            "WEB_WORKER_TIMEOUT_MS": 1,
        })
    for suffix in common:
        framework_name = f"DEEP_AGENT_{framework.upper()}_{suffix}"
        common_name = f"DEEP_AGENT_{suffix}"
        candidate = values.get(framework_name)
        source = framework_name if candidate is not None else common_name
        try:
            raw = _raw(source, values)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if raw is None or not raw.strip():
            errors.append(f"{framework_name} or {common_name} is required")
            continue
        try:
            parsed = int(raw.strip(), 10)
        except ValueError:
            errors.append(f"{source} (effective {common_name}) must be a positive integer")
            continue
        if parsed <= 0:
            errors.append(f"{source} (effective {common_name}) must be a positive integer")


_RUNTIME_FLOATS = (
    "AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS",
    "AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS",
    "AGENT_RUNTIME_READ_TIMEOUT_SECONDS",
    "AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS",
    "AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS",
    "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS",
    "AGENT_RUNTIME_SHUTDOWN_GRACE_SECONDS",
    "AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS",
    "AGENT_RUNTIME_TERMINAL_CONFIRM_TIMEOUT_SECONDS",
    "AGENT_EVENT_POLL_INTERVAL_SECONDS",
    "AGENT_SSE_HEARTBEAT_INTERVAL_SECONDS",
    "AGENT_CANCELLATION_POLL_INTERVAL_SECONDS",
    "AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS",
    "AGENT_RUNTIME_DEPENDENCY_TIMEOUT_SECONDS",
    "AGENT_RUNTIME_DEPENDENCY_STALE_SECONDS",
    "AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO",
    "AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS",
    "MCP_REQUEST_TIMEOUT_SECONDS",
    "NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS",
    "NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS",
)
_RUNTIME_INTS = (
    "AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS",
    "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_BYTES",
    "AGENT_RUNTIME_RECOVERY_BATCH_SIZE",
)


def validate_runtime_environment(
    *,
    service: str,
    environ: Mapping[str, str] | None = None,
) -> RuntimeEnvironment:
    """Validate configuration required by a control or runtime service."""

    if service not in {"control_plane", "langgraph", "hermes", "hermes_profile"}:
        raise ValueError(f"unsupported runtime configuration service: {service}")
    values = dict(os.environ if environ is None else environ)
    errors: list[str] = []

    # The profile renderer is a one-shot bootstrap job, not an HTTP runtime.
    # Validate only the inputs it consumes so it does not inherit connector,
    # polling, lease, or frontend configuration requirements.
    if service != "hermes_profile":
        for name in _RUNTIME_FLOATS:
            if name == "AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO":
                _jitter_ratio(name, values, errors)
            else:
                _positive_float(name, values, errors)
        for name in _RUNTIME_INTS:
            _positive_int(name, values, errors)
        _positive_int("AGENT_RUNTIME_LEASE_SECONDS", values, errors)
        _positive_int("HERMES_RUNTIME_WORKERS", values, errors) if service == "hermes" else None
        _boolean("AGENT_RUNTIME_RECOVERY_LOOP_ENABLED", values, errors) if service == "langgraph" else None
        _boolean("MCP_OTEL_ENABLED", values, errors)

        if service == "control_plane":
            transport = _required("MCP_TRANSPORT", values, errors)
            allowed_transports = {"in_process", "loopback_http"}
            if transport is not None and transport not in allowed_transports:
                errors.append(f"MCP_TRANSPORT must be 'in_process' or 'loopback_http' for {service}")
            if transport == "loopback_http":
                _url("MCP_LOOPBACK_URL", values, errors)
        else:
            # External runtimes always use the product MCP endpoint.  Reachability
            # is a readiness concern, but missing or malformed configuration is
            # a startup error.
            transport = _required("MCP_TRANSPORT", values, errors)
            loopback_url = _required("MCP_LOOPBACK_URL", values, errors)
            if transport is not None and transport != "loopback_http":
                errors.append(f"MCP_TRANSPORT must be 'loopback_http' for {service}")
            if loopback_url:
                _url("MCP_LOOPBACK_URL", values, errors)

    if service == "hermes_profile":
        provider = _required("HERMES_MODEL_PROVIDER", values, errors)
        if provider is not None and (any(character.isspace() for character in provider) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", provider)):
            errors.append("HERMES_MODEL_PROVIDER must be a nonempty provider identifier")
        context = _positive_int("HERMES_MODEL_CONTEXT_LENGTH", values, errors)
        if context is not None:
            if context < 2048:
                errors.append("HERMES_MODEL_CONTEXT_LENGTH must be at least 2048")
            elif provider is not None and provider.lower() != "lmstudio" and context < 64000:
                errors.append("HERMES_MODEL_CONTEXT_LENGTH must be at least 64000 for the selected Hermes provider")
        secret = _required("HERMES_MCP_CONTEXT_SECRET", values, errors)
        if secret is not None and len(secret) < 32:
            errors.append("HERMES_MCP_CONTEXT_SECRET must contain at least 32 characters")
        _required("API_SERVER_KEY", values, errors)
        _required("HERMES_PROFILE_ROOT", values, errors)
        _positive_int("HERMES_PROFILE_UID", values, errors)
        _positive_int("HERMES_PROFILE_GID", values, errors)
        provider_name = (provider or "").lower()
        if provider_name != "lmstudio" and not values.get("OPENAI_API_KEY", "").strip():
            errors.append("OPENAI_API_KEY is required for the selected Hermes provider")

    if service == "langgraph":
        for name in LANGGRAPH_LIMIT_NAMES:
            _positive_int(name, values, errors)
        _deep_agent_budgets("langgraph", values, errors)
    hermes_enabled = service == "hermes" or (
        service == "control_plane"
        and "hermes" in {item.strip().lower() for item in values.get("COMPOSE_PROFILES", "").split(",") if item.strip()}
    )
    if service == "hermes":
        _deep_agent_budgets("hermes", values, errors)

    if service == "langgraph":
        auth_mode = _required("LLM_AUTH_MODE", values, errors)
        if auth_mode is not None and auth_mode not in {"required", "none"}:
            errors.append("LLM_AUTH_MODE must be 'required' or 'none'")
        keyless_provider = values.get("LLM_KEYLESS_PROVIDER", "").strip().lower()
        if auth_mode == "none":
            if not keyless_provider:
                errors.append("LLM_KEYLESS_PROVIDER is required when LLM_AUTH_MODE=none")
            elif keyless_provider not in {"lmstudio", "ollama", "local"}:
                errors.append("LLM_KEYLESS_PROVIDER must be lmstudio, ollama, or local")
        elif auth_mode == "required" and not values.get("OPENAI_API_KEY", "").strip():
            errors.append("OPENAI_API_KEY is required when LLM_AUTH_MODE=required")
        binding_secret = _required("LANGGRAPH_RUNTIME_BINDING_SECRET", values, errors)
        if binding_secret is not None and len(binding_secret) < 32:
            errors.append("LANGGRAPH_RUNTIME_BINDING_SECRET must contain at least 32 characters")
        runtime_token = _required("LANGGRAPH_RUNTIME_TOKEN", values, errors)
        if runtime_token is not None and len(runtime_token) < 32:
            errors.append("LANGGRAPH_RUNTIME_TOKEN must contain at least 32 characters")
        checkpoint = _required("ASKPDF_AGENT_CHECKPOINTER", values, errors)
        if checkpoint is not None and checkpoint != "postgres":
            errors.append("ASKPDF_AGENT_CHECKPOINTER must be 'postgres' for the external runtime")
        _boolean("ASKPDF_AGENT_CHECKPOINTER_SETUP", values, errors)
        if checkpoint == "postgres":
            _database_url("AGENT_CHECKPOINT_DATABASE_URL", values, errors)
        _database_url("AGENT_RUNTIME_EXECUTION_DATABASE_URL", values, errors)
    if service == "control_plane":
        _url("LANGGRAPH_RUNTIME_URL", values, errors)
        runtime_token = _required("LANGGRAPH_RUNTIME_TOKEN", values, errors)
        if runtime_token is not None and len(runtime_token) < 32:
            errors.append("LANGGRAPH_RUNTIME_TOKEN must contain at least 32 characters")

    if hermes_enabled:
        if service == "hermes":
            _url("HERMES_API_URL", values, errors)
            _url("ASKPDF_MCP_URL", values, errors)
            _url("ASKPDF_MCP_HEALTH_URL", values, errors)
        else:
            _url("HERMES_RUNTIME_URL", values, errors)
        provider = _required("HERMES_MODEL_PROVIDER", values, errors)
        if provider is not None and (any(character.isspace() for character in provider) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", provider)):
            errors.append("HERMES_MODEL_PROVIDER must be a nonempty provider identifier")
        context = _positive_int("HERMES_MODEL_CONTEXT_LENGTH", values, errors)
        if context is not None:
            if context < 2048:
                errors.append("HERMES_MODEL_CONTEXT_LENGTH must be at least 2048")
            elif provider is not None and provider.lower() != "lmstudio" and context < 64000:
                errors.append("HERMES_MODEL_CONTEXT_LENGTH must be at least 64000 for the selected Hermes provider")
        secret = _required("HERMES_MCP_CONTEXT_SECRET", values, errors)
        if secret is not None and len(secret) < 32:
            errors.append("HERMES_MCP_CONTEXT_SECRET must contain at least 32 characters")
        _required("API_SERVER_KEY", values, errors)
        _boolean("ASKPDF_MCP_REQUIRED", values, errors)
        if service == "hermes":
            _required("HERMES_API_TOKEN", values, errors)
        revision = _required("HERMES_UPSTREAM_REVISION", values, errors)
        if revision is not None and revision != _HERMES_PINNED_REVISION:
            errors.append("HERMES_UPSTREAM_REVISION does not match the pinned Hermes revision")
        if service == "hermes":
            for name in (
                "HERMES_RUNTIME_VERSION", "HERMES_RUNTIME_STATE_PATH", "HERMES_PROFILE_ROOT",
                "HERMES_RUNTIME_STORAGE_BACKEND", "HERMES_RUNTIME_EVENT_ID_MODE",
            ):
                _required(name, values, errors)
            for name in ("HERMES_RUN_PROFILE_MAX_AGE_SECONDS", "HERMES_RUN_PROFILE_SWEEP_INTERVAL_SECONDS", "HERMES_PROFILE_UID", "HERMES_PROFILE_GID"):
                _positive_int(name, values, errors)
            storage = values.get("HERMES_RUNTIME_STORAGE_BACKEND", "").strip().lower()
            if storage and storage != "file":
                errors.append("HERMES_RUNTIME_STORAGE_BACKEND must be 'file'")
            event_mode = values.get("HERMES_RUNTIME_EVENT_ID_MODE", "").strip().lower()
            if event_mode and event_mode not in {"durable", "ephemeral"}:
                errors.append("HERMES_RUNTIME_EVENT_ID_MODE must be 'durable' or 'ephemeral'")
        provider_name = (provider or "").lower()
        if provider_name != "lmstudio" and not values.get("OPENAI_API_KEY", "").strip():
            errors.append("OPENAI_API_KEY is required for the selected Hermes provider")

    if errors:
        raise RuntimeConfigurationError(sorted(set(errors)))
    resolved_values = dict(values)
    if service in {"langgraph", "hermes"}:
        for name in values:
            if name.startswith("DEEP_AGENT_"):
                resolved = _raw(name, values)
                if resolved is not None:
                    resolved_values[name] = resolved.strip()
    return RuntimeEnvironment(resolved_values)
