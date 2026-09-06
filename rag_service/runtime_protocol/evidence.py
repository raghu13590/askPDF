"""Shared bounded evidence normalization for control plane and runtimes."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from runtime_protocol.contracts import RuntimeEvidenceKind, RuntimeInheritedEvidence

INHERITED_EVIDENCE_CONTENT_LIMIT = 24_000
TOOL_EVIDENCE_PREVIEW_LIMIT = 2_000
EXPLICIT_GAP_WARNINGS = frozenset({"missing_document_vectors", "missing_thread_context", "no_relevant_content", "no_relevant_conversation_history", "no_relevant_memory", "no_thread_documents", "no_usable_web_results", "web_search_disabled", "search_web_failed"})

def _bounded(value: Any, limit: int = INHERITED_EVIDENCE_CONTENT_LIMIT) -> str:
    return str(value or "").strip()[:limit]

def _source_text(sources: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for source in sources:
        text = _bounded(source.get("text") or source.get("snippet"), 1_000)
        if text:
            title = _bounded(source.get("title") or source.get("file_name") or source.get("url"), 300)
            lines.append(f"[{title or 'Source'}]\n{text}")
    return "\n\n".join(lines)[:INHERITED_EVIDENCE_CONTENT_LIMIT]

def _packet(kind: RuntimeEvidenceKind, content: Any, *, sources: Iterable[Mapping[str, Any]] = (), provenance: Mapping[str, Any] | None = None, warnings: Iterable[str] = (), explicit_gap: bool = False) -> RuntimeInheritedEvidence:
    normalized_sources = tuple(dict(item) for item in sources if isinstance(item, Mapping))[:100]
    normalized_content = _bounded(content) or _source_text(normalized_sources)
    identity = hashlib.sha256(f"{kind.value}\0{normalized_content}\0{len(normalized_sources)}".encode()).hexdigest()[:24]
    return RuntimeInheritedEvidence(packet_id=f"inherited:{kind.value}:{identity}", kind=kind, content=normalized_content, sources=normalized_sources, warnings=tuple(dict.fromkeys(str(item) for item in warnings if str(item).strip())), provenance=dict(provenance or {}), available=bool(normalized_content or normalized_sources) and not explicit_gap, explicit_gap=explicit_gap)

def inherited_evidence_packets(prefetch: Mapping[str, Any] | None, *, profile_id: str | None = None) -> tuple[RuntimeInheritedEvidence, ...]:
    bundle = dict(prefetch or {})
    packets = {
        RuntimeEvidenceKind.DOCUMENT: _packet(RuntimeEvidenceKind.DOCUMENT, bundle.get("document_evidence_text"), sources=bundle.get("document_sources") or bundle.get("documents") or (), provenance={"origin": "prefetch", "inventory": list(bundle.get("documents") or [])[:100]}),
        RuntimeEvidenceKind.WEB: _packet(RuntimeEvidenceKind.WEB, bundle.get("web_evidence_text"), sources=bundle.get("web_sources") or (), provenance={"origin": "prefetch"}),
        RuntimeEvidenceKind.CONVERSATION: _packet(RuntimeEvidenceKind.CONVERSATION, "\n\n".join(value for value in (_bounded(bundle.get("recent_history_text"), 12_000), _bounded(bundle.get("semantic_history_text"), 12_000)) if value), sources=bundle.get("semantic_memory_refs") or bundle.get("recent_message_refs") or (), provenance={"origin": "prefetch"}),
        RuntimeEvidenceKind.MEMORY: _packet(RuntimeEvidenceKind.MEMORY, bundle.get("durable_memory_text"), sources=bundle.get("durable_memory_refs") or (), provenance={"origin": "prefetch"}),
    }
    scopes = {"document_researcher": (RuntimeEvidenceKind.DOCUMENT,), "web_researcher": (RuntimeEvidenceKind.WEB,), "memory_researcher": (RuntimeEvidenceKind.CONVERSATION, RuntimeEvidenceKind.MEMORY), "evidence_critic": tuple(RuntimeEvidenceKind)}
    return tuple(packets[kind] for kind in scopes.get(str(profile_id or ""), tuple(RuntimeEvidenceKind)) if packets[kind].available or packets[kind].explicit_gap)

def tool_result_evidence(value: Mapping[str, Any]) -> RuntimeInheritedEvidence:
    trace = value.get("trace") if isinstance(value.get("trace"), Mapping) else {}
    name = str(trace.get("tool_name") or "")
    warnings = tuple(str(item) for item in value.get("warnings") or [])
    sources = value.get("sources") if isinstance(value.get("sources"), list) else []
    error = value.get("error") if isinstance(value.get("error"), Mapping) else {}
    gap = bool(error.get("evidence_gap") or (not sources and warnings and set(warnings).issubset(EXPLICIT_GAP_WARNINGS)))
    kind = RuntimeEvidenceKind.WEB if name in {"search_web", "internet_search"} else RuntimeEvidenceKind.CONVERSATION if name in {"search_thread_conversation_history", "search_thread_events"} else RuntimeEvidenceKind.MEMORY if name in {"search_durable_memory", "memory_search", "memory_get"} else RuntimeEvidenceKind.DOCUMENT
    return _packet(kind, value.get("content"), sources=sources, warnings=warnings, provenance={"origin": "tool", "tool_name": name or None, "tool_call_id": trace.get("tool_call_id")}, explicit_gap=gap)

def evidence_event_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    content = str(value.get("content") or "")
    sources = value.get("sources") if isinstance(value.get("sources"), list) else []
    warnings = [str(item) for item in value.get("warnings") or []]
    error = value.get("error") if isinstance(value.get("error"), Mapping) else {}
    return {"result_chars": len(content), "source_count": len(sources), "warnings": warnings, "explicit_gap": bool(error.get("evidence_gap") or (not sources and warnings and set(warnings).issubset(EXPLICIT_GAP_WARNINGS))), **({"result_preview": content[:TOOL_EVIDENCE_PREVIEW_LIMIT]} if content else {})}
