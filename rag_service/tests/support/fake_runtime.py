"""Small HTTP runtime double for control-plane tests.

This deliberately implements the wire contract, rather than importing any
LangGraph runtime code. Tests can override individual responses with
``responses`` while still exercising the real HTTP adapter and serializers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from runtime_protocol.protocol import versioned_payload


class FakeRuntimeServer:
    def __init__(self, responses: Mapping[str, Any] | None = None) -> None:
        self.responses = dict(responses or {})
        self.requests: list[httpx.Request] = []
        self.events: dict[str, list[dict[str, Any]]] = {}

    def _default(self, request: httpx.Request, body: Mapping[str, Any]) -> dict[str, Any]:
        path = request.url.path
        if path == "/v1/capabilities":
            return versioned_payload({"capabilities": {"operations": {}, "features": {}, "deployment": {}}})
        if path == "/v1/validate":
            return versioned_payload({"validation": {"valid": True, "issues": [], "diagnostics": {}}})
        if path == "/v1/resolve":
            return versioned_payload({"resolved_spec": dict(body.get("spec") or {})})
        if path == "/v1/catalog":
            return versioned_payload({"catalog": {"framework": "langgraph", "builder_id": "langgraph_graph"}})
        if path == "/v1/prompt-preview":
            return versioned_payload({"prompt": "fake runtime prompt"})
        if path.endswith("/events"):
            return versioned_payload({"events": self.events.get(str(body.get("run_id") or request.url.path), [])})
        if path.endswith("/inspect"):
            return versioned_payload({"state": {}})
        if path.endswith("/pause"):
            return versioned_payload({"result": {"status": "pause_requested"}})
        if path.endswith("/cancel"):
            return versioned_payload({"result": {"status": "cancelled"}})
        if path.endswith("/continue") or path.endswith("/resume") or path.endswith("/retry"):
            return versioned_payload({"result": {"status": "completed", "output": {}}})
        if path.endswith("/start"):
            return versioned_payload({"result": {"status": "completed", "output": {}}})
        return versioned_payload({"result": {}})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            body = json.loads(request.content or b"{}")
        except (TypeError, ValueError):
            body = {}
        response = self.responses.get(request.url.path)
        if callable(response):
            response = response(request, body)
        if response is None:
            response = self._default(request, body)
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response, request=request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url="http://fake-langgraph-runtime")
