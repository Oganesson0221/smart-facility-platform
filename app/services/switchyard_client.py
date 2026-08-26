"""Official NeMo Switchyard HTTP integration and routing diagnostics."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from app.config import ROOT, settings
from app.services.openai_compat import build_headers, chat_completions, list_models


LOGGER = logging.getLogger(__name__)
SELECTED_MODEL_HEADER = "x-model-router-selected-model"


class SwitchyardError(RuntimeError):
    """A Switchyard request failed with an actionable diagnostic."""


@dataclass(frozen=True)
class RoutedCompletion:
    payload: dict[str, Any]
    selected_model: str
    decision_sources: tuple[str, ...]
    latency_ms: float
    fallback_used: bool = False


def _server_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _route_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        str(item.get("id") or item.get("name"))
        for item in data
        if isinstance(item, dict) and (item.get("id") or item.get("name"))
    ]


def _decision_counts(stats: object) -> dict[tuple[str, str], int]:
    if not isinstance(stats, dict):
        return {}
    algorithm_stats = stats.get("algorithm_stats", {})
    stage = algorithm_stats.get("stage_router", {}) if isinstance(algorithm_stats, dict) else {}
    decisions = stage.get("routing_decisions", {}) if isinstance(stage, dict) else {}
    result: dict[tuple[str, str], int] = {}
    if not isinstance(decisions, dict):
        return result
    for source, source_data in decisions.items():
        if not isinstance(source_data, dict):
            continue
        targets = source_data.get("targets", {})
        if not isinstance(targets, dict):
            continue
        for target, count in targets.items():
            try:
                result[(str(source), str(target))] = int(count)
            except (TypeError, ValueError):
                continue
    return result


def _decision_delta(
    before: object,
    after: object,
    selected_model: str,
) -> tuple[str, ...]:
    old = _decision_counts(before)
    new = _decision_counts(after)
    sources = {
        source
        for (source, target), count in new.items()
        if target == selected_model and count > old.get((source, target), 0)
    }
    return tuple(sorted(sources))


def _routing_log_path() -> Path:
    path = Path(settings.switchyard_routing_log_path)
    return path if path.is_absolute() else ROOT / path


def latest_routing_record() -> dict[str, Any] | None:
    """Read the newest complete record from Switchyard's official JSONL log."""
    path = _routing_log_path()
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            lines = handle.read().splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        return {
            key: record.get(key)
            for key in (
                "ts",
                "session_id",
                "model",
                "tier",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
            if key in record
        }
    return None


async def _model_endpoint_status(
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        available = await list_models(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=min(8.0, timeout_seconds),
        )
    except httpx.HTTPError as exc:
        return {"reachable": False, "model_available": False, "detail": str(exc)}
    present = model in available
    return {
        "reachable": True,
        "model_available": present,
        "model": model,
        "base_url": base_url,
        "detail": "ready" if present else f"Configured model '{model}' is missing",
    }


class SwitchyardClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = _server_base_url(base_url or settings.switchyard_base_url)
        self.model = model or settings.switchyard_model
        self.api_key = settings.switchyard_api_key if api_key is None else api_key
        self.timeout_seconds = timeout_seconds or settings.switchyard_timeout_seconds
        self.transport = transport

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout or self.timeout_seconds, transport=self.transport)

    async def _get_json(self, path: str) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.get(
                f"{self.base_url}{path}",
                headers=build_headers(self.api_key),
            )
            response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def health(self) -> dict[str, Any]:
        try:
            async with self._client(min(8.0, self.timeout_seconds)) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
            return {"reachable": True, "detail": "ready"}
        except httpx.HTTPStatusError as exc:
            return {
                "reachable": False,
                "detail": f"HTTP {exc.response.status_code} from {self.base_url}/health",
            }
        except httpx.HTTPError as exc:
            return {"reachable": False, "detail": str(exc)}

    async def status(self) -> dict[str, Any]:
        if not settings.switchyard_enabled:
            return {
                "enabled": False,
                "reachable": False,
                "detail": "disabled",
                "route": self.model,
                "ui_url": "/switchyard",
            }
        health, efficient_endpoint, capable_endpoint = await asyncio.gather(
            self.health(),
            _model_endpoint_status(
                settings.llm_base_url,
                settings.llm_api_key,
                settings.llm_model,
                settings.llm_timeout_seconds,
            ),
            _model_endpoint_status(
                settings.vision_base_url,
                settings.vision_api_key,
                settings.vision_model,
                settings.vision_timeout_seconds,
            ),
        )
        result: dict[str, Any] = {
            "enabled": True,
            **health,
            "base_url": self.base_url,
            "route": self.model,
            "ui_url": "/switchyard",
            "latest_routing": latest_routing_record(),
            "efficient_endpoint": efficient_endpoint,
            "capable_endpoint": capable_endpoint,
        }
        if not health["reachable"]:
            return result
        try:
            models = await self._get_json("/v1/models")
            stats = await self._get_json("/v1/stats")
        except (httpx.HTTPError, ValueError) as exc:
            return {**result, "reachable": False, "detail": str(exc)}
        routes = _route_ids(models)
        result.update(
            {
                "routes": routes,
                "route_available": self.model in routes,
                "stats": stats,
                "stage_router": stats.get("algorithm_stats", {}).get("stage_router", {}),
                "detail": "ready" if self.model in routes else f"Route '{self.model}' is missing",
            }
        )
        return result

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> RoutedCompletion:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        for key, value in (
            ("max_tokens", max_tokens),
            ("response_format", response_format),
            ("tools", tools),
            ("tool_choice", tool_choice),
        ):
            if value is not None:
                body[key] = value
        headers = build_headers(self.api_key)
        if session_id:
            headers["x-switchyard-session-id"] = session_id
        started = monotonic()
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:500]
            raise SwitchyardError(
                f"Switchyard returned HTTP {exc.response.status_code}: {detail or 'empty response'}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SwitchyardError(f"Switchyard is unavailable at {self.base_url}: {exc}") from exc
        latency_ms = (monotonic() - started) * 1000
        selected = response.headers.get(SELECTED_MODEL_HEADER, "")
        if not selected:
            raise SwitchyardError(
                f"Switchyard response omitted required {SELECTED_MODEL_HEADER} metadata"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise SwitchyardError("Switchyard returned a non-object JSON response")
        LOGGER.info(
            "[Switchyard] route=%s selected_model=%s latency_ms=%.1f",
            self.model,
            selected,
            latency_ms,
        )
        return RoutedCompletion(payload, selected, (), latency_ms)

    async def diagnose(self, scenario: str) -> RoutedCompletion:
        messages = diagnostic_messages(scenario)
        try:
            before = await self._get_json("/v1/stats")
        except (httpx.HTTPError, ValueError):
            before = {}
        result = await self.chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=32,
            session_id=f"exitwatch-diagnostic-{scenario}",
        )
        try:
            after = await self._get_json("/v1/stats")
        except (httpx.HTTPError, ValueError):
            after = {}
        sources = _decision_delta(before, after, result.selected_model)
        LOGGER.info(
            "[Switchyard] route=%s selected_model=%s decision_source=%s latency_ms=%.1f",
            self.model,
            result.selected_model,
            ",".join(sources) or "not_exposed_for_request",
            result.latency_ms,
        )
        return RoutedCompletion(
            result.payload,
            result.selected_model,
            sources,
            result.latency_ms,
        )


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def diagnostic_messages(scenario: str) -> list[dict[str, Any]]:
    """Build supported tool histories that exercise official Stage Router signals."""
    if scenario == "routine":
        return [
            {"role": "user", "content": "Finish the settled implementation."},
            _tool_call("write-1", "write_file", {"path": "status.txt", "content": "done"}),
            {"role": "tool", "tool_call_id": "write-1", "content": "File written successfully."},
            _tool_call("test-1", "Bash", {"command": "pytest -q"}),
            {"role": "tool", "tool_call_id": "test-1", "content": "5 passed in 0.10s"},
        ]
    if scenario == "exploration":
        return [
            {"role": "user", "content": "Investigate and recover from the failing workflow."},
            _tool_call("read-1", "read_file", {"path": "app.py"}),
            {"role": "tool", "tool_call_id": "read-1", "content": "Read app.py"},
            _tool_call("plan-1", "update_plan", {"step": "inspect failure"}),
            {"role": "tool", "tool_call_id": "plan-1", "content": "Plan updated"},
            _tool_call("read-2", "read_file", {"path": "service.py"}),
            {"role": "tool", "tool_call_id": "read-2", "content": "Read service.py"},
            _tool_call("read-3", "read_file", {"path": "trace.log"}),
            {
                "role": "tool",
                "tool_call_id": "read-3",
                "content": "Traceback (most recent call last): ValueError: invalid workflow state",
            },
        ]
    if scenario == "critical":
        return [
            {"role": "user", "content": "Recover the unavailable local service."},
            _tool_call("health-1", "Bash", {"command": "curl http://127.0.0.1:8010/health"}),
            {
                "role": "tool",
                "tool_call_id": "health-1",
                "content": "curl: connection refused",
            },
        ]
    raise ValueError("scenario must be one of: routine, exploration, critical")


async def routed_text_completion(**kwargs: Any) -> RoutedCompletion:
    """Route a text call through Switchyard, optionally falling back to local Qwen."""
    client = SwitchyardClient()
    try:
        return await client.chat_completion(**kwargs)
    except SwitchyardError:
        if not settings.switchyard_fallback_enabled:
            raise
        payload = await chat_completions(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            model=settings.llm_model,
            messages=kwargs["messages"],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
            response_format=kwargs.get("response_format"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
        )
        LOGGER.warning(
            "[Switchyard] route=%s fallback_model=%s decision_source=local_fallback",
            settings.switchyard_model,
            settings.llm_model,
        )
        return RoutedCompletion(payload, settings.llm_model, ("local_fallback",), 0.0, True)
