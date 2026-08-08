from __future__ import annotations

import base64
from typing import Any

import httpx


def normalize_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def build_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def build_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def multimodal_user_message(prompt: str, image_bytes: bytes) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": build_data_url(image_bytes)}},
            ],
        }
    ]


def coerce_text_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text).strip())
                continue
            nested = item.get("content")
            if nested:
                parts.append(str(nested).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def extract_chat_message_content(payload: dict[str, Any]) -> Any:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            for key in ("content", "text"):
                value = message.get(key)
                if value not in (None, ""):
                    return value
    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if value not in (None, ""):
                return value
    for key in ("response", "content"):
        value = payload.get(key)
        if value not in (None, ""):
            return value
    raise KeyError("Chat completion response did not include assistant content")


def _filter_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


async def list_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> list[str]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(
            f"{normalize_base_url(base_url)}/models",
            headers=build_headers(api_key),
        )
        response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id") or item.get("name")
        if identifier:
            names.append(str(identifier))
    return names


def list_models_sync(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> list[str]:
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(
            f"{normalize_base_url(base_url)}/models",
            headers=build_headers(api_key),
        )
        response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id") or item.get("name")
        if identifier:
            names.append(str(identifier))
    return names


async def chat_completions(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _filter_none(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,
        }
    )
    if extra_body:
        payload.update(extra_body)
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{normalize_base_url(base_url)}/chat/completions",
            json=payload,
            headers=build_headers(api_key),
        )
        response.raise_for_status()
    return response.json()


def chat_completions_sync(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _filter_none(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,
        }
    )
    if extra_body:
        payload.update(extra_body)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            f"{normalize_base_url(base_url)}/chat/completions",
            json=payload,
            headers=build_headers(api_key),
        )
        response.raise_for_status()
    return response.json()
