import base64
import json
import re
from typing import Any

import httpx

from app.config import settings
from app.services.cv.types import Detection


SCENE_PROMPT = """Inspect this facility image using only visible evidence.
The image is accompanied by grounded YOLO detections. Use those detections as
the only source for object labels and object boxes. Do not invent new object
labels or bounding boxes. You may mention signs or doors in the evidence and
summary, but only the detected YOLO objects may appear in supporting_objects.

Return one JSON object with exactly these fields:
- violation: boolean
- category: short string
- summary: one clear sentence
- evidence: array of short visible observations
- confidence: number from 0 to 1
- visible_objects: array of important object names
- supporting_objects: array of detected object labels that directly support the
  decision. Use only labels present in the provided YOLO detections.
"""

FIRE_EXIT_VALIDATION_PROMPT = """Inspect this cropped facility image using only visible evidence.
Decide whether the visible object is obstructing an emergency exit or a required
fire-exit clearance zone. Do not assume obstruction unless the image supports it.

Return one JSON object with exactly these fields:
- confirmed: boolean
- category: short string
- summary: one clear sentence
- visible_evidence: array of short visible observations
- confidence: number from 0 to 1
"""


_THINKING_PREFIX = re.compile(r"^\s*(?:<think>.*?</think>\s*)+", re.DOTALL)


def _strip_json_wrappers(content: str) -> str:
    raw = _THINKING_PREFIX.sub("", content.strip()).strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return raw


def _coerce_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        result = content
    else:
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if text is not None:
                    parts.append(str(text).strip())
                    continue
                nested = item.get("content")
                if nested is not None:
                    parts.append(str(nested).strip())
            content = "\n".join(part for part in parts if part)
        if not isinstance(content, str):
            raise ValueError("Vision model did not return JSON content")
        result = json.loads(_strip_json_wrappers(content))
    if not isinstance(result, dict):
        raise ValueError("Vision model did not return a JSON object")
    return result


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        separator = "\n" if "\n" in text else ";"
        if separator in text:
            return [part.strip(" -") for part in text.split(separator) if part.strip(" -")]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [str(value).strip()]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "confirmed", "violation"}:
            return True
        if normalized in {"false", "no", "0", "clear"}:
            return False
    return default


def _normalize_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(max(0.0, min(1000.0, x1 / max(width, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, y1 / max(height, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, x2 / max(width, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, y2 / max(height, 1) * 1000)), 2),
    ]


def _serialize_scene_detections(
    detections: list[Detection] | None,
    image_shape: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not detections or not image_shape:
        return []
    height, width = image_shape
    return [
        {
            "label": item.label,
            "confidence": round(float(item.confidence), 3),
            "box": _normalize_box(item.box, width, height),
        }
        for item in detections[:12]
    ]


def _extract_chat_content(payload: dict[str, Any]) -> Any:
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
    raise KeyError("Vision response did not include assistant content")


async def _available_vision_models(client: httpx.AsyncClient) -> list[str]:
    try:
        response = await client.get(f"{settings.vision_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return []

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if name:
            names.append(str(name))
    return names


async def vision_runtime_status() -> dict[str, Any]:
    if not settings.vision_enabled:
        return {"enabled": False, "reachable": False, "model_available": False}

    timeout = min(5.0, settings.vision_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{settings.vision_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            models = await _available_vision_models(client)
    except httpx.HTTPError as exc:
        return {
            "enabled": True,
            "reachable": False,
            "model_available": False,
            "detail": str(exc),
            "available_models": [],
        }

    return {
        "enabled": True,
        "reachable": True,
        "model_available": settings.vision_model in models,
        "available_models": models,
        "detail": (
            "ready"
            if settings.vision_model in models
            else f"Configured model '{settings.vision_model}' is not installed"
        ),
    }


async def _call_vision_model(prompt: str, image_bytes: bytes) -> Any:
    payload = {
        "model": settings.vision_model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "options": {"temperature": 0},
    }

    async with httpx.AsyncClient(timeout=settings.vision_timeout_seconds) as client:
        try:
            response = await client.post(
                f"{settings.vision_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return _extract_chat_content(response.json())
        except httpx.HTTPStatusError as exc:
            body = exc.response.text.strip()
            detail = body
            try:
                error_payload = exc.response.json()
                if isinstance(error_payload, dict) and error_payload.get("error"):
                    detail = str(error_payload["error"])
            except (ValueError, json.JSONDecodeError):
                pass

            if exc.response.status_code == 404 and settings.vision_model in detail:
                available_models = await _available_vision_models(client)
                available_suffix = (
                    f" Available models: {', '.join(available_models)}."
                    if available_models
                    else ""
                )
                raise RuntimeError(
                    f"Configured vision model '{settings.vision_model}' is not installed on "
                    f"the local Ollama endpoint at '{settings.vision_base_url}'."
                    f"{available_suffix}"
                ) from exc
            raise RuntimeError(
                f"Local vision endpoint '{settings.vision_base_url}' returned HTTP "
                f"{exc.response.status_code}: {detail[:220]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Local vision endpoint '{settings.vision_base_url}' is unreachable"
            ) from exc


def _parse_assessment(content: Any) -> dict[str, Any]:
    result = _coerce_json_object(content)
    evidence = _coerce_text_list(result.get("evidence"))
    visible_objects = _coerce_text_list(result.get("visible_objects"))
    supporting_objects = _coerce_text_list(result.get("supporting_objects"))
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
    annotations = []
    annotation_source = result.get("annotations", [])
    if isinstance(annotation_source, dict):
        annotation_source = [annotation_source]
    for item in annotation_source[:8]:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            coordinates = [max(0.0, min(1000.0, float(value))) for value in box]
        except (TypeError, ValueError):
            continue
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        annotations.append(
            {"label": str(item.get("label") or "evidence"), "box": coordinates}
        )
    return {
        "violation": _coerce_bool(result.get("violation", False)),
        "category": str(result.get("category", "General")),
        "summary": str(result.get("summary", "No assessment was returned.")),
        "evidence": [str(item) for item in evidence[:8]],
        "confidence": confidence,
        "visible_objects": [str(item) for item in visible_objects[:12]],
        "supporting_objects": [str(item) for item in supporting_objects[:8]],
        "annotations": annotations,
        "model": settings.vision_model,
        "local": True,
    }


def _parse_fire_exit_validation(content: Any) -> dict[str, Any]:
    result = _coerce_json_object(content)
    visible_evidence = _coerce_text_list(result.get("visible_evidence"))
    return {
        "confirmed": _coerce_bool(result.get("confirmed", False)),
        "category": str(result.get("category", "fire_exit_obstruction")),
        "summary": str(result.get("summary", "No validation summary returned.")),
        "visible_evidence": [str(item) for item in visible_evidence[:8]],
        "confidence": max(0.0, min(1.0, float(result.get("confidence", 0)))),
        "model": settings.vision_model,
        "local": True,
    }


async def assess_scene(
    image_bytes: bytes,
    detections: list[Detection] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not settings.vision_enabled:
        raise RuntimeError("Automatic scene reasoning is disabled")

    prompt = (
        f"{SCENE_PROMPT}\n\n"
        "YOLO detections (JSON):\n"
        f"{json.dumps(_serialize_scene_detections(detections, image_shape))}"
    )
    try:
        return _parse_assessment(await _call_vision_model(prompt, image_bytes))
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' returned an invalid response"
        ) from exc


async def validate_fire_exit_obstruction(image_bytes: bytes) -> dict[str, Any]:
    if not settings.vision_enabled:
        raise RuntimeError("Automatic scene reasoning is disabled")

    try:
        return _parse_fire_exit_validation(
            await _call_vision_model(FIRE_EXIT_VALIDATION_PROMPT, image_bytes)
        )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' returned an invalid response"
        ) from exc
