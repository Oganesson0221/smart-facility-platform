import json
import logging
import re
from time import perf_counter
from typing import Any

import cv2
import httpx
import numpy as np

from app.config import settings
from app.services.cv.types import Detection
from app.services.nemo_agent_client import orchestrate_fire_exit_validation
from app.services.nemo_agent_client import orchestrate_scene_assessment
from app.services.openai_compat import chat_completions
from app.services.openai_compat import coerce_text_content
from app.services.openai_compat import extract_chat_message_content
from app.services.openai_compat import list_models
from app.services.openai_compat import multimodal_user_message


LOGGER = logging.getLogger(__name__)
_VEHICLE_IDENTIFIER_LABELS = {"vehicle", "car", "truck", "bus", "van"}


SCENE_PROMPT = (
    "Inspect this facility image using only visible evidence and the supplied YOLO detections. "
    "Use YOLO labels and boxes as the only allowed detected object references. "
    "Return JSON only with keys: violation, category, summary, evidence, confidence, "
    "visible_objects, supporting_objects. Keep summary short and evidence to at most 3 items."
)


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


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        qualitative = {
            "very high": 0.95,
            "high": 0.9,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.3,
            "very low": 0.1,
        }
        if normalized in qualitative:
            return qualitative[normalized]
        if normalized.endswith("%"):
            try:
                value = float(normalized[:-1]) / 100.0
            except ValueError:
                value = default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


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
    return extract_chat_message_content(payload)


async def _available_vision_models(client: httpx.AsyncClient) -> list[str]:
    del client
    try:
        return await list_models(
            base_url=settings.vision_base_url,
            api_key=settings.vision_api_key,
            timeout_seconds=min(5.0, settings.vision_timeout_seconds),
        )
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return []


async def vision_runtime_status() -> dict[str, Any]:
    if not settings.vision_enabled:
        return {"enabled": False, "reachable": False, "model_available": False}

    timeout = min(5.0, settings.vision_timeout_seconds)
    try:
        models = await list_models(
            base_url=settings.vision_base_url,
            api_key=settings.vision_api_key,
            timeout_seconds=timeout,
        )
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


async def _call_vision_model(
    prompt: str,
    image_bytes: bytes,
    *,
    timeout_seconds: float | None = None,
    max_response_tokens: int | None = None,
) -> Any:
    started = perf_counter()
    response_tokens = max(
        48,
        int(
            max_response_tokens
            if max_response_tokens is not None
            else settings.vision_validation_max_response_tokens
        ),
    )
    timeout = timeout_seconds if timeout_seconds is not None else settings.vision_timeout_seconds
    try:
        payload: dict[str, Any] = {}
        token_budgets = (response_tokens, max(512, response_tokens * 2))
        for attempt, token_budget in enumerate(token_budgets, start=1):
            LOGGER.info(
                "Vision request model=%s prompt_chars=%d image_bytes=%d response_tokens=%d attempt=%d",
                settings.vision_model,
                len(prompt),
                len(image_bytes),
                token_budget,
                attempt,
            )
            payload = await chat_completions(
                base_url=settings.vision_base_url,
                api_key=settings.vision_api_key,
                timeout_seconds=timeout,
                model=settings.vision_model,
                messages=multimodal_user_message(prompt, image_bytes),
                temperature=0,
                max_tokens=token_budget,
                response_format={"type": "json_object"},
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": settings.vision_enable_thinking,
                    }
                },
            )
            choices = payload.get("choices", [])
            finish_reason = (
                choices[0].get("finish_reason")
                if isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                else None
            )
            if finish_reason not in {"length", "max_tokens"}:
                break
            if attempt < len(token_budgets):
                LOGGER.warning(
                    "Vision response was truncated at %d tokens; retrying with %d tokens",
                    token_budget,
                    token_budgets[attempt],
                )
        else:
            raise RuntimeError(
                f"Local vision model '{settings.vision_model}' exhausted the structured "
                f"response token budget ({token_budgets[-1]} tokens)"
            )
        LOGGER.info(
            "Vision response model=%s elapsed_ms=%.2f",
            settings.vision_model,
            (perf_counter() - started) * 1000,
        )
        return _extract_chat_content(payload)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        detail = body
        try:
            error_payload = exc.response.json()
            if isinstance(error_payload, dict) and error_payload.get("error"):
                detail = str(error_payload["error"])
        except (ValueError, json.JSONDecodeError):
            pass

        if exc.response.status_code in {400, 404} and settings.vision_model in detail:
            available_models = await list_models(
                base_url=settings.vision_base_url,
                api_key=settings.vision_api_key,
                timeout_seconds=min(5.0, timeout),
            )
            available_suffix = (
                f" Available models: {', '.join(available_models)}."
                if available_models
                else ""
            )
            raise RuntimeError(
                f"Configured vision model '{settings.vision_model}' is not available on "
                f"the local OpenAI-compatible endpoint at '{settings.vision_base_url}'."
                f"{available_suffix}"
            ) from exc
        raise RuntimeError(
            f"Local vision endpoint '{settings.vision_base_url}' returned HTTP "
            f"{exc.response.status_code}: {detail[:220]}"
        ) from exc
    except httpx.TimeoutException as exc:
        LOGGER.warning(
            "Vision request timed out model=%s elapsed_ms=%.2f timeout_seconds=%.2f",
            settings.vision_model,
            (perf_counter() - started) * 1000,
            float(timeout),
        )
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' timed out after "
            f"{float(timeout):.1f} seconds"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Local vision endpoint '{settings.vision_base_url}' is unreachable"
        ) from exc


def _build_fire_exit_validation_prompt(object_label: str | None = None) -> str:
    label = str(object_label or "unknown object").strip() or "unknown object"
    include_vehicle_identifier = label.lower() in _VEHICLE_IDENTIFIER_LABELS
    base = (
        "Inspect this facility exit-clearance candidate image. "
        f"Primary YOLO object: {label}. "
        "YOLO and SAM have already established that the object lies inside the configured exit-clearance zone; the whole image is that zone when no polygon was drawn. "
        "Decide whether the named object is visibly present and occupies or could obstruct the emergency-exit path or required clearance area. "
        "Reject only when the object is absent, visibly misclassified, or clearly unrelated to exit clearance. "
        "Use only visible evidence. If uncertain, set confirmed to false. "
        "Return JSON only. Keep summary short and visible_evidence to at most 3 items."
    )
    if include_vehicle_identifier:
        return (
            f"{base} Use keys: confirmed, category, summary, visible_evidence, confidence, "
            "vehicle_identifier, vehicle_identifier_type, vehicle_identifier_confidence. "
            "Return an empty vehicle_identifier with type none when nothing legible is visible."
        )
    return (
        f"{base} Use keys: confirmed, category, summary, visible_evidence, confidence."
    )


def _prepare_vision_image_bytes(image_bytes: bytes) -> bytes:
    try:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error:
        return image_bytes
    if image is None or image.size == 0:
        return image_bytes
    height, width = image.shape[:2]
    max_dim = max(1, int(settings.vision_validation_image_max_dim))
    longest = max(height, width)
    if longest > max_dim:
        scale = max_dim / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        int(max(30, min(100, settings.vision_validation_jpeg_quality))),
    ]
    ok, encoded = cv2.imencode(".jpg", image, params)
    return encoded.tobytes() if ok else image_bytes


def _parse_assessment(content: Any) -> dict[str, Any]:
    result = _coerce_json_object(content)
    evidence = _coerce_text_list(result.get("evidence"))
    visible_objects = _coerce_text_list(result.get("visible_objects"))
    supporting_objects = _coerce_text_list(result.get("supporting_objects"))
    confidence = _coerce_confidence(result.get("confidence", 0))
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
    vehicle_identifier = str(result.get("vehicle_identifier") or "").strip()
    vehicle_identifier_type = str(result.get("vehicle_identifier_type") or "").strip()
    vehicle_identifier_confidence = result.get("vehicle_identifier_confidence")
    return {
        "confirmed": _coerce_bool(result.get("confirmed", False)),
        "category": str(result.get("category", "fire_exit_obstruction")),
        "summary": str(result.get("summary", "No validation summary returned.")),
        "visible_evidence": [str(item) for item in visible_evidence[:8]],
        "confidence": _coerce_confidence(result.get("confidence", 0)),
        "vehicle_identifier": vehicle_identifier,
        "vehicle_identifier_type": (
            vehicle_identifier_type
            or "unspecified"
            if vehicle_identifier
            else "none"
        ),
        "vehicle_identifier_confidence": (
            _coerce_confidence(vehicle_identifier_confidence)
            if vehicle_identifier_confidence not in (None, "")
            else None
        ),
        "model": settings.vision_model,
        "local": True,
    }


async def assess_scene_direct(
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
    image_bytes = _prepare_vision_image_bytes(image_bytes)
    try:
        return _parse_assessment(
            await _call_vision_model(
                prompt,
                image_bytes,
                timeout_seconds=settings.vision_timeout_seconds,
            )
        )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' returned an invalid response"
        ) from exc


async def validate_fire_exit_obstruction_direct(
    image_bytes: bytes,
    object_label: str | None = None,
) -> dict[str, Any]:
    if not settings.vision_enabled:
        raise RuntimeError("Automatic scene reasoning is disabled")

    prompt = _build_fire_exit_validation_prompt(object_label)
    image_bytes = _prepare_vision_image_bytes(image_bytes)
    try:
        return _parse_fire_exit_validation(
            await _call_vision_model(
                prompt,
                image_bytes,
                timeout_seconds=settings.vision_validation_timeout_seconds,
                max_response_tokens=settings.vision_validation_max_response_tokens,
            )
        )
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' returned an invalid response"
        ) from exc


async def assess_scene(
    image_bytes: bytes,
    detections: list[Detection] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if settings.nemo_agent_enabled and settings.nemo_agent_orchestrate_vision:
        try:
            return await orchestrate_scene_assessment(image_bytes, detections, image_shape)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "NeMo scene workflow unavailable; falling back to direct vision path: %s",
                exc,
            )
    return await assess_scene_direct(image_bytes, detections, image_shape)


async def validate_fire_exit_obstruction(
    image_bytes: bytes,
    object_label: str | None = None,
) -> dict[str, Any]:
    if settings.nemo_agent_enabled and settings.nemo_agent_orchestrate_vision:
        try:
            return await orchestrate_fire_exit_validation(image_bytes, object_label)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "NeMo fire-exit validation unavailable; falling back to direct vision path: %s",
                exc,
            )
    return await validate_fire_exit_obstruction_direct(image_bytes, object_label)
