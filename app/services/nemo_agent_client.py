from __future__ import annotations

import cv2
import json
import re
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np

from app.config import ROOT, settings
from app.services.cv.types import Detection
from app.services.openai_compat import chat_completions
from app.services.openai_compat import chat_completions_sync
from app.services.openai_compat import extract_chat_message_content


_JSON_PREFIX = re.compile(r"^\s*(?:```json|```)?\s*", re.IGNORECASE)
_JSON_SUFFIX = re.compile(r"\s*```+\s*$")

_SCENE_WORKFLOW_PROMPT = """You are the Smart Facility NeMo scene orchestration workflow.
Do not infer from the image directly. Use tools.
If yolo_detections_json is empty, call detect_image_objects exactly once with the supplied image_path.
Then call inspect_scene_image exactly once.
Return only the final JSON object for the scene assessment with no markdown."""

_FIRE_EXIT_VALIDATION_PROMPT = """You are the Smart Facility NeMo fire-exit validation workflow.
Do not infer from the image directly. Use tools.
Call validate_fire_exit_image exactly once with the supplied image_path and object_label.
Return only the validation JSON object with no markdown."""

_DETECTION_WORKFLOW_PROMPT = """You are the Smart Facility NeMo detection workflow.
Do not infer from the image directly. Use tools.
Call detect_image_objects exactly once with the supplied image_path and confidence_threshold.
Return only the detector JSON object with no markdown."""

_SEGMENTATION_WORKFLOW_PROMPT = """You are the Smart Facility NeMo segmentation workflow.
Do not infer from the image directly. Use tools.
Call segment_image_box exactly once with the supplied image_path and box_json.
Return only the segmentation JSON object with no markdown."""

_UNAVAILABLE_UNTIL: dict[str, float] = {
    "scene": 0.0,
    "validation": 0.0,
    "detect": 0.0,
    "segment": 0.0,
}
_FAILURE_BACKOFF_SECONDS = 15.0


def _stage_image(image_bytes: bytes, prefix: str) -> str:
    folder = ROOT / "uploads" / "nemo_staging"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}-{uuid4().hex}.jpg"
    path.write_bytes(image_bytes)
    return str(path)


def _stage_image_array(image: np.ndarray, prefix: str) -> str:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Could not encode staged image")
    return _stage_image(encoded.tobytes(), prefix)


def _serialize_detections(detections: list[Detection] | None) -> list[dict[str, object]]:
    items = detections or []
    return [
        {
            "label": item.label,
            "confidence": round(float(item.confidence), 4),
            "box": [int(value) for value in item.box],
        }
        for item in items
    ]


def _parse_json_content(content: object) -> dict:
    if isinstance(content, dict):
        return content
    raw = str(content).strip()
    raw = _JSON_SUFFIX.sub("", _JSON_PREFIX.sub("", raw)).strip()
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("NeMo workflow did not return a JSON object")
    return result


def _raise_if_backoff_active(cache_key: str) -> None:
    if _UNAVAILABLE_UNTIL.get(cache_key, 0.0) > monotonic():
        raise RuntimeError("NeMo workflow temporarily unavailable")


def _mark_unavailable(cache_key: str) -> None:
    _UNAVAILABLE_UNTIL[cache_key] = monotonic() + _FAILURE_BACKOFF_SECONDS


async def _run_json_workflow(
    cache_key: str,
    system_prompt: str,
    payload: dict,
    timeout_seconds: float,
) -> dict:
    _raise_if_backoff_active(cache_key)
    try:
        response = await chat_completions(
            base_url=settings.nemo_agent_base_url,
            api_key=settings.nemo_agent_api_key,
            timeout_seconds=timeout_seconds,
            model=settings.nemo_agent_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=0,
        )
        return _parse_json_content(extract_chat_message_content(response))
    except Exception:
        _mark_unavailable(cache_key)
        raise


def _run_json_workflow_sync(
    cache_key: str,
    system_prompt: str,
    payload: dict,
    timeout_seconds: float,
) -> dict:
    _raise_if_backoff_active(cache_key)
    try:
        response = chat_completions_sync(
            base_url=settings.nemo_agent_base_url,
            api_key=settings.nemo_agent_api_key,
            timeout_seconds=timeout_seconds,
            model=settings.nemo_agent_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=0,
        )
        return _parse_json_content(extract_chat_message_content(response))
    except Exception:
        _mark_unavailable(cache_key)
        raise


async def orchestrate_scene_assessment(
    image_bytes: bytes,
    detections: list[Detection] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict:
    payload = {
        "image_path": _stage_image(image_bytes, "scene"),
        "image_shape": list(image_shape) if image_shape is not None else None,
        "yolo_detections_json": json.dumps(_serialize_detections(detections)),
    }
    result = await _run_json_workflow(
        "scene",
        _SCENE_WORKFLOW_PROMPT,
        payload,
        settings.nemo_agent_timeout_seconds,
    )
    result["nemo_orchestrated"] = True
    return result


async def orchestrate_fire_exit_validation(
    image_bytes: bytes,
    object_label: str | None = None,
) -> dict:
    payload = {
        "image_path": _stage_image(image_bytes, "fire-exit"),
        "object_label": str(object_label or "").strip(),
    }
    result = await _run_json_workflow(
        "validation",
        _FIRE_EXIT_VALIDATION_PROMPT,
        payload,
        settings.nemo_agent_timeout_seconds,
    )
    result["nemo_orchestrated"] = True
    return result


def orchestrate_object_detection_sync(
    image: np.ndarray,
    confidence_threshold: float | None = None,
) -> dict:
    payload = {
        "image_path": _stage_image_array(image, "detect"),
        "confidence_threshold": confidence_threshold,
    }
    result = _run_json_workflow_sync(
        "detect",
        _DETECTION_WORKFLOW_PROMPT,
        payload,
        settings.nemo_agent_timeout_seconds,
    )
    result["nemo_orchestrated"] = True
    return result


def orchestrate_segmentation_sync(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict:
    payload = {
        "image_path": _stage_image_array(image, "segment"),
        "box_json": json.dumps([int(value) for value in box]),
    }
    result = _run_json_workflow_sync(
        "segment",
        _SEGMENTATION_WORKFLOW_PROMPT,
        payload,
        settings.nemo_agent_timeout_seconds,
    )
    result["nemo_orchestrated"] = True
    return result
