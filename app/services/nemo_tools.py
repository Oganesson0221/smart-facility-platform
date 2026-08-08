from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import ROOT, settings
from app.services.cv.detector import get_detector
from app.services.cv.segmenter import get_segmenter
from app.services.cv.types import Detection
from app.services.scene_reasoning import assess_scene_direct
from app.services.scene_reasoning import validate_fire_exit_obstruction_direct
from app.services.sop import search_sops


def resolve_image_path(image_path: str) -> Path:
    candidate = Path(image_path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Image path '{image_path}' does not exist")
    return candidate


def load_image(image_path: str) -> np.ndarray:
    resolved = resolve_image_path(image_path)
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image path '{resolved}' is not a readable image")
    return image


def load_image_bytes(image_path: str) -> bytes:
    return resolve_image_path(image_path).read_bytes()


def serialize_scene_detections(
    detections: list[Detection],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    items = detections[:limit] if limit is not None else detections
    return [
        {
            "label": item.label,
            "confidence": round(float(item.confidence), 4),
            "box": [int(value) for value in item.box],
        }
        for item in items
    ]


def detect_image_objects_payload(
    image_path: str,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    image = load_image(image_path)
    detector = get_detector()
    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else settings.yolo_confidence_threshold
    )
    detections = [
        item for item in detector.detect(image) if float(item.confidence) >= threshold
    ]
    return {
        "provider": detector.name,
        "image_path": str(resolve_image_path(image_path)),
        "detections": serialize_scene_detections(detections, limit=24),
    }


def _detection_from_payload(item: dict[str, Any]) -> Detection:
    label = str(item.get("label") or "").strip().lower()
    if not label:
        raise ValueError("Each detection requires a label")
    box = item.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Each detection requires a four-value box")
    return Detection(
        label,
        max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
        tuple(int(round(float(value))) for value in box),
    )


def segment_image_box_payload(image_path: str, box_json: str) -> dict[str, Any]:
    image = load_image(image_path)
    segmenter = get_segmenter()
    if segmenter is None:
        raise RuntimeError("SAM segmentation is not available")
    try:
        box_payload = json.loads(box_json)
    except json.JSONDecodeError as exc:
        raise ValueError("box_json must be valid JSON") from exc
    if not isinstance(box_payload, list) or len(box_payload) != 4:
        raise ValueError("box_json must contain four numeric values")
    box = tuple(int(round(float(value))) for value in box_payload)
    result = segmenter.segment(image, box)
    return {
        "provider": getattr(segmenter, "name", "sam"),
        "image_path": str(resolve_image_path(image_path)),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "box": [int(value) for value in box],
        "segmentation": {
            "polygon": [[int(x), int(y)] for x, y in result.polygon],
            "model_name": result.model_name,
            "score": result.score,
            "inference_ms": result.inference_ms,
            "area_pixels": result.area_pixels,
            "prompt_box": [int(value) for value in result.prompt_box],
            "prompt_point": (
                [int(value) for value in result.prompt_point]
                if result.prompt_point is not None
                else None
            ),
        },
    }


async def inspect_scene_image_payload(
    image_path: str,
    yolo_detections_json: str = "[]",
) -> dict[str, Any]:
    image = load_image(image_path)
    image_bytes = load_image_bytes(image_path)
    detections_payload: list[dict[str, Any]]
    if yolo_detections_json.strip() and yolo_detections_json.strip() != "[]":
        try:
            raw = json.loads(yolo_detections_json)
        except json.JSONDecodeError as exc:
            raise ValueError("yolo_detections_json must be valid JSON") from exc
        if not isinstance(raw, list):
            raise ValueError("yolo_detections_json must be a JSON array")
        detections = [_detection_from_payload(item) for item in raw if isinstance(item, dict)]
        provider = "external"
    else:
        detected = detect_image_objects_payload(image_path)
        detections_payload = detected["detections"]
        detections = [_detection_from_payload(item) for item in detections_payload]
        provider = str(detected["provider"])
    result = await assess_scene_direct(image_bytes, detections, image.shape[:2])
    result["detector_provider"] = provider
    result["scene_detections"] = serialize_scene_detections(detections, limit=12)
    result["image_path"] = str(resolve_image_path(image_path))
    return result


async def validate_fire_exit_image_payload(
    image_path: str,
    object_label: str = "",
) -> dict[str, Any]:
    image_bytes = load_image_bytes(image_path)
    result = await validate_fire_exit_obstruction_direct(
        image_bytes,
        object_label=object_label or None,
    )
    result["image_path"] = str(resolve_image_path(image_path))
    return result


def lookup_safety_sop_payload(
    event_type: str,
    object_type: str,
    facility: str = "",
) -> dict[str, Any]:
    matches = search_sops(event_type, facility, object_type)
    return {
        "event_type": event_type,
        "object_type": object_type,
        "facility": facility,
        "matches": [
            {
                "title": item.title,
                "source": item.source,
                "score": item.score,
                "metadata": item.metadata,
                "content": item.content[:5000],
            }
            for item in matches
        ],
    }
