import asyncio
from dataclasses import dataclass
import json
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.database import SessionLocal
from app.models import AnalysisJob, Camera, Incident
from app.services.agent import enrich_and_notify
from app.services.cv.annotator import annotate, annotate_scene
from app.services.cv.detector import get_detector
from app.services.cv.roi_engine import (
    box_intersects_or_near_polygon,
    evaluate_detection,
    validate_polygon,
)
from app.services.cv.segmenter import get_segmenter
from app.services.cv.tracker import IoUTracker
from app.services.cv.types import Detection, Obstruction, SegmentationResult
from app.services.events import event_hub
from app.services.nemo_agent_client import orchestrate_object_detection_sync
from app.services.nemo_agent_client import orchestrate_segmentation_sync
from app.services.scene_reasoning import validate_fire_exit_obstruction


LOGGER = logging.getLogger(__name__)
_IMAGE_ANALYSIS_STAGE_TTL = timedelta(minutes=5)
_IMAGE_ANALYSIS_STAGE_LOCK = Lock()


@dataclass
class ImageAnalysisStage:
    token: str
    camera_id: str
    provider: str
    image: np.ndarray
    polygon_points: list[list[float]]
    zone_mode: str
    obstructions: list[Obstruction]
    annotated_relative: str
    created_at: datetime


_IMAGE_ANALYSIS_STAGES: dict[str, ImageAnalysisStage] = {}


def make_incident_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    prefix = now.strftime("INC-%Y%m%d-%H%M%S")
    return f"{prefix}-{now.microsecond // 1000:03d}"


def _normalized_label(label: str) -> str:
    return label.strip().lower()


def _camera_classes(camera: Camera) -> set[str]:
    values = camera.blocked_classes or list(settings.blocked_class_set)
    classes = {value.strip().lower() for value in values if str(value).strip()}
    if settings.include_person_as_obstruction:
        classes.add("person")
    else:
        classes.discard("person")
    return classes


def _minimum_duration_for_label(camera: Camera, label: str) -> float:
    return (
        settings.person_minimum_duration_seconds
        if _normalized_label(label) == "person"
        else camera.persistence_seconds
    )


def _obstruction_to_dict(item: Obstruction) -> dict[str, Any]:
    segmentation = item.segmentation
    return {
        "label": item.detection.label,
        "confidence": item.detection.confidence,
        "box": item.detection.box,
        "track_id": item.detection.track_id,
        "overlap": item.overlap,
        "object_intrusion_ratio": item.object_intrusion_ratio,
        "exit_blockage_ratio": item.exit_blockage_ratio,
        "blocked_duration_seconds": item.blocked_duration_seconds,
        "is_blocking": item.is_blocking,
        "spatial_method": item.spatial_method,
        "mask_zone_iou": item.mask_zone_iou,
        "fallback_reason": item.fallback_reason,
        "yolo_box": item.detection.box,
        "yolo_class": item.detection.label,
        "yolo_confidence": item.detection.confidence,
        "sam_polygon": (
            [[int(x), int(y)] for x, y in segmentation.polygon]
            if segmentation is not None
            else None
        ),
        "sam_model": segmentation.model_name if segmentation is not None else None,
        "sam_score": segmentation.score if segmentation is not None else None,
        "sam_inference_ms": segmentation.inference_ms if segmentation is not None else None,
        "sam_prompt_point": (
            [int(value) for value in segmentation.prompt_point]
            if segmentation is not None and segmentation.prompt_point is not None
            else None
        ),
        "segmentation": (
            {
                "sam_polygon": [[int(x), int(y)] for x, y in segmentation.polygon],
                "sam_model": segmentation.model_name,
                "sam_score": segmentation.score,
                "sam_inference_ms": segmentation.inference_ms,
                "mask_area_pixels": segmentation.area_pixels,
                "prompt_box": [int(value) for value in segmentation.prompt_box],
                "prompt_point": (
                    [int(value) for value in segmentation.prompt_point]
                    if segmentation.prompt_point is not None
                    else None
                ),
            }
            if segmentation is not None
            else None
        ),
        "segmentation_state": (
            "succeeded"
            if segmentation is not None
            else "rejected"
            if item.spatial_method == "sam_rejected"
            else "fallback"
            if item.fallback_reason
            else "not_requested"
        ),
    }


_SCENE_OBJECT_ALIASES: tuple[tuple[str, str], ...] = (
    ("pickup truck", "vehicle"),
    ("hand truck", "trolley"),
    ("delivery truck", "truck"),
    ("fire exit", "exit"),
    ("emergency exit", "exit"),
    ("no parking", "parking"),
    ("parking zone", "parking"),
    ("parking area", "parking"),
    ("restricted parking", "parking"),
    ("access zone", "parking"),
    ("access route", "parking"),
    ("forklifts", "forklift"),
    ("forklift", "forklift"),
    ("vehicles", "vehicle"),
    ("vehicle", "vehicle"),
    ("buses", "vehicle"),
    ("bus", "vehicle"),
    ("vans", "vehicle"),
    ("van", "vehicle"),
    ("cars", "car"),
    ("car", "car"),
    ("trucks", "truck"),
    ("truck", "truck"),
    ("motorcycles", "motorcycle"),
    ("motorcycle", "motorcycle"),
    ("motorbikes", "motorcycle"),
    ("motorbike", "motorcycle"),
    ("scooters", "motorcycle"),
    ("scooter", "motorcycle"),
    ("pallets", "pallet"),
    ("pallet", "pallet"),
    ("skids", "pallet"),
    ("skid", "pallet"),
    ("boxes", "box"),
    ("box", "box"),
    ("cartons", "box"),
    ("carton", "box"),
    ("crates", "box"),
    ("crate", "box"),
    ("trolleys", "trolley"),
    ("trolley", "trolley"),
    ("dollies", "trolley"),
    ("dolly", "trolley"),
    ("carts", "trolley"),
    ("cart", "trolley"),
    ("people", "person"),
    ("person", "person"),
    ("workers", "person"),
    ("worker", "person"),
    ("pedestrians", "person"),
    ("pedestrian", "person"),
    ("blocked", "blocked"),
    ("blocking", "blocked"),
    ("blocks", "blocked"),
    ("block", "blocked"),
    ("obstructs", "blocked"),
    ("obstructed", "blocked"),
    ("obstructing", "blocked"),
    ("obstruction", "blocked"),
    ("impeding", "blocked"),
    ("impede", "blocked"),
)
_SCENE_OBJECT_PRIORITY = {
    label: index
    for index, label in enumerate(
        ("box", "pallet", "trolley", "forklift", "car", "truck", "motorcycle", "vehicle", "person")
    )
}
_SCENE_VEHICLE_TYPES = {"vehicle", "car", "truck", "motorcycle"}
_SCENE_DIRECT_VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}


def _scene_text_segments(assessment: dict[str, Any]) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = []
    category = str(assessment.get("category") or "").strip()
    summary = str(assessment.get("summary") or "").strip()
    if category:
        segments.append((category, 0.6))
    if summary:
        segments.append((summary, 1.8))
    segments.extend(
        (str(item).strip(), 2.4)
        for item in assessment.get("evidence", [])
        if str(item).strip()
    )
    segments.extend(
        (str(item).strip(), 2.8)
        for item in assessment.get("visible_objects", [])
        if str(item).strip() and "sign" not in str(item).lower()
    )
    return segments


def _extract_scene_terms(text: str) -> list[str]:
    normalized = re.sub(r"[_/\\-]+", " ", text.lower())
    matches: list[tuple[int, int, str]] = []
    for alias, canonical in _SCENE_OBJECT_ALIASES:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        for match in pattern.finditer(normalized):
            matches.append((match.start(), match.end(), canonical))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    found: list[str] = []
    used_spans: list[tuple[int, int]] = []
    for start, end, canonical in matches:
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_spans):
            continue
        used_spans.append((start, end))
        found.append(canonical)
    return found


def _canonical_detection_label(label: str) -> str | None:
    terms = _extract_scene_terms(label)
    for term in terms:
        if term in _SCENE_OBJECT_PRIORITY or term in _SCENE_VEHICLE_TYPES:
            return term
    return None


def _phrase_present(assessment: dict[str, Any], canonical: str) -> bool:
    return any(canonical in _extract_scene_terms(text) for text, _ in _scene_text_segments(assessment))


def _scene_object_scores(
    assessment: dict[str, Any],
    detections: list[Detection] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for text, weight in _scene_text_segments(assessment):
        for term in _extract_scene_terms(text):
            if term not in _SCENE_OBJECT_PRIORITY:
                continue
            scores[term] = scores.get(term, 0.0) + weight
            first_seen.setdefault(term, order)
            order += 1

    mentioned = set(scores)
    for detection in detections or []:
        canonical = _canonical_detection_label(detection.label)
        if not canonical:
            continue
        if mentioned and canonical not in mentioned:
            continue
        scores[canonical] = scores.get(canonical, 0.0) + max(0.4, detection.confidence) * 2.5
        first_seen.setdefault(canonical, order)
        order += 1
    return scores, first_seen


def _fallback_scene_object_type(assessment: dict[str, Any]) -> str:
    category = str(assessment.get("category") or "facility_safety").strip().lower()
    return category.replace(" ", "_")[:80] or "facility_safety"


def _normalize_annotation_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(max(0.0, min(1000.0, x1 / max(width, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, y1 / max(height, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, x2 / max(width, 1) * 1000)), 2),
        round(max(0.0, min(1000.0, y2 / max(height, 1) * 1000)), 2),
    ]


def _detection_to_annotation(
    detection: Detection,
    width: int,
    height: int,
    label_override: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label_override or detection.label,
        "box": _normalize_annotation_box(detection.box, width, height),
    }


def _generic_scene_annotation_label(assessment: dict[str, Any]) -> str:
    labels = set()
    for text, _ in _scene_text_segments(assessment):
        labels.update(term for term in _extract_scene_terms(text) if term in {"box", "pallet", "trolley", "forklift"})
    if labels == {"box", "pallet"} or labels == {"pallet", "box"}:
        return "boxes / pallets"
    if "box" in labels:
        return "boxes"
    if "pallet" in labels:
        return "pallets"
    if "trolley" in labels:
        return "hand truck"
    if "forklift" in labels:
        return "forklift"
    return "obstruction evidence"


def _normalize_scene_support_label(label: str) -> str:
    normalized = label.strip().lower()
    canonical = _canonical_detection_label(normalized)
    if canonical:
        return canonical
    if normalized in _SCENE_DIRECT_VEHICLE_LABELS:
        return "vehicle"
    return normalized


def _scene_detection_matches(detection: Detection, support_labels: set[str]) -> bool:
    raw = detection.label.strip().lower()
    canonical = _canonical_detection_label(raw)
    keys = {raw}
    if canonical:
        keys.add(canonical)
    if raw in _SCENE_DIRECT_VEHICLE_LABELS:
        keys.add("vehicle")
    return bool(keys & support_labels)


def _scene_detections_from_assessment(assessment: dict[str, Any]) -> list[Detection]:
    return [
        Detection(
            str(item.get("label") or "object"),
            float(item.get("confidence", 0)),
            tuple(int(v) for v in item.get("box", (0, 0, 0, 0))),
        )
        for item in assessment.get("scene_detections", [])
        if isinstance(item, dict) and len(item.get("box", [])) == 4
    ]


def _scene_support_labels(assessment: dict[str, Any]) -> set[str]:
    return {
        _normalize_scene_support_label(str(item))
        for item in assessment.get("supporting_objects", [])
        if str(item).strip()
    }


def _select_scene_detections(assessment: dict[str, Any]) -> list[Detection]:
    detections = _scene_detections_from_assessment(assessment)
    support_labels = _scene_support_labels(assessment)
    if support_labels:
        detections = [
            item for item in detections if _scene_detection_matches(item, support_labels)
    ]
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def detect_scene_objects(camera: Camera, image: np.ndarray) -> tuple[str, list[Detection]]:
    return _run_detection_stage(camera, image)


def serialize_scene_detections(
    detections: list[Detection], limit: int | None = None
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


def parse_scene_detections_payload(raw: str) -> list[Detection]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("scene_detections must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("scene_detections must be a JSON array")

    detections: list[Detection] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each scene detection must be an object")
        label = str(item.get("label") or "").strip().lower()
        if not label:
            raise ValueError("Each scene detection requires a label")
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Each scene detection requires a four-value box")
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Each scene detection must contain numeric coordinates") from exc
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        detections.append(
            Detection(
                label,
                max(0.0, min(1.0, confidence)),
                (left, top, right, bottom),
            )
        )
    return detections


def ground_scene_assessment(
    image: np.ndarray,
    assessment: dict[str, Any],
    detections: list[Detection],
    detector_name: str,
) -> dict[str, Any]:
    assessment["model_annotations"] = list(assessment.get("annotations", []))
    assessment.setdefault("grounded_annotations", [])
    assessment["scene_detections"] = serialize_scene_detections(detections, limit=12)
    height, width = image.shape[:2]
    relevant = _select_scene_detections(assessment)
    grounded_annotations = [
        _detection_to_annotation(item, width, height)
        for item in relevant[:6]
    ]

    assessment["grounded_annotations"] = grounded_annotations
    assessment["annotations"] = grounded_annotations
    assessment["grounded_annotation_source"] = detector_name or "none"
    return assessment


def classify_scene_assessment(assessment: dict) -> tuple[str, str]:
    selected = _select_scene_detections(assessment)
    detections = _scene_detections_from_assessment(assessment)
    support_labels = _scene_support_labels(assessment)
    primary = (
        selected[0]
        if selected
        else detections[0]
        if detections and not support_labels
        else None
    )
    object_type = (
        primary.label.strip().lower()
        if primary is not None
        else _fallback_scene_object_type(assessment)
    )

    evidence_text = " ".join(text for text, _ in _scene_text_segments(assessment))
    terms = set(_extract_scene_terms(evidence_text))
    exit_related = "exit" in terms
    blocked = "blocked" in terms
    parking_related = "parking" in terms
    if exit_related and blocked:
        event_type = (
            "exit_blocked"
            if object_type in _SCENE_DIRECT_VEHICLE_LABELS or object_type == "vehicle"
            else "fire_exit_obstruction"
        )
    elif parking_related:
        event_type = "parking_violation"
    else:
        event_type = "scene_violation"
    return event_type, object_type


def _find_duplicate(
    db: Session,
    camera: Camera,
    object_type: str,
    now: datetime,
    incident_metadata: dict[str, Any] | None = None,
) -> Incident | None:
    cutoff = now - timedelta(seconds=camera.alert_cooldown_seconds)
    candidates = db.scalars(
        select(Incident)
        .where(
            Incident.camera_id == camera.id,
            Incident.object_type == object_type,
            Incident.status.in_(["open", "acknowledged"]),
            Incident.created_at >= cutoff,
        )
        .order_by(Incident.created_at.desc())
    ).all()
    if not incident_metadata:
        return candidates[0] if candidates else None
    expected_job = incident_metadata.get("analysis_job_id")
    expected_track = incident_metadata.get("track_id")
    if expected_job is None or expected_track is None:
        return candidates[0] if candidates else None
    for candidate in candidates:
        metadata = candidate.incident_metadata or {}
        if (
            metadata.get("analysis_job_id") == expected_job
            and metadata.get("track_id") == expected_track
        ):
            return candidate
    return None


def _encode_jpeg(image: np.ndarray, quality: int | None = None) -> bytes:
    params: list[int] = []
    if quality is not None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(30, min(100, quality)))]
    ok, encoded = cv2.imencode(".jpg", image, params)
    if not ok:
        raise ValueError("Could not encode evidence image")
    return encoded.tobytes()


def _full_frame_polygon(image: np.ndarray) -> list[list[int]]:
    height, width = image.shape[:2]
    max_x = max(1, width) - 1
    max_y = max(1, height) - 1
    return [[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]]


def _is_full_frame_zone(
    polygon_points: list[list[float]],
    width: int,
    height: int,
) -> bool:
    if len(polygon_points) < 4:
        return False
    xs = [int(point[0]) for point in polygon_points]
    ys = [int(point[1]) for point in polygon_points]
    return (
        min(xs) <= 1
        and min(ys) <= 1
        and max(xs) >= width - 2
        and max(ys) >= height - 2
    )


def _prepare_validation_image(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    max_dim = max(1, int(settings.vision_validation_image_max_dim))
    longest = max(height, width)
    if longest <= max_dim:
        return crop
    scale = max_dim / float(longest)
    resized = cv2.resize(
        crop,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _resolve_image_polygon(
    polygon_points: list[list[float]],
    image: np.ndarray,
) -> tuple[list[list[float]], str]:
    if polygon_points:
        try:
            validate_polygon(polygon_points)
            return polygon_points, "polygon"
        except ValueError as exc:
            LOGGER.warning(
                "Invalid image polygon provided; using full-frame ROI instead reason=%s",
                exc,
            )
    return _full_frame_polygon(image), "full_frame"


def _crop_validation_region(
    image: np.ndarray, polygon_points: list[list[float]], box: tuple[int, int, int, int]
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    full_frame_zone = _is_full_frame_zone(polygon_points, width, height)
    margin = max(24, int(max(x2 - x1, y2 - y1) * (0.18 if full_frame_zone else 0.12)))
    if full_frame_zone:
        # The whole image is the configured exit zone in this mode. Preserve
        # the door, signage, and approach-path context for Nemotron instead of
        # sending an object-only crop that cannot prove an exit obstruction.
        return image.copy()
    else:
        xs = [int(point[0]) for point in polygon_points]
        ys = [int(point[1]) for point in polygon_points]
        left = max(0, min(xs + [x1]) - margin)
        top = max(0, min(ys + [y1]) - margin)
        right = min(width, max(xs + [x2]) + margin)
        bottom = min(height, max(ys + [y2]) + margin)
    return image[top:bottom, left:right].copy()


async def _confirm_fire_exit_candidate(
    job: AnalysisJob | None,
    image: np.ndarray,
    obstruction: Obstruction,
    polygon_points: list[list[float]],
) -> tuple[bool, dict[str, Any], str | None]:
    if not settings.validate_fire_exit_incidents_with_vision:
        return True, {"mode": "disabled"}, None

    mask_zone_iou = obstruction.mask_zone_iou
    threshold = settings.vision_validation_iou_threshold
    if mask_zone_iou is not None and mask_zone_iou >= threshold:
        return True, {
            "mode": "deterministic_iou",
            "confirmed": True,
            "confidence": mask_zone_iou,
            "summary": (
                f"SAM mask/exit-zone IoU {mask_zone_iou:.1%} met the "
                f"{threshold:.1%} direct-alert threshold; Nemotron was skipped."
            ),
            "visible_evidence": [
                "YOLO detected a configured obstruction class",
                "SAM mask overlap met the deterministic direct-alert threshold",
            ],
            "model": "YOLO + SAM deterministic gate",
            "iou": mask_zone_iou,
            "threshold": threshold,
        }, None

    crop = _prepare_validation_image(
        _crop_validation_region(image, polygon_points, obstruction.detection.box)
    )
    crop_bytes = _encode_jpeg(crop, quality=settings.vision_validation_jpeg_quality)
    crop_relative = f"evidence/{make_incident_id()}-crop.jpg"
    crop_path = ROOT / crop_relative
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(crop_bytes)
    if job:
        job.message = "Validating incident"

    try:
        validation = await validate_fire_exit_obstruction(
            crop_bytes,
            object_label=obstruction.detection.label,
        )
    except RuntimeError as exc:
        validation = {"mode": "unavailable", "error": str(exc)}
        if settings.vision_validation_fail_closed:
            return False, validation, crop_relative
        return True, validation, crop_relative
    return bool(validation.get("confirmed")), validation, crop_relative


def _detector_runtime_label(detector: Any) -> str:
    return str(getattr(detector, "device", "cpu"))


def _matches_blocked_class(label: str, blocked_classes: set[str]) -> bool:
    normalized = _normalized_label(label)
    return normalized in blocked_classes or any(item in normalized for item in blocked_classes)


def _full_frame_accepts_all_detected_objects(zone_mode: str) -> bool:
    return zone_mode == "full_frame"


def _should_attempt_segmentation(
    detection: Detection,
    polygon: Any,
    blocked_classes: set[str],
    allow_all_classes: bool = False,
) -> bool:
    if not settings.sam_enabled:
        return False
    if not allow_all_classes and not _matches_blocked_class(detection.label, blocked_classes):
        return False
    if detection.confidence < settings.sam_min_yolo_confidence:
        return False
    if settings.sam_only_for_zone_candidates and not box_intersects_or_near_polygon(
        detection.box,
        polygon,
        settings.sam_boundary_margin_pixels,
    ):
        return False
    return True


def _segment_detection(
    image: np.ndarray,
    detection: Detection,
) -> tuple[Any | None, str | None, bool]:
    if not settings.sam_enabled:
        return None, None, False
    if settings.nemo_agent_enabled and settings.nemo_agent_orchestrate_cv:
        try:
            payload = orchestrate_segmentation_sync(image, detection.box)
            segmentation = _segmentation_from_orchestrated_payload(payload)
            LOGGER.info(
                "SAM segmentation succeeded provider=%s model=%s inference_ms=%.2f track=%s label=%s",
                payload.get("provider", "nemo-segmenter"),
                segmentation.model_name,
                segmentation.inference_ms,
                detection.track_id,
                detection.label,
            )
            return segmentation, None, False
        except Exception as exc:
            LOGGER.warning(
                "NeMo SAM workflow unavailable; falling back to local segmenter track=%s label=%s reason=%s",
                detection.track_id,
                detection.label,
                exc,
            )
    try:
        segmenter = get_segmenter()
        if segmenter is None:
            return None, None, False
        segmentation = segmenter.segment(image, detection.box)
        if segmentation is None:
            raise RuntimeError("SAM returned no segmentation result")
        LOGGER.info(
            "SAM segmentation succeeded provider=%s model=%s inference_ms=%.2f track=%s label=%s",
            getattr(segmenter, "name", "sam"),
            segmentation.model_name,
            segmentation.inference_ms,
            detection.track_id,
            detection.label,
        )
        return segmentation, None, False
    except RuntimeError as exc:
        message = str(exc)
        if settings.sam_fail_open:
            LOGGER.warning(
                "SAM segmentation failed open track=%s label=%s reason=%s",
                detection.track_id,
                detection.label,
                message,
            )
            return None, message, False
        LOGGER.warning(
            "SAM segmentation rejected candidate track=%s label=%s reason=%s",
            detection.track_id,
            detection.label,
            message,
        )
        return None, message, True


def _incident_metadata_for_obstruction(
    obstruction: Obstruction,
    duration: float,
    validation: dict[str, Any],
    crop_relative: str | None,
    analysis_job_id: str | None = None,
    zone_mode: str = "polygon",
) -> dict[str, Any]:
    segmentation = obstruction.segmentation
    metadata = {
        "analysis_job_id": analysis_job_id,
        "track_id": obstruction.detection.track_id,
        "object_intrusion_ratio": obstruction.object_intrusion_ratio,
        "exit_blockage_ratio": obstruction.exit_blockage_ratio,
        "blocked_duration_seconds": duration,
        "vision_validation": validation,
        "validation_crop": crop_relative,
        "spatial_method": obstruction.spatial_method,
        "mask_zone_iou": obstruction.mask_zone_iou,
        "mask_area_pixels": segmentation.area_pixels if segmentation is not None else None,
        "sam_model": segmentation.model_name if segmentation is not None else None,
        "sam_score": segmentation.score if segmentation is not None else None,
        "sam_inference_ms": (
            round(segmentation.inference_ms, 3) if segmentation is not None else None
        ),
        "sam_polygon": (
            [[int(x), int(y)] for x, y in segmentation.polygon]
            if segmentation is not None
            else None
        ),
        "sam_prompt_point": (
            [int(value) for value in segmentation.prompt_point]
            if segmentation is not None and segmentation.prompt_point is not None
            else None
        ),
        "yolo_box": [int(value) for value in obstruction.detection.box],
        "yolo_class": obstruction.detection.label,
        "yolo_confidence": obstruction.detection.confidence,
        "segmentation_fallback_reason": obstruction.fallback_reason,
        "vehicle_identifier": str(validation.get("vehicle_identifier") or "").strip() or None,
        "vehicle_identifier_type": str(validation.get("vehicle_identifier_type") or "").strip()
        or None,
        "vehicle_identifier_confidence": (
            float(validation["vehicle_identifier_confidence"])
            if validation.get("vehicle_identifier_confidence") is not None
            else None
        ),
        "zone_mode": zone_mode,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _cleanup_expired_image_analysis_stages() -> None:
    cutoff = datetime.now(timezone.utc) - _IMAGE_ANALYSIS_STAGE_TTL
    with _IMAGE_ANALYSIS_STAGE_LOCK:
        expired = [
            token
            for token, stage in _IMAGE_ANALYSIS_STAGES.items()
            if stage.created_at < cutoff
        ]
        for token in expired:
            _IMAGE_ANALYSIS_STAGES.pop(token, None)


def _store_image_analysis_stage(stage: ImageAnalysisStage) -> None:
    _cleanup_expired_image_analysis_stages()
    with _IMAGE_ANALYSIS_STAGE_LOCK:
        _IMAGE_ANALYSIS_STAGES[stage.token] = stage


def _take_image_analysis_stage(
    preview_token: str,
    camera_id: str,
) -> ImageAnalysisStage | None:
    if not preview_token:
        return None
    _cleanup_expired_image_analysis_stages()
    with _IMAGE_ANALYSIS_STAGE_LOCK:
        stage = _IMAGE_ANALYSIS_STAGES.pop(preview_token, None)
    if stage is None or stage.camera_id != camera_id:
        return None
    return stage


def _prepare_image_analysis_stage(
    camera: Camera,
    image: np.ndarray,
    polygon_points: list[list[float]],
    *,
    cache_stage: bool,
) -> ImageAnalysisStage:
    pipeline_started = perf_counter()
    provider, detections = _run_detection_stage(camera, image)
    resolved_polygon, zone_mode = _resolve_image_polygon(polygon_points, image)
    allow_all_classes = _full_frame_accepts_all_detected_objects(zone_mode)
    _, obstructions = _evaluate_obstructions(
        camera,
        detections,
        resolved_polygon,
        image,
        allow_all_classes=allow_all_classes,
    )
    created_at = datetime.now(timezone.utc)
    preview_name = f"preview-{make_incident_id(created_at)}.jpg"
    preview_relative = f"evidence/{preview_name}"
    annotate(image, resolved_polygon, obstructions, ROOT / preview_relative)
    stage = ImageAnalysisStage(
        token=uuid4().hex,
        camera_id=camera.id,
        provider=provider,
        image=image,
        polygon_points=resolved_polygon,
        zone_mode=zone_mode,
        obstructions=obstructions,
        annotated_relative=preview_relative,
        created_at=created_at,
    )
    if cache_stage:
        _store_image_analysis_stage(stage)
    LOGGER.info(
        "Prepared image CV stage provider=%s total_ms=%.2f obstructions=%d sam_used=%d zone_mode=%s cached=%s",
        provider,
        (perf_counter() - pipeline_started) * 1000,
        len(obstructions),
        sum(1 for item in obstructions if item.segmentation is not None),
        zone_mode,
        cache_stage,
    )
    return stage


def _analysis_telegram_status(incidents: list[Incident]) -> str:
    if not incidents:
        return "not_sent"
    statuses = [str(item.telegram_status or "not_sent") for item in incidents]
    unique = list(dict.fromkeys(statuses))
    if len(unique) == 1:
        return unique[0]
    if any(status.startswith("failed") for status in unique):
        return "partial"
    if any(status.startswith("partial") for status in unique):
        return "partial"
    if all(status == "sent" for status in unique):
        return "sent"
    return "mixed"


def _image_analysis_payload(
    stage: ImageAnalysisStage,
    incidents: list[Incident],
    *,
    preview_only: bool,
    vision_validations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blocking = [item for item in stage.obstructions if item.is_blocking]
    direct_alert = [
        item
        for item in blocking
        if item.mask_zone_iou is not None
        and item.mask_zone_iou >= settings.vision_validation_iou_threshold
    ]
    will_validate = (
        settings.validate_fire_exit_incidents_with_vision
        and len(direct_alert) < len(blocking)
    )
    payload = {
        "provider": stage.provider,
        "detections": [_obstruction_to_dict(item) for item in stage.obstructions],
        "incidents": [item.id for item in incidents],
        "annotated_image": f"/{stage.annotated_relative}",
        "zone_mode": stage.zone_mode,
        "telegram_status": _analysis_telegram_status(incidents),
        "vision_validations": vision_validations or [],
    }
    if preview_only:
        payload.update(
            {
                "preview_token": stage.token,
                "blocking_candidates": len(blocking),
                "direct_alert_candidates": len(direct_alert),
                "vision_validation_iou_threshold": settings.vision_validation_iou_threshold,
                "will_validate_with_vision": will_validate,
                "next_step": (
                    "IoU gate: direct alert or Nemotron validation"
                    if blocking and settings.validate_fire_exit_incidents_with_vision
                    else "Incident finalization"
                    if blocking
                    else "No escalation"
                ),
            }
        )
    return payload


async def _save_incident(
    db: Session,
    camera: Camera,
    obstruction: Obstruction,
    duration: float,
    frame: np.ndarray,
    polygon_points: list[list[float]],
    all_obstructions: list[Obstruction],
    first_seen: datetime,
    event_type: str = "fire_exit_obstruction",
    incident_metadata: dict[str, Any] | None = None,
) -> Incident:
    now = datetime.now(timezone.utc)
    duplicate = _find_duplicate(
        db, camera, obstruction.detection.label, now, incident_metadata=incident_metadata
    )
    if duplicate:
        duplicate.last_seen = now
        duplicate.duration_seconds = max(duplicate.duration_seconds, duration)
        duplicate.confidence = max(duplicate.confidence, obstruction.detection.confidence)
        duplicate.overlap = max(duplicate.overlap, obstruction.object_intrusion_ratio)
        merged_metadata = dict(duplicate.incident_metadata or {})
        merged_metadata.update(incident_metadata or {})
        duplicate.incident_metadata = merged_metadata
        db.commit()
        db.refresh(duplicate)
        return duplicate

    incident_id = make_incident_id(now)
    evidence_relative = f"evidence/{incident_id}.jpg"
    annotate(frame, polygon_points, all_obstructions, ROOT / evidence_relative)
    incident = Incident(
        id=incident_id,
        camera_id=camera.id,
        facility=camera.facility,
        zone=camera.zone,
        event_type=event_type,
        object_type=obstruction.detection.label,
        confidence=obstruction.detection.confidence,
        overlap=obstruction.object_intrusion_ratio,
        duration_seconds=duration,
        first_seen=first_seen,
        last_seen=now,
        evidence_image=evidence_relative,
        incident_metadata=incident_metadata or {},
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    await enrich_and_notify(db, incident, camera)
    await event_hub.broadcast({"type": "incident.created", "incident_id": incident.id})
    return incident


def _filtered_detections(camera: Camera, image: np.ndarray) -> list[Detection]:
    _, detections = _run_detection_stage(camera, image)
    return detections


def _detection_from_orchestrated_payload(item: dict[str, Any]) -> Detection:
    return Detection(
        str(item.get("label") or "object").strip().lower(),
        max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
        tuple(int(round(float(value))) for value in item.get("box", [0, 0, 0, 0])),
    )


def _segmentation_from_orchestrated_payload(payload: dict[str, Any]) -> SegmentationResult:
    data = payload.get("segmentation") if isinstance(payload.get("segmentation"), dict) else payload
    polygon = [
        (int(round(float(point[0]))), int(round(float(point[1]))))
        for point in data.get("polygon", [])
        if isinstance(point, list) and len(point) == 2
    ]
    image_shape = payload.get("image_shape") or data.get("image_shape") or [0, 0]
    if not isinstance(image_shape, list) or len(image_shape) < 2:
        raise RuntimeError("NeMo segmentation payload did not include image_shape")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or len(polygon) < 3:
        raise RuntimeError("NeMo segmentation payload did not include a usable polygon")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    prompt_box = tuple(int(round(float(value))) for value in data.get("prompt_box", [0, 0, 0, 0]))
    prompt_point_raw = data.get("prompt_point")
    prompt_point = (
        tuple(int(round(float(value))) for value in prompt_point_raw)
        if isinstance(prompt_point_raw, list) and len(prompt_point_raw) == 2
        else None
    )
    return SegmentationResult(
        mask=mask.astype(bool),
        polygon=polygon,
        area_pixels=int(data.get("area_pixels") or int(mask.sum())),
        prompt_box=prompt_box,
        prompt_point=prompt_point,
        score=float(data["score"]) if data.get("score") is not None else None,
        model_name=str(data.get("model_name") or data.get("model") or "nemo-segmenter"),
        inference_ms=float(data.get("inference_ms") or 0.0),
    )


def _run_detection_stage(camera: Camera, image: np.ndarray) -> tuple[str, list[Detection]]:
    if settings.nemo_agent_enabled and settings.nemo_agent_orchestrate_cv:
        started = perf_counter()
        try:
            payload = orchestrate_object_detection_sync(
                image,
                confidence_threshold=camera.confidence_threshold,
            )
            detections = [
                _detection_from_orchestrated_payload(item)
                for item in payload.get("detections", [])
                if isinstance(item, dict)
            ]
            provider = str(payload.get("provider") or "nemo-detector")
            inference_ms = (perf_counter() - started) * 1000
            LOGGER.info(
                "Detector stage provider=%s device=%s inference_ms=%.2f detections=%d kept=%d",
                provider,
                "nemo-agent",
                inference_ms,
                len(detections),
                len(detections),
            )
            return provider, detections
        except Exception as exc:
            LOGGER.warning(
                "NeMo detector workflow unavailable; falling back to local detector: %s",
                exc,
            )

    detector = get_detector()
    started = perf_counter()
    raw = detector.detect(image)
    filtered = [
        detection
        for detection in raw
        if detection.confidence >= camera.confidence_threshold
    ]
    inference_ms = (perf_counter() - started) * 1000
    LOGGER.info(
        "Detector stage provider=%s device=%s inference_ms=%.2f detections=%d kept=%d",
        detector.name,
        _detector_runtime_label(detector),
        inference_ms,
        len(raw),
        len(filtered),
    )
    return detector.name, filtered


def _evaluate_obstructions(
    camera: Camera,
    detections: list[Detection],
    polygon_points: list[list[float]],
    image: np.ndarray,
    *,
    allow_all_classes: bool = False,
) -> tuple[Any, list[Obstruction]]:
    polygon = validate_polygon(polygon_points)
    blocked_classes = _camera_classes(camera)
    evaluated: list[Obstruction] = []
    for detection in detections:
        segmentation = None
        fallback_reason = None
        reject_candidate = False
        relevant_candidate = (
            allow_all_classes
            or _matches_blocked_class(
                detection.label,
                blocked_classes,
            )
        ) and detection.confidence >= settings.sam_min_yolo_confidence
        zone_candidate = box_intersects_or_near_polygon(
            detection.box,
            polygon,
            settings.sam_boundary_margin_pixels,
        )
        if relevant_candidate and not settings.sam_enabled and (
            not settings.sam_only_for_zone_candidates or zone_candidate
        ):
            fallback_reason = "SAM is disabled in the current configuration"
        if _should_attempt_segmentation(
            detection,
            polygon,
            blocked_classes,
            allow_all_classes=allow_all_classes,
        ):
            segmentation, fallback_reason, reject_candidate = _segment_detection(
                image, detection
            )
        obstruction = evaluate_detection(
            detection,
            polygon,
            blocked_classes,
            camera.minimum_overlap,
            # A percentage-of-zone threshold is useful for a drawn clearance
            # polygon, but not when the entire image is explicitly the exit.
            # In full-frame mode, any qualifying SAM/box intrusion can proceed.
            0.0 if allow_all_classes else settings.minimum_exit_blockage_ratio,
            segmentation=segmentation,
            polygon_points=polygon_points,
            reject_candidate=reject_candidate,
            fallback_reason=fallback_reason,
            allow_all_classes=allow_all_classes,
        )
        if reject_candidate:
            obstruction.spatial_method = "sam_rejected"
        evaluated.append(obstruction)
    return polygon, evaluated


def preview_image_analysis(
    camera: Camera,
    image: np.ndarray,
    polygon_points: list[list[float]],
) -> dict[str, Any]:
    stage = _prepare_image_analysis_stage(
        camera,
        image,
        polygon_points,
        cache_stage=True,
    )
    return _image_analysis_payload(stage, [], preview_only=True)


async def analyse_image(
    db: Session,
    camera: Camera,
    image: np.ndarray,
    polygon_points: list[list[float]],
    preview_token: str = "",
) -> dict:
    pipeline_started = perf_counter()
    stage = _take_image_analysis_stage(preview_token, camera.id)
    if stage is None:
        stage = _prepare_image_analysis_stage(
            camera,
            image,
            polygon_points,
            cache_stage=False,
        )
    incidents = []
    vision_validations: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for obstruction in stage.obstructions:
        if not obstruction.is_blocking:
            continue
        confirmed, validation, crop_relative = await _confirm_fire_exit_candidate(
            None, stage.image, obstruction, stage.polygon_points
        )
        vision_validations.append(
            {
                "label": obstruction.detection.label,
                "box": [int(value) for value in obstruction.detection.box],
                "accepted": confirmed,
                "confirmed": validation.get("confirmed"),
                "confidence": validation.get("confidence"),
                "summary": str(validation.get("summary") or validation.get("error") or "No validation summary returned."),
                "visible_evidence": list(validation.get("visible_evidence") or []),
                "mode": str(validation.get("mode") or "vision"),
                "model": str(validation.get("model") or settings.vision_model),
            }
        )
        LOGGER.info(
            "Nemotron candidate decision label=%s accepted=%s confirmed=%s confidence=%s summary=%s",
            obstruction.detection.label,
            confirmed,
            validation.get("confirmed"),
            validation.get("confidence"),
            validation.get("summary") or validation.get("error") or "n/a",
        )
        if not confirmed:
            continue
        obstruction.blocked_duration_seconds = 0.0
        incident = await _save_incident(
            db,
            camera,
            obstruction,
            0.0,
            stage.image,
            stage.polygon_points,
            stage.obstructions,
            now,
            incident_metadata=_incident_metadata_for_obstruction(
                obstruction,
                0.0,
                validation,
                crop_relative,
                zone_mode=stage.zone_mode,
            ),
        )
        if incident.id not in [item.id for item in incidents]:
            incidents.append(incident)
    LOGGER.info(
        "Completed image CV pipeline provider=%s total_ms=%.2f obstructions=%d incidents=%d sam_used=%d",
        stage.provider,
        (perf_counter() - pipeline_started) * 1000,
        len(stage.obstructions),
        len(incidents),
        sum(1 for item in stage.obstructions if item.segmentation is not None),
    )
    return _image_analysis_payload(
        stage,
        incidents,
        preview_only=False,
        vision_validations=vision_validations,
    )


async def create_scene_incident(
    db: Session,
    camera: Camera,
    image: np.ndarray,
    assessment: dict,
) -> Incident | None:
    if not assessment.get("violation"):
        return None
    if not _select_scene_detections(assessment):
        return None

    now = datetime.now(timezone.utc)
    category = str(assessment.get("category") or "facility_safety").strip()
    event_type, object_type = classify_scene_assessment(assessment)
    duplicate = _find_duplicate(db, camera, object_type, now)
    if duplicate:
        duplicate.last_seen = now
        duplicate.confidence = max(
            duplicate.confidence, float(assessment.get("confidence", 0))
        )
        duplicate.incident_metadata = {"scene_reasoning": True, "assessment": assessment}
        db.commit()
        db.refresh(duplicate)
        return duplicate

    incident_id = make_incident_id(now)
    evidence_relative = f"evidence/{incident_id}.jpg"
    source_relative = f"evidence/{incident_id}-source.jpg"
    (ROOT / source_relative).write_bytes(_encode_jpeg(image))
    annotate_scene(image, assessment, ROOT / evidence_relative)
    evidence = "; ".join(str(item) for item in assessment.get("evidence", [])[:5])
    violation_summary = str(assessment.get("summary") or f"{category} violation detected")
    if evidence:
        violation_summary = f"{violation_summary} Visible evidence: {evidence}."
    incident = Incident(
        id=incident_id,
        camera_id=camera.id,
        facility=camera.facility,
        zone=camera.zone,
        event_type=event_type,
        object_type=object_type,
        confidence=float(assessment.get("confidence", 0)),
        overlap=0.0,
        duration_seconds=0.0,
        first_seen=now,
        last_seen=now,
        evidence_image=evidence_relative,
        incident_metadata={
            "scene_reasoning": True,
            "assessment": assessment,
            "source_image": source_relative,
        },
        summary=violation_summary,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    await enrich_and_notify(db, incident, camera, preserve_summary=False)
    await event_hub.broadcast({"type": "incident.created", "incident_id": incident.id})
    return incident


async def process_video_job(job_id: str, path: Path, polygon_points: list[list[float]]):
    db = SessionLocal()
    job = db.get(AnalysisJob, job_id)
    camera = db.get(Camera, job.camera_id) if job else None
    if not job or not camera:
        db.close()
        return
    capture = None
    try:
        job.status = "processing"
        job.message = "Loading video"
        db.commit()
        validate_polygon(polygon_points)
        detector = get_detector()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_every = max(1, round(source_fps / max(settings.video_sample_fps, 0.1)))
        tracker = IoUTracker(max_missed=max(5, int(settings.video_sample_fps * 2)))
        blocking_since: dict[int, float] = {}
        emitted_tracks: set[int] = set()
        frame_index = 0
        incident_ids: list[str] = []

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % sample_every:
                continue
            timestamp = frame_index / source_fps
            job.message = "Detecting objects"
            detections = tracker.update(_filtered_detections(camera, frame))
            _, evaluated = _evaluate_obstructions(camera, detections, polygon_points, frame)
            currently_blocking = {
                item.detection.track_id
                for item in evaluated
                if item.is_blocking and item.detection.track_id is not None
            }
            for track_id in list(blocking_since):
                if track_id not in currently_blocking:
                    del blocking_since[track_id]

            job.message = "Tracking obstruction"
            for obstruction in evaluated:
                track_id = obstruction.detection.track_id
                if not obstruction.is_blocking or track_id is None or track_id in emitted_tracks:
                    continue
                blocking_since.setdefault(track_id, timestamp)
                duration = timestamp - blocking_since[track_id]
                obstruction.blocked_duration_seconds = duration
                required_duration = _minimum_duration_for_label(camera, obstruction.detection.label)
                if duration < required_duration:
                    continue
                confirmed, validation, crop_relative = await _confirm_fire_exit_candidate(
                    job, frame, obstruction, polygon_points
                )
                if not confirmed:
                    emitted_tracks.add(track_id)
                    continue
                job.message = "Retrieving SOP"
                first_seen = datetime.now(timezone.utc) - timedelta(seconds=duration)
                incident = await _save_incident(
                    db,
                    camera,
                    obstruction,
                    duration,
                    frame,
                    polygon_points,
                    evaluated,
                    first_seen,
                    incident_metadata=_incident_metadata_for_obstruction(
                        obstruction,
                        duration,
                        validation,
                        crop_relative,
                        analysis_job_id=job.id,
                    ),
                )
                incident_ids.append(incident.id)
                emitted_tracks.add(track_id)
                job.message = "Sending alert"
            if total_frames:
                job.progress = min(99.0, frame_index / total_frames * 100)
            job.message = (
                f"Tracking obstruction: {frame_index}/{total_frames or '?'} frames sampled "
                f"with {detector.name}"
            )
            job.incidents = list(dict.fromkeys(incident_ids))
            db.commit()
            event_hub.broadcast_threadsafe(
                {"type": "job.progress", "job_id": job.id, "progress": job.progress}
            )
            await asyncio.sleep(0)
        job.status = "completed"
        job.progress = 100
        job.message = (
            f"Completed with {len(set(incident_ids))} incident(s)"
            if incident_ids
            else "Completed with no persistent fire-exit obstruction"
        )
        job.incidents = list(dict.fromkeys(incident_ids))
        db.commit()
        await event_hub.broadcast({"type": "job.completed", "job_id": job.id})
    except Exception as exc:
        job.status = "failed"
        job.message = str(exc)[:500]
        db.commit()
        await event_hub.broadcast(
            {"type": "job.failed", "job_id": job.id, "message": job.message}
        )
    finally:
        if capture is not None:
            capture.release()
        db.close()


def synthetic_demo_frame() -> np.ndarray:
    image = np.full((720, 1100, 3), (21, 27, 35), dtype=np.uint8)
    cv2.rectangle(image, (250, 100), (850, 680), (70, 83, 94), 8)
    cv2.rectangle(image, (285, 135), (815, 680), (37, 45, 55), -1)
    cv2.putText(
        image, "KEEP CLEAR", (395, 190), cv2.FONT_HERSHEY_DUPLEX, 1.5,
        (245, 245, 245), 3, cv2.LINE_AA
    )
    cv2.rectangle(image, (350, 395), (780, 625), (42, 70, 150), -1)
    cv2.rectangle(image, (390, 350), (685, 455), (55, 86, 170), -1)
    cv2.circle(image, (435, 620), 48, (20, 20, 25), -1)
    cv2.circle(image, (700, 620), 48, (20, 20, 25), -1)
    cv2.rectangle(image, (170, 655), (930, 690), (90, 95, 100), -1)
    return image
