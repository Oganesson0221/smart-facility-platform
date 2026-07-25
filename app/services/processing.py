import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from app.services.cv.roi_engine import evaluate_detection, validate_polygon
from app.services.cv.tracker import IoUTracker
from app.services.events import event_hub


def make_incident_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    prefix = now.strftime("INC-%Y%m%d-%H%M%S")
    return f"{prefix}-{now.microsecond // 1000:03d}"


def _camera_classes(camera: Camera) -> set[str]:
    values = camera.blocked_classes or list(settings.blocked_class_set)
    return {value.strip().lower() for value in values}


def classify_scene_assessment(assessment: dict) -> tuple[str, str]:
    category = str(assessment.get("category") or "facility_safety").strip()
    combined_evidence = " ".join(
        [
            category,
            str(assessment.get("summary") or ""),
            *[str(item) for item in assessment.get("evidence", [])],
            *[str(item) for item in assessment.get("visible_objects", [])],
        ]
    ).lower()
    visible_objects = [
        str(item).lower()
        for item in assessment.get("visible_objects", [])
        if "sign" not in str(item).lower()
    ]
    known_objects = (
        "vehicle", "car", "truck", "motorcycle", "trolley", "pallet", "box", "person"
    )
    object_type = next(
        (
            known
            for known in known_objects
            if known in combined_evidence
            or any(known in visible for visible in visible_objects)
        ),
        category.lower().replace(" ", "_")[:80],
    )
    if "exit" in combined_evidence and any(
        word in combined_evidence for word in ("block", "obstruct", "impede")
    ):
        event_type = "exit_blocked"
    elif "parking" in combined_evidence or "no parking" in combined_evidence:
        event_type = "parking_violation"
    else:
        event_type = "scene_violation"
    return event_type, object_type


def _find_duplicate(
    db: Session, camera: Camera, object_type: str, now: datetime
) -> Incident | None:
    cutoff = now - timedelta(seconds=camera.alert_cooldown_seconds)
    return db.scalar(
        select(Incident)
        .where(
            Incident.camera_id == camera.id,
            Incident.object_type == object_type,
            Incident.status.in_(["open", "acknowledged"]),
            Incident.created_at >= cutoff,
        )
        .order_by(Incident.created_at.desc())
    )


async def _save_incident(
    db: Session,
    camera: Camera,
    obstruction,
    duration: float,
    frame: np.ndarray,
    polygon_points: list[list[float]],
    all_obstructions: list,
    first_seen: datetime,
) -> Incident:
    now = datetime.now(timezone.utc)
    duplicate = _find_duplicate(db, camera, obstruction.detection.label, now)
    if duplicate:
        duplicate.last_seen = now
        duplicate.duration_seconds = max(duplicate.duration_seconds, duration)
        duplicate.confidence = max(duplicate.confidence, obstruction.detection.confidence)
        duplicate.overlap = max(duplicate.overlap, obstruction.overlap)
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
        object_type=obstruction.detection.label,
        confidence=obstruction.detection.confidence,
        overlap=obstruction.overlap,
        duration_seconds=duration,
        first_seen=first_seen,
        last_seen=now,
        evidence_image=evidence_relative,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    await enrich_and_notify(db, incident, camera)
    await event_hub.broadcast({"type": "incident.created", "incident_id": incident.id})
    return incident


async def analyse_image(
    db: Session, camera: Camera, image: np.ndarray, polygon_points: list[list[float]]
) -> dict:
    polygon = validate_polygon(polygon_points)
    detector = get_detector()
    detections = detector.detect(image)
    obstructions = [
        evaluate_detection(
            detection,
            polygon,
            _camera_classes(camera),
            camera.minimum_overlap,
        )
        for detection in detections
        if detection.confidence >= camera.confidence_threshold
    ]
    incidents = []
    now = datetime.now(timezone.utc)
    for obstruction in obstructions:
        if obstruction.is_blocking:
            incident = await _save_incident(
                db,
                camera,
                obstruction,
                camera.persistence_seconds,
                image,
                polygon_points,
                obstructions,
                now,
            )
            if incident.id not in [item.id for item in incidents]:
                incidents.append(incident)
    preview_name = f"preview-{make_incident_id(now)}.jpg"
    preview_relative = f"evidence/{preview_name}"
    annotate(image, polygon_points, obstructions, ROOT / preview_relative)
    return {
        "provider": detector.name,
        "detections": [
            {
                "label": item.detection.label,
                "confidence": item.detection.confidence,
                "box": item.detection.box,
                "overlap": item.overlap,
                "is_blocking": item.is_blocking,
            }
            for item in obstructions
        ],
        "incidents": [item.id for item in incidents],
        "annotated_image": f"/{preview_relative}",
    }


async def create_scene_incident(
    db: Session,
    camera: Camera,
    image: np.ndarray,
    assessment: dict,
) -> Incident | None:
    if not assessment.get("violation"):
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
        db.commit()
        db.refresh(duplicate)
        return duplicate

    incident_id = make_incident_id(now)
    evidence_relative = f"evidence/{incident_id}.jpg"
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
        summary=violation_summary,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    await enrich_and_notify(db, incident, camera, preserve_summary=True)
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
        job.message = "Opening video"
        db.commit()
        polygon = validate_polygon(polygon_points)
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
            detections = tracker.update(
                [
                    detection
                    for detection in detector.detect(frame)
                    if detection.confidence >= camera.confidence_threshold
                ]
            )
            evaluated = [
                evaluate_detection(
                    detection, polygon, _camera_classes(camera), camera.minimum_overlap
                )
                for detection in detections
            ]
            currently_blocking = {
                item.detection.track_id
                for item in evaluated
                if item.is_blocking and item.detection.track_id is not None
            }
            for track_id in list(blocking_since):
                if track_id not in currently_blocking:
                    del blocking_since[track_id]
            for obstruction in evaluated:
                track_id = obstruction.detection.track_id
                if not obstruction.is_blocking or track_id is None:
                    continue
                blocking_since.setdefault(track_id, timestamp)
                duration = timestamp - blocking_since[track_id]
                if duration >= camera.persistence_seconds and track_id not in emitted_tracks:
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
                    )
                    incident_ids.append(incident.id)
                    emitted_tracks.add(track_id)
            if total_frames:
                job.progress = min(99.0, frame_index / total_frames * 100)
            job.message = f"Analysed {frame_index}/{total_frames or '?'} frames"
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
            else "Completed — no persistent obstruction found"
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
