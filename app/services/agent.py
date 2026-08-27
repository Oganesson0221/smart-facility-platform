"""Incident orchestration through NVIDIA NeMo Agent Toolkit."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Camera, Incident
from app.services.llm import _fallback as create_grounded_summary_fallback
from app.services.llm import create_grounded_summary
from app.services.sop import search_sops
from app.services.telegram import send_incident_alert, subscriber_chat_ids


async def enrich_and_notify(
    db: Session,
    incident: Incident,
    camera: Camera,
    preserve_summary: bool = False,
) -> Incident:
    sops = search_sops(incident.event_type, incident.facility, incident.object_type)
    metadata = incident.incident_metadata or {}
    assessment = (
        metadata.get("assessment", {})
        if isinstance(metadata, dict)
        else {}
    )
    event = {
        "incident_id": incident.id,
        "camera_id": camera.id,
        "facility": incident.facility,
        "zone": incident.zone,
        "event_type": incident.event_type,
        "object_type": incident.object_type,
        "confidence": incident.confidence,
        "overlap": incident.overlap,
        "spatial_method": metadata.get("spatial_method"),
        "object_intrusion_ratio": metadata.get("object_intrusion_ratio"),
        "exit_blockage_ratio": metadata.get("exit_blockage_ratio"),
        "mask_zone_iou": metadata.get("mask_zone_iou"),
        "mask_area_pixels": metadata.get("mask_area_pixels"),
        "sam_model": metadata.get("sam_model"),
        "sam_inference_ms": metadata.get("sam_inference_ms"),
        "sam_polygon": metadata.get("sam_polygon"),
        "yolo_box": metadata.get("yolo_box"),
        "yolo_class": metadata.get("yolo_class"),
        "yolo_confidence": metadata.get("yolo_confidence"),
        "vehicle_identifier": metadata.get("vehicle_identifier"),
        "vehicle_identifier_type": metadata.get("vehicle_identifier_type"),
        "vehicle_identifier_confidence": metadata.get("vehicle_identifier_confidence"),
        "first_seen": incident.first_seen.isoformat(),
        "duration_seconds": incident.duration_seconds,
        "summary_hint": incident.summary,
        "scene_assessment": {
            "category": assessment.get("category"),
            "summary": assessment.get("summary"),
            "evidence": assessment.get("evidence", []),
            "visible_objects": assessment.get("visible_objects", []),
            "supporting_objects": assessment.get("supporting_objects", []),
            "scene_detections": assessment.get("scene_detections", []),
        },
    }
    validation = metadata.get("vision_validation") or {}
    direct_iou_alert = (
        isinstance(validation, dict)
        and validation.get("mode") == "deterministic_iou"
    )
    if direct_iou_alert:
        generated_summary, incident.recommended_action = (
            create_grounded_summary_fallback(event, sops)
        )
    else:
        generated_summary, incident.recommended_action = await create_grounded_summary(
            event, sops
        )
    if not preserve_summary:
        incident.summary = generated_summary
    incident.sop_title = sops[0].title if sops else "No matching SOP"
    incident.sop_sources = [
        {"title": sop.title, "source": sop.source, "score": sop.score} for sop in sops
    ]
    db.commit()
    db.refresh(incident)
    status, message_id = await send_incident_alert(incident, subscriber_chat_ids(db))
    incident.telegram_status = status
    incident.telegram_message_id = message_id
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(incident)
    return incident
