"""Incident orchestration through NVIDIA NeMo Agent Toolkit."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Camera, Incident
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
    event = {
        "incident_id": incident.id,
        "camera_id": camera.id,
        "facility": incident.facility,
        "zone": incident.zone,
        "event_type": incident.event_type,
        "object_type": incident.object_type,
        "confidence": incident.confidence,
        "overlap": incident.overlap,
        "first_seen": incident.first_seen.isoformat(),
        "duration_seconds": incident.duration_seconds,
    }
    generated_summary, incident.recommended_action = await create_grounded_summary(event, sops)
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
