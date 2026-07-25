import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.database import SessionLocal, get_db
from app.models import AnalysisJob, Camera, Incident, TelegramSubscriber
from app.schemas import CameraIn, CameraOut, IncidentOut, JobOut
from app.services.cv.detector import get_detector
from app.services.processing import (
    analyse_image,
    create_scene_incident,
    process_video_job,
    synthetic_demo_frame,
)
from app.services.scene_reasoning import assess_scene
from app.services.telegram import (
    answer_callback,
    deactivate_subscriber,
    is_configured,
    register_subscriber,
    send_bot_message,
    send_incident_alert,
    subscriber_chat_ids,
    telegram_api_status,
)


router = APIRouter(prefix="/api")


def _parse_polygon(raw: str, camera: Camera) -> list[list[float]]:
    try:
        polygon = json.loads(raw) if raw else camera.exit_zone
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "exit_zone must be valid JSON") from exc
    if not polygon or len(polygon) < 3:
        raise HTTPException(422, "Draw an exit zone with at least three points")
    return polygon


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB")
    return content


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.scalar(select(func.count()).select_from(Camera))
    detector_status = "ready"
    detector_name = settings.detector_provider
    try:
        if settings.detector_provider == "demo":
            detector_name = get_detector().name
    except Exception as exc:
        detector_status = str(exc)
    latest_incident = db.scalar(select(Incident).order_by(Incident.created_at.desc()))
    recipients = subscriber_chat_ids(db)
    return {
        "status": "ok",
        "detector": detector_name,
        "detector_status": detector_status,
        "llm": {
            "enabled": settings.llm_enabled,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        },
        "nemo_agent": {
            "enabled": settings.nemo_agent_enabled,
            "required": settings.nemo_agent_required,
            "model": settings.nemo_agent_model,
            "base_url": settings.nemo_agent_base_url,
        },
        "vision": {
            "enabled": settings.vision_enabled,
            "model": settings.vision_model,
            "base_url": settings.vision_base_url,
        },
        "telegram_configured": is_configured(),
        "telegram_subscribers": db.scalar(
            select(func.count())
            .select_from(TelegramSubscriber)
            .where(TelegramSubscriber.active.is_(True))
        ),
        "telegram_recipients": len(recipients),
        "telegram_delivery": (
            latest_incident.telegram_status if latest_incident else "no_incidents"
        ),
        "time": datetime.now(timezone.utc),
    }


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    counts = dict(
        db.execute(
            select(Incident.status, func.count(Incident.id)).group_by(Incident.status)
        ).all()
    )
    return {
        "open": counts.get("open", 0),
        "acknowledged": counts.get("acknowledged", 0),
        "false_alarm": counts.get("false_alarm", 0),
        "closed": counts.get("closed", 0),
        "cameras": db.scalar(select(func.count()).select_from(Camera)),
    }


@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return db.scalars(select(Camera).order_by(Camera.created_at)).all()


@router.post("/cameras", response_model=CameraOut, status_code=201)
def create_camera(payload: CameraIn, db: Session = Depends(get_db)):
    values = payload.model_dump()
    if not values["blocked_classes"]:
        values["blocked_classes"] = sorted(settings.blocked_class_set)
    camera = Camera(**values)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.put("/cameras/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, payload: CameraIn, db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    for key, value in payload.model_dump().items():
        setattr(camera, key, value)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("/incidents", response_model=list[IncidentOut])
def list_incidents(
    status: str | None = None, limit: int = 100, db: Session = Depends(get_db)
):
    query = select(Incident).order_by(Incident.created_at.desc()).limit(min(limit, 500))
    if status:
        query = query.where(Incident.status == status)
    return db.scalars(query).all()


@router.get("/incidents/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    return incident


def _set_incident_status(
    incident_id: str, status: str, actor: str, db: Session
) -> Incident:
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    incident.status = status
    if status == "acknowledged":
        incident.acknowledged_by = actor
        incident.acknowledged_at = datetime.now(timezone.utc)
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(incident)
    return incident


@router.post("/incidents/{incident_id}/acknowledge", response_model=IncidentOut)
def acknowledge_incident(
    incident_id: str, actor: str = "dashboard-user", db: Session = Depends(get_db)
):
    return _set_incident_status(incident_id, "acknowledged", actor, db)


@router.post("/incidents/{incident_id}/false-alarm", response_model=IncidentOut)
def false_alarm_incident(
    incident_id: str, actor: str = "dashboard-user", db: Session = Depends(get_db)
):
    return _set_incident_status(incident_id, "false_alarm", actor, db)


@router.post("/incidents/{incident_id}/close", response_model=IncidentOut)
def close_incident(
    incident_id: str, actor: str = "dashboard-user", db: Session = Depends(get_db)
):
    return _set_incident_status(incident_id, "closed", actor, db)


@router.post("/analyse/image")
async def analyse_uploaded_image(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    exit_zone: str = Form(""),
    db: Session = Depends(get_db),
):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    content = await _read_upload(file)
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Unsupported or invalid image")
    polygon = _parse_polygon(exit_zone, camera)
    if exit_zone:
        camera.exit_zone = polygon
        db.commit()
    try:
        return await analyse_image(db, camera, image, polygon)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/analyse/scene")
async def analyse_scene(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    db: Session = Depends(get_db),
):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    content = await _read_upload(file)
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Unsupported or invalid image")
    try:
        result = await assess_scene(content)
        incident = await create_scene_incident(db, camera, image, result)
        result["incidents"] = [incident.id] if incident else []
        result["annotated_image"] = (
            f"/{incident.evidence_image}" if incident and incident.evidence_image else None
        )
        result["telegram_status"] = incident.telegram_status if incident else "not_sent"
        return result
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/analyse/video", response_model=JobOut, status_code=202)
async def analyse_uploaded_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    exit_zone: str = Form(""),
    db: Session = Depends(get_db),
):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    polygon = _parse_polygon(exit_zone, camera)
    content = await _read_upload(file)
    safe_suffix = Path(file.filename or "video.mp4").suffix.lower()
    if safe_suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(415, "Supported video types: MP4, MOV, AVI, MKV, WebM")
    upload_path = ROOT / "uploads" / f"{uuid4()}{safe_suffix}"
    upload_path.write_bytes(content)
    camera.exit_zone = polygon
    job = AnalysisJob(
        media_type="video",
        filename=file.filename or upload_path.name,
        camera_id=camera.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(process_video_job, job.id, upload_path, polygon)
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/demo/frame")
def demo_frame():
    ok, encoded = cv2.imencode(".jpg", synthetic_demo_frame())
    if not ok:
        raise HTTPException(500, "Could not generate demo frame")
    return Response(encoded.tobytes(), media_type="image/jpeg")


@router.post("/telegram/test")
async def test_telegram(db: Session = Depends(get_db)):
    if not is_configured():
        raise HTTPException(400, "Telegram bot token is not configured")
    targets = subscriber_chat_ids(db)
    if not targets:
        raise HTTPException(
            400,
            f"No alert chats are subscribed. Send /start to "
            f"@{settings.telegram_bot_username}.",
        )
    incident = db.scalar(select(Incident).order_by(Incident.created_at.desc()))
    if not incident:
        raise HTTPException(400, "Create an incident before sending a test alert")
    status, message_id = await send_incident_alert(incident, targets)
    incident.telegram_status = status
    incident.telegram_message_id = message_id
    db.commit()
    return {"status": status, "message_id": message_id}


@router.get("/telegram/status")
async def telegram_status(db: Session = Depends(get_db)):
    api_status = await telegram_api_status()
    return {
        **api_status,
        "configured": is_configured(),
        "recipients": len(subscriber_chat_ids(db)),
        "polling_enabled": settings.telegram_polling_enabled,
        "bot_username": api_status.get("bot_username") or settings.telegram_bot_username,
        "message": (
            f"Send /start to @{settings.telegram_bot_username} to subscribe. "
            "USER_ID and TELEGRAM_ALERT_CHAT_ID are also accepted as recipients."
        ),
    }


@router.get("/telegram/webhook", include_in_schema=False)
async def telegram_webhook_info(db: Session = Depends(get_db)):
    status = await telegram_status(db)
    status["webhook_note"] = (
        "This URL receives Telegram POST updates. Browser GET requests show this "
        "diagnostic page instead of returning 404."
    )
    return status


@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if settings.telegram_webhook_secret and (
        x_telegram_bot_api_secret_token != settings.telegram_webhook_secret
    ):
        raise HTTPException(403, "Invalid Telegram webhook secret")
    payload = await request.json()
    callback = payload.get("callback_query")
    message = payload.get("message")
    if message:
        chat_id = str((message.get("chat") or {}).get("id", ""))
        text = str(message.get("text") or "").strip().lower()
        db = SessionLocal()
        try:
            if text.startswith("/stop"):
                deactivate_subscriber(db, chat_id)
                reply = "Smart Facility alerts are now disabled. Send /start to subscribe again."
            else:
                register_subscriber(db, message)
                reply = (
                    "You are subscribed to Smart Facility safety alerts. "
                    "Annotated incident images will be sent here. Send /stop to opt out."
                )
        finally:
            db.close()
        await send_bot_message(chat_id, reply)
        return {"ok": True, "subscribed": not text.startswith("/stop")}
    if not callback:
        return {"ok": True}
    data = callback.get("data", "")
    action, _, incident_id = data.partition(":")
    db = SessionLocal()
    try:
        if action == "ack":
            user = callback.get("from", {})
            actor = user.get("username") or user.get("first_name") or "telegram-user"
            _set_incident_status(incident_id, "acknowledged", actor, db)
            message = "Incident acknowledged"
        elif action == "false":
            _set_incident_status(incident_id, "false_alarm", "telegram-user", db)
            message = "Incident marked as a false alarm"
        else:
            message = "Unsupported action"
        await answer_callback(callback.get("id", ""), message)
        return {"ok": True, "message": message}
    finally:
        db.close()
