import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import logging
import re

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.models import Incident, TelegramSubscriber
from app.services.llm import answer_sop_question
from app.services.sop import load_telegram_sop_reference, search_sops


LOGGER = logging.getLogger(__name__)
_POLLING_STATE: dict[str, object] = {
    "status": "not_started",
    "last_error": None,
    "consecutive_failures": 0,
}


def _telegram_api_root() -> str:
    return settings.telegram_api_base_url.rstrip("/")


def _telegram_client(timeout: float) -> httpx.AsyncClient:
    kwargs: dict[str, object] = {"timeout": timeout}
    if settings.telegram_proxy_url.strip():
        kwargs["proxy"] = settings.telegram_proxy_url.strip()
    return httpx.AsyncClient(**kwargs)


def telegram_polling_status() -> dict[str, object]:
    return dict(_POLLING_STATE)


def is_configured() -> bool:
    return bool(settings.telegram_bot_token)


def _normalized_chat_id(value: str | int | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text == settings.telegram_bot_username:
        return ""
    if text == f"@{settings.telegram_bot_username}":
        return ""
    if text.lstrip("@").lower().endswith("bot"):
        return ""
    if text.startswith("@"):
        return text
    if text.lstrip("-").isdigit():
        return text
    return f"@{text}"


def subscriber_chat_ids(db: Session) -> list[str]:
    chat_ids = list(
        db.scalars(
            select(TelegramSubscriber.chat_id)
            .where(TelegramSubscriber.active.is_(True))
            .order_by(TelegramSubscriber.started_at)
        ).all()
    )
    if settings.telegram_alert_chat_id:
        chat_ids.append(settings.telegram_alert_chat_id)
    if settings.user_id:
        chat_ids.append(settings.user_id)
    return list(
        dict.fromkeys(
            normalized
            for normalized in (_normalized_chat_id(chat_id) for chat_id in chat_ids)
            if normalized
        )
    )


async def telegram_api_status() -> dict:
    if not is_configured():
        return {"reachable": False, "detail": "Bot token is not configured"}
    url = f"{_telegram_api_root()}/bot{settings.telegram_bot_token}/getMe"
    try:
        async with _telegram_client(timeout=5) as client:
            response = await client.get(url)
            response.raise_for_status()
        result = response.json().get("result") or {}
        return {
            "reachable": True,
            "detail": "Telegram Bot API is reachable",
            "bot_username": result.get("username"),
            "api_base_url": _telegram_api_root(),
            "proxy_configured": bool(settings.telegram_proxy_url.strip()),
        }
    except httpx.HTTPStatusError as exc:
        return {
            "reachable": False,
            "detail": f"Telegram returned HTTP {exc.response.status_code}",
            "api_base_url": _telegram_api_root(),
            "proxy_configured": bool(settings.telegram_proxy_url.strip()),
        }
    except httpx.ConnectError:
        return {
            "reachable": False,
            "detail": "TLS connection to api.telegram.org was reset or blocked",
        }
    except httpx.TimeoutException:
        return {"reachable": False, "detail": "Telegram connection timed out"}
    except (KeyError, ValueError, TypeError):
        return {"reachable": False, "detail": "Telegram returned an invalid response"}


def register_subscriber(db: Session, message: dict) -> TelegramSubscriber | None:
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    sender = message.get("from") or {}
    subscriber = db.get(TelegramSubscriber, str(chat_id))
    if subscriber is None:
        subscriber = TelegramSubscriber(chat_id=str(chat_id))
        db.add(subscriber)
    subscriber.username = sender.get("username") or chat.get("username")
    subscriber.first_name = sender.get("first_name") or chat.get("first_name")
    subscriber.active = True
    db.commit()
    db.refresh(subscriber)
    return subscriber


def deactivate_subscriber(db: Session, chat_id: str) -> bool:
    subscriber = db.get(TelegramSubscriber, str(chat_id))
    if subscriber is None:
        return False
    subscriber.active = False
    db.commit()
    return True


_INCIDENT_ID_PATTERN = re.compile(r"INC-\d{8}-\d{6}-\d{3}", re.IGNORECASE)
_PROCESSED_MESSAGE_KEYS: dict[str, None] = {}
_MESSAGE_CACHE_LIMIT = 512


def _normalized_message_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _format_percent(value: object) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "unknown"


def _incident_time_text(value: datetime | None) -> str:
    if not isinstance(value, datetime):
        return "unknown time"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _vehicle_identifier_description(incident: Incident) -> str:
    metadata = incident.incident_metadata or {}
    identifier = str(metadata.get("vehicle_identifier") or "").strip()
    identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    if not identifier:
        return "No vehicle identifier was readable in the saved evidence."
    confidence = metadata.get("vehicle_identifier_confidence")
    confidence_text = (
        f", read confidence {_format_percent(confidence)}"
        if confidence is not None
        else ""
    )
    type_text = _vehicle_identifier_suffix(identifier_type)
    return (
        f"The visible vehicle identifier for {incident.id} is {identifier}"
        f"{type_text}{confidence_text}. Treat this as a vision-assisted evidence read "
        "and verify it against the saved image before taking action."
    )


def _vision_validation_description(metadata: dict) -> str:
    validation = metadata.get("vision_validation") or {}
    if not isinstance(validation, dict):
        return "not recorded"
    mode = str(validation.get("mode") or "").strip().lower()
    if mode == "disabled":
        return "disabled"
    if mode == "unavailable":
        return "unavailable; deterministic CV decision used"
    if mode == "deterministic_iou":
        iou = validation.get("iou", validation.get("confidence"))
        threshold = validation.get("threshold")
        threshold_text = (
            f" met the {_format_percent(threshold)} threshold"
            if threshold is not None
            else " met the direct-alert threshold"
        )
        return f"direct alert; SAM IoU {_format_percent(iou)}{threshold_text}"
    confirmed = validation.get("confirmed")
    if confirmed is None:
        return "not recorded"
    confidence = validation.get("confidence")
    suffix = f" at {_format_percent(confidence)} confidence" if confidence is not None else ""
    return f"{'confirmed' if bool(confirmed) else 'rejected'}{suffix}"


def _vision_validation_label(metadata: dict) -> str:
    validation = metadata.get("vision_validation") or {}
    if isinstance(validation, dict) and validation.get("mode") == "deterministic_iou":
        return "IoU gate"
    return "Nemotron review"


def _incident_detail_answer(incident: Incident) -> str:
    metadata = incident.incident_metadata or {}
    identifier = str(metadata.get("vehicle_identifier") or "").strip()
    identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    vehicle_line = (
        f"\nVehicle identifier: {identifier}{_vehicle_identifier_suffix(identifier_type)} "
        f"— {_format_percent(metadata.get('vehicle_identifier_confidence'))} read confidence"
        if identifier
        else ""
    )
    duration = float(metadata.get("blocked_duration_seconds", incident.duration_seconds) or 0)
    duration_text = f"{duration:.1f}s" if duration > 0 else "snapshot (not timed)"
    return (
        f"Incident {incident.id}\n"
        f"Location: {incident.facility} — {incident.zone}\n"
        f"Object: {incident.object_type} ({incident.confidence:.0%} YOLO confidence)"
        f"{vehicle_line}\n"
        f"Object inside zone: {_format_percent(metadata.get('object_intrusion_ratio', incident.overlap))}\n"
        f"Exit area blocked: {_format_percent(metadata.get('exit_blockage_ratio'))}\n"
        f"Segmentation: {'SAM mask' if metadata.get('spatial_method') == 'sam_mask' else 'YOLO box'}\n"
        f"{_vision_validation_label(metadata)}: {_vision_validation_description(metadata)}\n"
        f"Duration: {duration_text}\n"
        f"Status: {incident.status.replace('_', ' ')}"
    )


def _reference_section_steps(reference_text: str, title: str) -> list[str]:
    match = re.search(
        rf"^\[{re.escape(title)}\]\s*$\n(.*?)(?=^\[[^\]]+\]\s*$|\Z)",
        reference_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return []
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if re.match(r"^\d+\.\s+", line.strip())
    ]


def _sop_section_title(incident: Incident | None) -> str:
    if incident is not None and incident.event_type == "parking_violation":
        return "Restricted Parking and Access Zone"
    if incident is not None and incident.object_type.lower() in {
        "vehicle",
        "car",
        "truck",
        "bus",
        "van",
        "motorcycle",
    }:
        return "Vehicle Blocking an Emergency Exit"
    return "Fire Exit Obstruction"


def _sop_answer(incident: Incident | None, reference_text: str) -> str:
    title = _sop_section_title(incident)
    steps = _reference_section_steps(reference_text, title)
    if not steps:
        return "The local SOP could not be loaded. Use /help or review the incident in the dashboard."
    context = f" for incident {incident.id}" if incident is not None else ""
    vehicle = ""
    if incident is not None:
        metadata = incident.incident_metadata or {}
        identifier = str(metadata.get("vehicle_identifier") or "").strip()
        if identifier:
            vehicle = f"\nRecorded visible vehicle identifier: {identifier}."
    return f"{title} SOP{context}:{vehicle}\n\n" + "\n".join(steps)


def _answer_known_incident_question(
    question: str,
    incident: Incident,
    reference_text: str,
) -> str | None:
    normalized = _normalized_message_text(question)
    metadata = incident.incident_metadata or {}
    identifier = str(metadata.get("vehicle_identifier") or "").strip()

    vehicle_terms = ("vehicle", "car", "truck", "van", "plate", "registration")
    identifier_query = _contains_any(
        normalized,
        ("number", "identifier", "plate", "registration"),
    ) or bool(re.search(r"\bid\b", normalized))
    if _contains_any(normalized, ("incident id", "incident number", "reference number")):
        return f"The incident ID is {incident.id}."

    if (
        _contains_any(normalized, ("license plate", "licence plate", "number plate"))
        or (
            identifier_query
            and (_contains_any(normalized, vehicle_terms) or (identifier and "incident" not in normalized))
        )
    ):
        return _vehicle_identifier_description(incident)

    if _contains_any(normalized, ("what does sop", "what is the sop", "sop say", "next step", "what should i do", "what do i do", "what to do")):
        return _sop_answer(incident, reference_text)

    if _contains_any(normalized, ("can i move", "can we move", "tow", "enter the vehicle")):
        return (
            f"No. For {incident.id}, the SOP says not to enter, tow, or move the vehicle "
            "without authorisation. Notify Facilities Security to locate the authorised driver."
        )

    if _contains_any(normalized, ("who do i notify", "who should i notify", "who to notify", "contact")):
        return f"Notify Facilities Security and give them incident ID {incident.id}."

    if _contains_any(normalized, ("when do i escalate", "when should i escalate", "how long before", "escalat")):
        return (
            f"If {incident.id} remains unresolved after 5 minutes, escalate to the duty "
            "facility manager. Record the acknowledgement and resolution time."
        )

    if _contains_any(normalized, ("where", "location", "which camera", "which exit")):
        return f"{incident.id} is at {incident.facility} — {incident.zone}."

    if _contains_any(normalized, ("status", "acknowledged", "closed", "false alarm")):
        return f"Incident {incident.id} is currently {incident.status.replace('_', ' ')}."

    if _contains_any(normalized, ("how much", "overlap", "inside the zone", "area blocked", "blockage", "iou")):
        return (
            f"For {incident.id}, {_format_percent(metadata.get('object_intrusion_ratio', incident.overlap))} "
            f"of the detected object is inside the zone and it covers "
            f"{_format_percent(metadata.get('exit_blockage_ratio'))} of the exit area. "
            f"The spatial method was {'a SAM mask' if metadata.get('spatial_method') == 'sam_mask' else 'a YOLO box'}."
        )

    if _contains_any(normalized, ("confidence", "confident", "how sure", "certain", "accurate")):
        return (
            f"YOLO confidence for {incident.id} is {incident.confidence:.0%}. "
            f"{_vision_validation_label(metadata)}: {_vision_validation_description(metadata)}."
        )

    if _contains_any(normalized, ("when", "what time", "how long", "duration")):
        duration = float(metadata.get("blocked_duration_seconds", incident.duration_seconds) or 0)
        duration_text = f"{duration:.1f} seconds" if duration > 0 else "a single image snapshot, so no duration was measured"
        return (
            f"{incident.id} was first seen at {_incident_time_text(incident.first_seen)}. "
            f"This incident came from {duration_text}."
        )

    if _contains_any(normalized, ("evidence", "photo", "image", "show me", "review")):
        return (
            f"Review the annotated evidence for {incident.id} here: "
            f"{settings.public_base_url.rstrip('/')}/#incident={incident.id}"
        )

    if normalized in {"why", "what happened", "tell me more", "details", "incident details", "what is this"}:
        return _incident_detail_answer(incident)

    return None


def _incident_context(incident: Incident | None) -> dict | None:
    if incident is None:
        return None
    metadata = incident.incident_metadata or {}
    return {
        "incident_id": incident.id,
        "facility": incident.facility,
        "zone": incident.zone,
        "event_type": incident.event_type,
        "object_type": incident.object_type,
        "status": incident.status,
        "severity": incident.severity,
        "confidence": incident.confidence,
        "summary": incident.summary,
        "recommended_action": incident.recommended_action,
        "spatial_method": metadata.get("spatial_method"),
        "object_intrusion_ratio": metadata.get("object_intrusion_ratio"),
        "exit_blockage_ratio": metadata.get("exit_blockage_ratio"),
        "blocked_duration_seconds": metadata.get(
            "blocked_duration_seconds", incident.duration_seconds
        ),
        "mask_zone_iou": metadata.get("mask_zone_iou"),
        "mask_area_pixels": metadata.get("mask_area_pixels"),
        "sam_model": metadata.get("sam_model"),
        "sam_score": metadata.get("sam_score"),
        "sam_inference_ms": metadata.get("sam_inference_ms"),
        "yolo_box": metadata.get("yolo_box"),
        "yolo_class": metadata.get("yolo_class"),
        "yolo_confidence": metadata.get("yolo_confidence"),
        "vehicle_identifier": metadata.get("vehicle_identifier"),
        "vehicle_identifier_type": metadata.get("vehicle_identifier_type"),
        "vehicle_identifier_confidence": metadata.get("vehicle_identifier_confidence"),
        "zone_mode": metadata.get("zone_mode"),
        "vision_validation": metadata.get("vision_validation"),
        "evidence_image": incident.evidence_image,
        "created_at": incident.created_at.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "first_seen": incident.first_seen.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }


def _question_prefers_latest_incident(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        phrase in lowered
        for phrase in (
            "what should i do",
            "what do i do",
            "what to do",
            "what next",
            "next step",
            "next steps",
            "latest incident",
            "latest alert",
        )
    )


def _resolve_incident_for_question(db: Session, text: str) -> Incident | None:
    match = _INCIDENT_ID_PATTERN.search(text)
    if match:
        return db.get(Incident, match.group(0).upper())
    if _question_prefers_latest_incident(text):
        incident = db.scalar(
            select(Incident)
            .where(Incident.status.in_(["open", "acknowledged"]))
            .order_by(Incident.created_at.desc())
        )
        if incident is not None:
            return incident
    return db.scalar(select(Incident).order_by(Incident.created_at.desc()))


async def answer_telegram_message(db: Session, text: str) -> str:
    question = str(text or "").strip()
    if not question:
        return "Send a question, for example: What should I do next for the latest incident?"

    normalized = _normalized_message_text(question)
    if normalized in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return (
            "Hello. I can help with the latest facility incident, vehicle details, "
            "overlap measurements, status, evidence, or SOP next steps. What would you like to know?"
        )

    explicit_match = _INCIDENT_ID_PATTERN.search(question)
    incident = _resolve_incident_for_question(db, question)
    if explicit_match is not None and incident is None:
        return f"I could not find incident {explicit_match.group(0).upper()}. Check the ID and try again."

    if normalized in {
        "ok",
        "okay",
        "got it",
        "understood",
        "thanks",
        "thank you",
        "thank you very much",
        "cheers",
    }:
        if incident is None:
            return "You're welcome. Send /help whenever you need facility-safety assistance."
        return (
            f"You're welcome. Incident {incident.id} is currently "
            f"{incident.status.replace('_', ' ')}. Ask me for its vehicle details, evidence, "
            "status, overlap, or SOP steps at any time."
        )

    sop_reference_path, sop_reference_text = load_telegram_sop_reference()
    if incident is not None:
        known_answer = _answer_known_incident_question(
            question,
            incident,
            sop_reference_text,
        )
        if known_answer is not None:
            return known_answer

    safety_terms = (
        "incident",
        "sop",
        "exit",
        "obstruction",
        "vehicle",
        "car",
        "truck",
        "parking",
        "security",
        "camera",
        "zone",
        "alert",
        "safety",
        "driver",
    )
    if explicit_match is None and not _contains_any(normalized, safety_terms):
        return (
            "I’m the Smart Facility safety assistant, so I can help with incidents, "
            "vehicle details, evidence, status, and local SOP actions. Send /help for examples."
        )

    incident_context = _incident_context(incident)
    if incident is not None:
        matched_sops = search_sops(
            incident.event_type,
            incident.facility,
            incident.object_type,
            limit=2,
        )
    else:
        # Use a broad fire-exit obstruction default when there is no incident context.
        matched_sops = search_sops("fire_exit_obstruction", "all", "vehicle", limit=2)
    reply = await answer_sop_question(
        question,
        incident_context,
        sop_reference_path,
        sop_reference_text,
        matched_sops,
    )
    cleaned = reply.strip()
    if len(cleaned) > 1400:
        cleaned = cleaned[:1399].rstrip() + "…"
    return cleaned


async def handle_incoming_message(db: Session, message: dict) -> tuple[str, bool]:
    chat_id = str((message.get("chat") or {}).get("id", ""))
    message_id = str(message.get("message_id") or "").strip()
    message_key = f"{chat_id}:{message_id}" if chat_id and message_id else ""
    if message_key and message_key in _PROCESSED_MESSAGE_KEYS:
        subscriber = db.get(TelegramSubscriber, chat_id)
        return "", bool(subscriber and subscriber.active)

    def finish(reply: str, subscribed: bool) -> tuple[str, bool]:
        if message_key:
            _PROCESSED_MESSAGE_KEYS[message_key] = None
            while len(_PROCESSED_MESSAGE_KEYS) > _MESSAGE_CACHE_LIMIT:
                _PROCESSED_MESSAGE_KEYS.pop(next(iter(_PROCESSED_MESSAGE_KEYS)))
        return reply, subscribed

    raw_text = str(message.get("text") or "").strip()
    text = raw_text.lower()
    if text.startswith("/stop"):
        deactivate_subscriber(db, chat_id)
        return finish(
            "Smart Facility alerts are now disabled. Send /start to subscribe again.",
            False,
        )
    if text.startswith("/start"):
        register_subscriber(db, message)
        return finish(
            "You are subscribed to Smart Facility safety alerts. "
            "Annotated incident images will be sent here.\n\n"
            "You can also ask the bot questions like:\n"
            "- What should I do next for the latest incident?\n"
            "- What should I do next for INC-20260730-042724-593?\n"
            "- What is the vehicle number in the latest incident?\n"
            "- How much of the exit is blocked?\n"
            "- What does the fire exit obstruction SOP say?\n\n"
            "Send /stop to opt out.",
            True,
        )
    if text.startswith("/help"):
        register_subscriber(db, message)
        return finish(
            "Ask about the latest incident or a specific incident ID.\n"
            "Examples:\n"
            "- What should I do next for the latest incident?\n"
            "- What should I do next for INC-20260730-042724-593?\n"
            "- What is the vehicle number?\n"
            "- Where and when was it detected?\n"
            "- How confident are YOLO and Nemotron?\n"
            "- Show me the evidence.\n"
            "- What does the restricted parking SOP say?\n\n"
            "The bot answers from the local SOP reference file and incident data.",
            True,
        )
    register_subscriber(db, message)
    return finish(await answer_telegram_message(db, raw_text), True)


def _keyboard(incident_id: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Acknowledge", "callback_data": f"ack:{incident_id}"},
                {"text": "⚪ False alarm", "callback_data": f"false:{incident_id}"},
            ],
            [
                {
                    "text": "View incident",
                    "url": f"{settings.public_base_url.rstrip('/')}/#incident={incident_id}",
                }
            ],
        ]
    }


def _vehicle_identifier_suffix(identifier_type: str) -> str:
    normalized = identifier_type.strip()
    if not normalized or normalized == "none":
        return ""
    return f" ({normalized.replace('_', ' ')})"


def _alert_text(incident) -> str:
    incident_time = getattr(incident, "created_at", None) or getattr(incident, "first_seen", None)
    metadata = getattr(incident, "incident_metadata", {}) or {}
    if isinstance(incident_time, datetime):
        if incident_time.tzinfo is None:
            incident_time = incident_time.replace(tzinfo=timezone.utc)
        incident_time_text = incident_time.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    else:
        incident_time_text = "Unknown"
    event_name = (
        "FIRE EXIT OBSTRUCTION"
        if incident.event_type in {"exit_blocked", "fire_exit_obstruction"}
        else incident.object_type.replace("_", " ").upper()
    )
    intrusion = float(metadata.get("object_intrusion_ratio", getattr(incident, "overlap", 0)) or 0)
    blockage = float(metadata.get("exit_blockage_ratio", 0) or 0)
    duration = float(metadata.get("blocked_duration_seconds", getattr(incident, "duration_seconds", 0)) or 0)
    duration_text = f"{duration:.1f}s" if duration > 0 else "snapshot (not timed)"
    method = str(metadata.get("spatial_method") or "yolo_box_fallback")
    method_label = "SAM mask" if method == "sam_mask" else "YOLO bounding box"
    first_action = (
        str(incident.recommended_action).splitlines()[0].strip()
        if str(getattr(incident, "recommended_action", "")).strip()
        else "Confirm the location and notify Facilities Security."
    )
    vehicle_identifier = str(metadata.get("vehicle_identifier") or "").strip()
    vehicle_identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    vehicle_identifier_confidence = metadata.get("vehicle_identifier_confidence")
    vehicle_identifier_confidence_suffix = (
        f" — {_format_percent(vehicle_identifier_confidence)} read confidence"
        if vehicle_identifier_confidence is not None
        else ""
    )
    vehicle_identifier_line = (
        f"Vehicle identifier: {vehicle_identifier}"
        f"{_vehicle_identifier_suffix(vehicle_identifier_type)}"
        f"{vehicle_identifier_confidence_suffix}\n"
        if vehicle_identifier
        else ""
    )
    return (
        f"🚨 {event_name} VIOLATION DETECTED\n\n"
        f"Location: {incident.facility} – {incident.zone}\n"
        f"Detected: {incident_time_text}\n"
        f"Object type: {incident.object_type}\n"
        f"{vehicle_identifier_line}"
        f"YOLO confidence: {incident.confidence:.0%}\n"
        f"Object inside zone: {intrusion:.0%}\n"
        f"Exit area blocked: {blockage:.0%}\n"
        f"Blocked duration: {duration_text}\n"
        f"Segmentation method: {method_label}\n"
        f"{_vision_validation_label(metadata)}: {_vision_validation_description(metadata)}\n"
        f"Violation: {incident.summary}\n"
        f"Incident: {incident.id}\n\n"
        f"Recommended first action: {first_action}\n\n"
        f"Required response:\n{incident.recommended_action}"
    )


def _alert_caption(incident, limit: int = 900) -> str:
    metadata = getattr(incident, "incident_metadata", {}) or {}
    method = str(metadata.get("spatial_method") or "yolo_box_fallback")
    method_label = "SAM mask" if method == "sam_mask" else "YOLO bounding box"
    intrusion = float(metadata.get("object_intrusion_ratio", getattr(incident, "overlap", 0)) or 0)
    blockage = float(metadata.get("exit_blockage_ratio", 0) or 0)
    duration = float(metadata.get("blocked_duration_seconds", getattr(incident, "duration_seconds", 0)) or 0)
    duration_text = f"{duration:.1f}s" if duration > 0 else "snapshot (not timed)"
    first_action = (
        str(getattr(incident, "recommended_action", "") or "").splitlines()[0].strip()
        or "Confirm the location and notify Facilities Security."
    )
    vehicle_identifier = str(metadata.get("vehicle_identifier") or "").strip()
    vehicle_identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    vehicle_identifier_confidence = metadata.get("vehicle_identifier_confidence")
    vehicle_identifier_confidence_suffix = (
        f" — {_format_percent(vehicle_identifier_confidence)} read confidence"
        if vehicle_identifier_confidence is not None
        else ""
    )
    lines = [
        "🚨 FIRE EXIT OBSTRUCTION",
        f"Incident: {incident.id}",
        f"Location: {incident.facility} — {incident.zone}",
        f"Object: {incident.object_type}",
        *(
            [
                "Vehicle ID: "
                f"{vehicle_identifier}"
                f"{_vehicle_identifier_suffix(vehicle_identifier_type)}"
                f"{vehicle_identifier_confidence_suffix}"
            ]
            if vehicle_identifier
            else []
        ),
        f"YOLO confidence: {incident.confidence:.0%}",
        f"Object inside zone: {intrusion:.0%}",
        f"Exit area blocked: {blockage:.0%}",
        f"Blocked duration: {duration_text}",
        f"Segmentation: {method_label}",
        f"{_vision_validation_label(metadata)}: {_vision_validation_description(metadata)}",
        f"First action: {first_action}",
    ]
    caption = "\n".join(lines)
    return caption if len(caption) <= limit else caption[: limit - 1].rstrip() + "…"


async def _send_to_chat(
    client: httpx.AsyncClient,
    chat_id: str,
    incident,
    evidence: Path | None,
) -> str:
    url = f"{_telegram_api_root()}/bot{settings.telegram_bot_token}"
    data = {
        "chat_id": chat_id,
        "reply_markup": json.dumps(_keyboard(incident.id)),
    }
    if evidence is not None:
        data["caption"] = _alert_caption(incident)
        try:
            with evidence.open("rb") as image:
                response = await client.post(
                    f"{url}/sendPhoto",
                    data=data,
                    files={"photo": (evidence.name, image, "image/jpeg")},
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            retry_data = dict(data)
            retry_data.pop("caption", None)
            with evidence.open("rb") as image:
                response = await client.post(
                    f"{url}/sendPhoto",
                    data=retry_data,
                    files={"photo": (evidence.name, image, "image/jpeg")},
                )
    else:
        data["text"] = _alert_text(incident)
        response = await client.post(f"{url}/sendMessage", data=data)
    response.raise_for_status()
    return str(response.json()["result"]["message_id"])


async def send_incident_alert(
    incident,
    chat_ids: list[str] | None = None,
) -> tuple[str, str | None]:
    if not is_configured():
        return "not_configured", None
    targets = list(dict.fromkeys(chat_ids or [settings.telegram_alert_chat_id]))
    targets = [str(chat_id) for chat_id in targets if chat_id]
    if not targets:
        return "no_subscribers", None

    evidence_path = ROOT / (incident.evidence_image or "")
    evidence = evidence_path if incident.evidence_image and evidence_path.exists() else None
    message_ids: dict[str, str] = {}
    failures: list[str] = []
    try:
        async with _telegram_client(timeout=30) as client:
            for chat_id in targets:
                try:
                    message_ids[chat_id] = await _send_to_chat(
                        client, chat_id, incident, evidence
                    )
                except httpx.HTTPStatusError as exc:
                    try:
                        detail = exc.response.json().get("description", "")
                    except (ValueError, AttributeError):
                        detail = exc.response.text[:120]
                    failures.append(detail or f"HTTP {exc.response.status_code}")
                except httpx.ConnectError:
                    failures.append("could not connect to api.telegram.org")
                except httpx.TimeoutException:
                    failures.append("Telegram request timed out")
                except (KeyError, OSError) as exc:
                    failures.append(str(exc).strip() or type(exc).__name__)
    except httpx.HTTPError as exc:
        return f"failed: {str(exc)[:160]}", None

    first_message_id = next(iter(message_ids.values())) if message_ids else None
    if failures and not message_ids:
        return f"failed: {failures[0][:160]}", None
    if failures:
        return f"partial: sent {len(message_ids)}/{len(targets)}", first_message_id
    status = "sent" if len(message_ids) == 1 else f"sent to {len(message_ids)} chats"
    return status, first_message_id


async def send_bot_message(chat_id: str, text: str) -> None:
    if not is_configured() or not str(text or "").strip():
        return
    url = f"{_telegram_api_root()}/bot{settings.telegram_bot_token}/sendMessage"
    try:
        async with _telegram_client(timeout=15) as client:
            response = await client.post(url, json={"chat_id": chat_id, "text": text})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        LOGGER.warning("Telegram reply failed chat_id=%s error=%s", chat_id, exc)
        return


async def answer_callback(callback_id: str, text: str):
    if not settings.telegram_bot_token:
        return
    url = f"{_telegram_api_root()}/bot{settings.telegram_bot_token}/answerCallbackQuery"
    try:
        async with _telegram_client(timeout=15) as client:
            response = await client.post(
                url,
                json={"callback_query_id": callback_id, "text": text},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        LOGGER.warning("Telegram callback reply failed error=%s", exc)
        return


async def _process_polled_update(update: dict) -> None:
    from app.database import SessionLocal

    message = update.get("message")
    callback = update.get("callback_query")
    db = SessionLocal()
    try:
        if message:
            chat_id = str((message.get("chat") or {}).get("id", ""))
            reply, _ = await handle_incoming_message(db, message)
            await send_bot_message(chat_id, reply)
        elif callback:
            action, _, incident_id = str(callback.get("data") or "").partition(":")
            incident = db.get(Incident, incident_id)
            if incident and action in {"ack", "false"}:
                if action == "ack":
                    sender = callback.get("from") or {}
                    incident.status = "acknowledged"
                    incident.acknowledged_by = (
                        sender.get("username")
                        or sender.get("first_name")
                        or "telegram-user"
                    )
                    incident.acknowledged_at = datetime.now(timezone.utc)
                    response = "Incident acknowledged"
                else:
                    incident.status = "false_alarm"
                    response = "Incident marked as a false alarm"
                incident.updated_at = datetime.now(timezone.utc)
                db.commit()
            else:
                response = "Incident or action is no longer available"
            await answer_callback(str(callback.get("id") or ""), response)
    finally:
        db.close()


async def poll_telegram_updates() -> None:
    if not (is_configured() and settings.telegram_polling_enabled):
        _POLLING_STATE.update(status="disabled", last_error=None, consecutive_failures=0)
        return
    base_url = f"{_telegram_api_root()}/bot{settings.telegram_bot_token}"
    offset: int | None = None
    retry_seconds = 5
    LOGGER.info(
        "Telegram polling starting api_base_url=%s proxy_configured=%s",
        _telegram_api_root(),
        bool(settings.telegram_proxy_url.strip()),
    )
    while True:
        try:
            _POLLING_STATE["status"] = "connecting"
            async with _telegram_client(
                timeout=settings.telegram_poll_timeout_seconds + 10
            ) as client:
                delete_response = await client.post(
                    f"{base_url}/deleteWebhook",
                    json={"drop_pending_updates": False},
                )
                delete_response.raise_for_status()
                _POLLING_STATE.update(
                    status="polling",
                    last_error=None,
                    consecutive_failures=0,
                )
                retry_seconds = 5
                while True:
                    response = await client.get(
                        f"{base_url}/getUpdates",
                        params={
                            "timeout": settings.telegram_poll_timeout_seconds,
                            **({"offset": offset} if offset is not None else {}),
                        },
                    )
                    response.raise_for_status()
                    for update in response.json().get("result", []):
                        offset = int(update["update_id"]) + 1
                        await _process_polled_update(update)
        except asyncio.CancelledError:
            _POLLING_STATE["status"] = "stopped"
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            failures = int(_POLLING_STATE.get("consecutive_failures") or 0) + 1
            detail = f"{type(exc).__name__}: {str(exc) or 'Telegram polling failed'}"
            _POLLING_STATE.update(
                status="retrying",
                last_error=detail[:240],
                consecutive_failures=failures,
            )
            LOGGER.warning(
                "Telegram polling unavailable; retrying in %ss error=%s",
                retry_seconds,
                detail,
            )
            await asyncio.sleep(retry_seconds)
            retry_seconds = min(60, retry_seconds * 2)
