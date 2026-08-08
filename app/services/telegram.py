import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.models import Incident, TelegramSubscriber
from app.services.llm import answer_sop_question
from app.services.sop import load_telegram_sop_reference, search_sops


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
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url)
            response.raise_for_status()
        result = response.json().get("result") or {}
        return {
            "reachable": True,
            "detail": "Telegram Bot API is reachable",
            "bot_username": result.get("username"),
        }
    except httpx.HTTPStatusError as exc:
        return {
            "reachable": False,
            "detail": f"Telegram returned HTTP {exc.response.status_code}",
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
        "mask_zone_iou": metadata.get("mask_zone_iou"),
        "sam_model": metadata.get("sam_model"),
        "yolo_box": metadata.get("yolo_box"),
        "yolo_class": metadata.get("yolo_class"),
        "yolo_confidence": metadata.get("yolo_confidence"),
        "vehicle_identifier": metadata.get("vehicle_identifier"),
        "vehicle_identifier_type": metadata.get("vehicle_identifier_type"),
        "vehicle_identifier_confidence": metadata.get("vehicle_identifier_confidence"),
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

    incident = _resolve_incident_for_question(db, question)
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
    sop_reference_path, sop_reference_text = load_telegram_sop_reference()
    reply = await answer_sop_question(
        question,
        incident_context,
        sop_reference_path,
        sop_reference_text,
        matched_sops,
    )
    return reply.strip()


async def handle_incoming_message(db: Session, message: dict) -> tuple[str, bool]:
    chat_id = str((message.get("chat") or {}).get("id", ""))
    raw_text = str(message.get("text") or "").strip()
    text = raw_text.lower()
    if text.startswith("/stop"):
        deactivate_subscriber(db, chat_id)
        return (
            "Smart Facility alerts are now disabled. Send /start to subscribe again.",
            False,
        )
    if text.startswith("/start"):
        register_subscriber(db, message)
        return (
            "You are subscribed to Smart Facility safety alerts. "
            "Annotated incident images will be sent here.\n\n"
            "You can also ask the bot questions like:\n"
            "- What should I do next for the latest incident?\n"
            "- What should I do next for INC-20260730-042724-593?\n"
            "- What does the fire exit obstruction SOP say?\n\n"
            "Send /stop to opt out.",
            True,
        )
    if text.startswith("/help"):
        register_subscriber(db, message)
        return (
            "Ask about the latest incident or a specific incident ID.\n"
            "Examples:\n"
            "- What should I do next for the latest incident?\n"
            "- What should I do next for INC-20260730-042724-593?\n"
            "- What does the restricted parking SOP say?\n\n"
            "The bot answers from the local SOP reference file and incident data.",
            True,
        )
    register_subscriber(db, message)
    return await answer_telegram_message(db, raw_text), True


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
    method = str(metadata.get("spatial_method") or "yolo_box_fallback")
    method_label = "SAM mask" if method == "sam_mask" else "YOLO bounding box"
    first_action = (
        str(incident.recommended_action).splitlines()[0].strip()
        if str(getattr(incident, "recommended_action", "")).strip()
        else "Confirm the location and notify Facilities Security."
    )
    vehicle_identifier = str(metadata.get("vehicle_identifier") or "").strip()
    vehicle_identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    vehicle_identifier_line = (
        f"Vehicle identifier: {vehicle_identifier}"
        f"{f' ({vehicle_identifier_type.replace('_', ' ')})' if vehicle_identifier_type and vehicle_identifier_type != 'none' else ''}\n"
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
        f"Blocked duration: {duration:.1f}s\n"
        f"Segmentation method: {method_label}\n"
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
    first_action = (
        str(getattr(incident, "recommended_action", "") or "").splitlines()[0].strip()
        or "Confirm the location and notify Facilities Security."
    )
    vehicle_identifier = str(metadata.get("vehicle_identifier") or "").strip()
    vehicle_identifier_type = str(metadata.get("vehicle_identifier_type") or "").strip()
    lines = [
        f"Incident: {incident.id}",
        f"Object: {incident.object_type}",
        *(
            [
                "Vehicle ID: "
                f"{vehicle_identifier}"
                f"{f' ({vehicle_identifier_type.replace('_', ' ')})' if vehicle_identifier_type and vehicle_identifier_type != 'none' else ''}"
            ]
            if vehicle_identifier
            else []
        ),
        f"YOLO confidence: {incident.confidence:.0%}",
        f"Object inside zone: {intrusion:.0%}",
        f"Exit area blocked: {blockage:.0%}",
        f"Blocked duration: {duration:.1f}s",
        f"Segmentation: {method_label}",
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
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
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
        async with httpx.AsyncClient(timeout=30) as client:
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
    if not is_configured():
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json={"chat_id": chat_id, "text": text})
    except httpx.HTTPError:
        return


async def answer_callback(callback_id: str, text: str):
    if not settings.telegram_bot_token:
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/answerCallbackQuery"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json={"callback_query_id": callback_id, "text": text})
    except httpx.HTTPError:
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
        return
    base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    offset: int | None = None
    while True:
        try:
            async with httpx.AsyncClient(
                timeout=settings.telegram_poll_timeout_seconds + 10
            ) as client:
                await client.post(
                    f"{base_url}/deleteWebhook",
                    json={"drop_pending_updates": False},
                )
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
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            await asyncio.sleep(5)
