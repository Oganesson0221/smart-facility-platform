import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.models import TelegramSubscriber


def is_configured() -> bool:
    return bool(settings.telegram_bot_token)


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
    return list(dict.fromkeys(str(chat_id) for chat_id in chat_ids if chat_id))


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
    event_name = (
        "ACCESS OBSTRUCTION"
        if incident.event_type == "exit_blocked"
        else incident.object_type.replace("_", " ").upper()
    )
    return (
        f"🚨 {event_name} VIOLATION DETECTED\n\n"
        f"Location: {incident.facility} – {incident.zone}\n"
        f"Violation: {incident.summary}\n"
        f"Confidence: {incident.confidence:.0%}\n"
        f"Incident: {incident.id}\n\n"
        f"Required response:\n{incident.recommended_action}"
    )


async def _send_to_chat(
    client: httpx.AsyncClient,
    chat_id: str,
    incident,
    evidence: Path | None,
) -> str:
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    data = {
        "chat_id": chat_id,
        "caption": _alert_text(incident),
        "reply_markup": json.dumps(_keyboard(incident.id)),
    }
    if evidence is not None:
        with evidence.open("rb") as image:
            response = await client.post(
                f"{url}/sendPhoto",
                data=data,
                files={"photo": (evidence.name, image, "image/jpeg")},
            )
    else:
        data["text"] = data.pop("caption")
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
    from app.models import Incident

    message = update.get("message")
    callback = update.get("callback_query")
    db = SessionLocal()
    try:
        if message:
            chat_id = str((message.get("chat") or {}).get("id", ""))
            text = str(message.get("text") or "").strip().lower()
            if text.startswith("/stop"):
                deactivate_subscriber(db, chat_id)
                reply = (
                    "Smart Facility alerts are now disabled. "
                    "Send /start to subscribe again."
                )
            else:
                register_subscriber(db, message)
                reply = (
                    "You are subscribed to Smart Facility safety alerts. "
                    "Annotated incident images will be sent here. Send /stop to opt out."
                )
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
