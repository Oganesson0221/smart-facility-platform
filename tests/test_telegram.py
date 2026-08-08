import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import unquote_plus

import httpx
import numpy as np
import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.database import Base
from app.models import Incident, TelegramSubscriber
from app.services.telegram import (
    _alert_caption,
    handle_incoming_message,
    register_subscriber,
    send_incident_alert,
    subscriber_chat_ids,
)


class TelegramTests(unittest.TestCase):
    def test_text_alert_contains_required_response(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 731}},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        incident = SimpleNamespace(
            id="INC-TEST-001",
            facility="Building A",
            zone="Fire Exit South",
            object_type="vehicle",
            event_type="exit_blocked",
            confidence=0.91,
            duration_seconds=12.4,
            created_at=datetime(2026, 7, 30, 4, 30, 0, tzinfo=timezone.utc),
            summary="A vehicle is obstructing the protected fire exit.",
            recommended_action=(
                "Record the visible vehicle details and notify Facilities Security."
            ),
            evidence_image=None,
            incident_metadata={
                "spatial_method": "yolo_box_fallback",
                "vehicle_identifier": "ABC1234",
                "vehicle_identifier_type": "license_plate",
            },
        )

        with patch("app.services.telegram.is_configured", return_value=True), patch(
            "app.services.telegram.settings.telegram_bot_token", "test-token"
        ), patch(
            "app.services.telegram.settings.telegram_alert_chat_id", "123456"
        ), patch("app.services.telegram.httpx.AsyncClient", client_factory):
            status, message_id = asyncio.run(send_incident_alert(incident))

        self.assertEqual(status, "sent")
        self.assertEqual(message_id, "731")
        body = unquote_plus(captured["body"].decode("latin1"))
        self.assertIn("Required response", body)
        self.assertIn("Detected: 2026-07-30 04:30:00 UTC", body)
        self.assertIn("A vehicle is obstructing", body)
        self.assertIn("Vehicle identifier: ABC1234 (license plate)", body)
        self.assertIn("Acknowledge", body)
        self.assertIn("False alarm", body)
        self.assertIn("INC-TEST-001", body)

    def test_photo_alert_uses_compact_caption(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 741}},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            evidence_dir = Path(folder) / "evidence"
            evidence_dir.mkdir()
            evidence_path = evidence_dir / "alert.jpg"
            image = np.full((40, 40, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(evidence_path), image)
            incident = SimpleNamespace(
                id="INC-TEST-001-PHOTO",
                facility="Building A",
                zone="Fire Exit South",
                object_type="vehicle",
                event_type="exit_blocked",
                confidence=0.91,
                duration_seconds=12.4,
                created_at=datetime(2026, 7, 30, 4, 30, 0, tzinfo=timezone.utc),
                summary="A vehicle is obstructing the protected fire exit.",
                recommended_action=(
                    "1. Notify Facilities Security immediately and preserve the evidence image."
                ),
                evidence_image="evidence/alert.jpg",
                incident_metadata={
                    "spatial_method": "sam_mask",
                    "object_intrusion_ratio": 0.42,
                    "exit_blockage_ratio": 0.18,
                    "blocked_duration_seconds": 12.4,
                    "vehicle_identifier": "ABC1234",
                    "vehicle_identifier_type": "license_plate",
                },
            )
            with patch("app.services.telegram.is_configured", return_value=True), patch(
                "app.services.telegram.settings.telegram_bot_token", "test-token"
            ), patch(
                "app.services.telegram.settings.telegram_alert_chat_id", "123456"
            ), patch("app.services.telegram.ROOT", Path(folder)), patch(
                "app.services.telegram.httpx.AsyncClient", client_factory
            ):
                status, message_id = asyncio.run(send_incident_alert(incident))

        self.assertEqual(status, "sent")
        self.assertEqual(message_id, "741")
        body = unquote_plus(captured["body"].decode("latin1"))
        self.assertIn("Segmentation: SAM mask", body)
        self.assertIn("Vehicle ID: ABC1234 (license plate)", body)
        self.assertIn("YOLO confidence: 91%", body)
        self.assertNotIn("Required response", body)
        self.assertIn("/sendPhoto", captured["url"])

    def test_photo_alert_retries_without_caption_if_telegram_rejects_caption(self):
        calls = []

        def handler(request: httpx.Request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    400,
                    json={"ok": False, "description": "Bad Request: caption is too long"},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 742}},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            evidence_dir = Path(folder) / "evidence"
            evidence_dir.mkdir()
            evidence_path = evidence_dir / "alert.jpg"
            image = np.full((40, 40, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(evidence_path), image)
            incident = SimpleNamespace(
                id="INC-TEST-001-RETRY",
                facility="Building A",
                zone="Fire Exit South",
                object_type="vehicle",
                event_type="exit_blocked",
                confidence=0.91,
                duration_seconds=12.4,
                created_at=datetime(2026, 7, 30, 4, 30, 0, tzinfo=timezone.utc),
                summary="A vehicle is obstructing the protected fire exit.",
                recommended_action="1. Notify Facilities Security immediately.",
                evidence_image="evidence/alert.jpg",
                incident_metadata={"spatial_method": "sam_mask"},
            )
            with patch("app.services.telegram.is_configured", return_value=True), patch(
                "app.services.telegram.settings.telegram_bot_token", "test-token"
            ), patch(
                "app.services.telegram.settings.telegram_alert_chat_id", "123456"
            ), patch("app.services.telegram.ROOT", Path(folder)), patch(
                "app.services.telegram.httpx.AsyncClient", client_factory
            ):
                status, message_id = asyncio.run(send_incident_alert(incident))

        self.assertEqual(status, "sent")
        self.assertEqual(message_id, "742")
        self.assertEqual(len(calls), 2)
        self.assertIn("/sendPhoto", str(calls[0].url))
        self.assertIn("/sendPhoto", str(calls[1].url))

    def test_connection_failure_is_actionable(self):
        def handler(request: httpx.Request):
            raise httpx.ConnectError("blocked", request=request)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        incident = SimpleNamespace(
            id="INC-TEST-002",
            facility="Building A",
            zone="Fire Exit South",
            object_type="vehicle",
            event_type="exit_blocked",
            confidence=0.91,
            duration_seconds=12,
            created_at=datetime(2026, 7, 30, 4, 31, 0, tzinfo=timezone.utc),
            summary="A vehicle is obstructing the protected fire exit.",
            recommended_action="Notify Facilities Security.",
            evidence_image=None,
            incident_metadata={"spatial_method": "yolo_box_fallback"},
        )
        with patch("app.services.telegram.is_configured", return_value=True), patch(
            "app.services.telegram.settings.telegram_bot_token", "test-token"
        ), patch(
            "app.services.telegram.settings.telegram_alert_chat_id", "123456"
        ), patch("app.services.telegram.httpx.AsyncClient", client_factory):
            status, _ = asyncio.run(send_incident_alert(incident))
        self.assertEqual(status, "failed: could not connect to api.telegram.org")

    def test_alert_uses_saved_annotated_evidence_image_when_available(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 900}},
                request=request,
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            evidence_dir = Path(folder) / "evidence"
            evidence_dir.mkdir()
            evidence_path = evidence_dir / "telegram-sam.jpg"
            image = np.full((40, 40, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(evidence_path), image)
            incident = SimpleNamespace(
                id="INC-TEST-003",
                facility="Building A",
                zone="Fire Exit South",
                object_type="vehicle",
                event_type="exit_blocked",
                confidence=0.91,
                duration_seconds=12.4,
                created_at=datetime(2026, 7, 30, 4, 30, 0, tzinfo=timezone.utc),
                summary="A vehicle is obstructing the protected fire exit.",
                recommended_action="1. Notify Facilities Security.",
                evidence_image="evidence/telegram-sam.jpg",
                incident_metadata={"spatial_method": "sam_mask"},
            )
            with patch("app.services.telegram.is_configured", return_value=True), patch(
                "app.services.telegram.settings.telegram_bot_token", "test-token"
            ), patch(
                "app.services.telegram.settings.telegram_alert_chat_id", "123456"
            ), patch("app.services.telegram.ROOT", Path(folder)), patch(
                "app.services.telegram.httpx.AsyncClient", client_factory
            ):
                status, message_id = asyncio.run(send_incident_alert(incident))

        self.assertEqual(status, "sent")
        self.assertEqual(message_id, "900")
        self.assertIn("/sendPhoto", captured["url"])

    def test_compact_caption_is_bounded_for_long_recommendations(self):
        incident = SimpleNamespace(
            id="INC-LONG-001",
            facility="Building A",
            zone="Fire Exit South",
            object_type="vehicle",
            event_type="exit_blocked",
            confidence=0.91,
            duration_seconds=12.4,
            created_at=datetime(2026, 7, 30, 4, 30, 0, tzinfo=timezone.utc),
            summary="A vehicle is obstructing the protected fire exit.",
            recommended_action=" ".join(["Notify Facilities Security immediately."] * 80),
            incident_metadata={
                "spatial_method": "sam_mask",
                "object_intrusion_ratio": 0.42,
                "exit_blockage_ratio": 0.18,
                "blocked_duration_seconds": 12.4,
                "vehicle_identifier": "ABC1234",
                "vehicle_identifier_type": "license_plate",
            },
        )
        caption = _alert_caption(incident)
        self.assertLessEqual(len(caption), 900)
        self.assertIn("Incident: INC-LONG-001", caption)
        self.assertIn("Vehicle ID: ABC1234 (license plate)", caption)

    def test_started_chats_are_registered_as_alert_subscribers(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            register_subscriber(
                db,
                {
                    "chat": {"id": 7001},
                    "from": {"username": "facilities_user", "first_name": "Pat"},
                    "text": "/start",
                },
            )
            subscriber = db.get(TelegramSubscriber, "7001")
            self.assertTrue(subscriber.active)
            self.assertEqual(subscriber.username, "facilities_user")
            with patch("app.services.telegram.settings.telegram_alert_chat_id", ""):
                with patch("app.services.telegram.settings.user_id", "8002"):
                    self.assertEqual(subscriber_chat_ids(db), ["7001", "8002"])
        finally:
            db.close()

    def test_bot_username_is_not_used_as_alert_recipient(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            with patch(
                "app.services.telegram.settings.telegram_alert_chat_id",
                "SmartFacilityAssistant_bot",
            ), patch("app.services.telegram.settings.user_id", "8002"):
                self.assertEqual(subscriber_chat_ids(db), ["8002"])
        finally:
            db.close()

    def test_settings_accepts_telegram_user_id_alias(self):
        settings = Settings(
            _env_file=None,
            TELEGRAM_BOT_TOKEN="token",
            TELEGRAM_USER_ID="7001",
        )
        self.assertEqual(settings.user_id, "7001")

    def test_start_command_returns_interactive_help(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            reply, subscribed = asyncio.run(
                handle_incoming_message(
                    db,
                    {
                        "chat": {"id": 7001},
                        "from": {"username": "facilities_user", "first_name": "Pat"},
                        "text": "/start",
                    },
                )
            )
            self.assertTrue(subscribed)
            self.assertIn("What should I do next", reply)
            subscriber = db.get(TelegramSubscriber, "7001")
            self.assertTrue(subscriber.active)
        finally:
            db.close()

    def test_freeform_message_routes_to_sop_assistant(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            with patch(
                "app.services.telegram.answer_sop_question",
                AsyncMock(return_value="1. Notify Facilities Security."),
            ):
                reply, subscribed = asyncio.run(
                    handle_incoming_message(
                        db,
                        {
                            "chat": {"id": 7001},
                            "from": {"username": "facilities_user", "first_name": "Pat"},
                            "text": "What should I do next for the latest incident?",
                        },
                    )
                )
            self.assertTrue(subscribed)
            self.assertEqual(reply, "1. Notify Facilities Security.")
            self.assertTrue(db.get(TelegramSubscriber, "7001").active)
        finally:
            db.close()

    def test_latest_incident_is_passed_to_assistant_context(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        try:
            incident = Incident(
                id="INC-20260730-050000-000",
                camera_id="cam-1",
                facility="Building A",
                zone="South Access Zone",
                event_type="exit_blocked",
                object_type="car",
                confidence=0.91,
                overlap=0.4,
                duration_seconds=12.0,
                status="open",
                severity="high",
                first_seen=datetime(2026, 7, 30, 5, 0, 0, tzinfo=timezone.utc),
                last_seen=datetime(2026, 7, 30, 5, 1, 0, tzinfo=timezone.utc),
                summary="A car is blocking the fire exit.",
                recommended_action="1. Notify Facilities Security.",
                created_at=datetime(2026, 7, 30, 5, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 7, 30, 5, 0, 0, tzinfo=timezone.utc),
            )
            db.add(incident)
            db.commit()
            captured = {}

            async def fake_answer(question, incident, sop_reference_path, sop_reference_text, sops):
                captured["question"] = question
                captured["incident"] = incident
                captured["path"] = sop_reference_path
                return "Use the latest incident SOP."

            with patch("app.services.telegram.answer_sop_question", AsyncMock(side_effect=fake_answer)):
                reply, _ = asyncio.run(
                    handle_incoming_message(
                        db,
                        {
                            "chat": {"id": 7001},
                            "from": {"username": "facilities_user", "first_name": "Pat"},
                            "text": "What should I do next for the latest incident?",
                        },
                    )
                )
            self.assertEqual(reply, "Use the latest incident SOP.")
            self.assertEqual(captured["incident"]["incident_id"], "INC-20260730-050000-000")
            self.assertEqual(captured["path"], "sops/telegram_assistant_reference.txt")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
