import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote_plus

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import TelegramSubscriber
from app.services.telegram import register_subscriber, subscriber_chat_ids
from app.services.telegram import send_incident_alert


class TelegramTests(unittest.TestCase):
    def test_photo_alert_contains_actions_and_recommendation(self):
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
            summary="A vehicle is obstructing the protected fire exit.",
            recommended_action=(
                "Record the visible vehicle details and notify Facilities Security."
            ),
            evidence_image=None,
        )

        with patch("app.services.telegram.is_configured", return_value=True), patch(
            "app.services.telegram.settings.telegram_bot_token", "test-token"
        ), patch(
            "app.services.telegram.settings.telegram_alert_chat_id", "123456"
        ), patch("app.services.telegram.httpx.AsyncClient", client_factory):
            status, message_id = asyncio.run(send_incident_alert(incident))

        self.assertEqual(status, "sent")
        self.assertEqual(message_id, "731")
        body = unquote_plus(captured["body"].decode())
        self.assertIn("Required response", body)
        self.assertIn("A vehicle is obstructing", body)
        self.assertIn("Acknowledge", body)
        self.assertIn("False alarm", body)
        self.assertIn("INC-TEST-001", body)

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
            summary="A vehicle is obstructing the protected fire exit.",
            recommended_action="Notify Facilities Security.",
            evidence_image=None,
        )
        with patch("app.services.telegram.is_configured", return_value=True), patch(
            "app.services.telegram.settings.telegram_bot_token", "test-token"
        ), patch(
            "app.services.telegram.settings.telegram_alert_chat_id", "123456"
        ), patch("app.services.telegram.httpx.AsyncClient", client_factory):
            status, _ = asyncio.run(send_incident_alert(incident))
        self.assertEqual(status, "failed: could not connect to api.telegram.org")

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


if __name__ == "__main__":
    unittest.main()
