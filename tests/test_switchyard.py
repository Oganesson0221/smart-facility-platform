import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.services.switchyard_client import SELECTED_MODEL_HEADER
from app.services.switchyard_client import SwitchyardClient
from app.services.switchyard_client import SwitchyardError
from app.services.switchyard_client import _decision_delta
from app.services.switchyard_client import diagnostic_messages
from app.services.switchyard_client import latest_routing_record


class SwitchyardSignalTests(unittest.TestCase):
    def test_routine_trajectory_contains_write_and_test_result(self):
        messages = diagnostic_messages("routine")
        calls = [
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls", [])
        ]
        self.assertEqual(calls, ["write_file", "Bash"])
        self.assertIn("passed", messages[-1]["content"])
        self.assertEqual(messages[-1]["role"], "tool")

    def test_exploration_trajectory_is_deep_and_carries_hard_error_text(self):
        messages = diagnostic_messages("exploration")
        self.assertGreaterEqual(len(messages), 8)
        self.assertTrue(any(message.get("role") == "tool" for message in messages))
        self.assertIn("Traceback", messages[-1]["content"])
        self.assertIn("ValueError", messages[-1]["content"])

    def test_critical_trajectory_uses_official_critical_error_pattern(self):
        messages = diagnostic_messages("critical")
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertIn("connection refused", messages[-1]["content"])

    def test_invalid_diagnostic_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "routine, exploration, critical"):
            diagnostic_messages("unknown")

    def test_decision_delta_reports_only_new_selected_model_source(self):
        before = {
            "algorithm_stats": {
                "stage_router": {
                    "routing_decisions": {
                        "fall_open": {"targets": {"Qwen": 2}},
                    }
                }
            }
        }
        after = {
            "algorithm_stats": {
                "stage_router": {
                    "routing_decisions": {
                        "fall_open": {"targets": {"Qwen": 2}},
                        "override": {"targets": {"Nemotron": 1}},
                    }
                }
            }
        }
        self.assertEqual(_decision_delta(before, after, "Nemotron"), ("override",))

    def test_latest_routing_record_reads_official_jsonl_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "routing.jsonl"
            path.write_text(
                "not json\n"
                + json.dumps(
                    {
                        "ts": "2026-08-27T12:00:00Z",
                        "model": "Qwen",
                        "tier": "weak",
                        "total_tokens": 7,
                        "internal": "not exposed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "app.services.switchyard_client.settings.switchyard_routing_log_path",
                str(path),
            ):
                record = latest_routing_record()
        self.assertEqual(record["model"], "Qwen")
        self.assertEqual(record["tier"], "weak")
        self.assertNotIn("internal", record)


class SwitchyardClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_observability_proxy_methods_use_internal_switchyard_routes(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="switchyard_requests_total 1\n")
            return httpx.Response(200, json={"path": request.url.path})

        client = SwitchyardClient(
            base_url="http://switchyard.test",
            transport=httpx.MockTransport(handler),
        )
        self.assertEqual((await client.models())["path"], "/v1/models")
        self.assertEqual((await client.stats())["path"], "/v1/stats")
        self.assertIn("switchyard_requests_total", await client.metrics())
        self.assertEqual(seen, ["/v1/models", "/v1/stats", "/metrics"])

    async def test_chat_completion_extracts_official_selected_model_header(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "switchyard/exitwatch-stage")
            self.assertEqual(
                request.headers.get("x-switchyard-session-id"),
                "test-session",
            )
            return httpx.Response(
                200,
                headers={SELECTED_MODEL_HEADER: "Qwen/Qwen2.5-7B-Instruct"},
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        client = SwitchyardClient(
            base_url="http://switchyard.test/v1",
            model="switchyard/exitwatch-stage",
            transport=httpx.MockTransport(handler),
        )
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            session_id="test-session",
        )
        self.assertEqual(result.selected_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(result.payload["choices"][0]["message"]["content"], "ok")

    async def test_chat_completion_rejects_missing_routing_header(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        client = SwitchyardClient(
            base_url="http://switchyard.test",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(SwitchyardError, SELECTED_MODEL_HEADER):
            await client.chat_completion(
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_health_error_is_actionable(self):
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = SwitchyardClient(
            base_url="http://switchyard.test",
            transport=httpx.MockTransport(handler),
        )
        status = await client.health()
        self.assertFalse(status["reachable"])
        self.assertIn("connection refused", status["detail"])


if __name__ == "__main__":
    unittest.main()
