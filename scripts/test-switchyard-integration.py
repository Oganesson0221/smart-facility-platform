#!/usr/bin/env python3
"""Exercise the real switchyard-server with deterministic local mock targets."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.switchyard_client import SwitchyardClient


EFFICIENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CAPABLE_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
ROUTE = "switchyard/exitwatch-stage"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 - stdlib handler contract
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        model = str(request.get("model") or "")
        if self.path != "/v1/chat/completions" or model not in {
            EFFICIENT_MODEL,
            CAPABLE_MODEL,
        }:
            self._send(404, {"error": {"message": "unknown mock request"}})
            return
        self.server.seen_models.append(model)
        self._send(
            200,
            {
                "id": f"mock-{len(self.server.seen_models)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"served by {model}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    def log_message(self, _format, *_args):
        return

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), MockOpenAIHandler)
        self.seen_models: list[str] = []


def find_server() -> str:
    candidates = [
        os.environ.get("SWITCHYARD_SERVER_BIN", ""),
        shutil.which("switchyard-server") or "",
        str(Path.home() / ".cargo" / "bin" / "switchyard-server"),
    ]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "switchyard-server not found; install it with ./scripts/setup-switchyard.sh"
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, process: subprocess.Popen, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"switchyard-server exited early:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"switchyard-server did not become healthy at {url}")


async def verify(base_url: str):
    client = SwitchyardClient(
        base_url=base_url,
        model=ROUTE,
        api_key="",
        timeout_seconds=10,
    )
    expectations = {
        "routine": (EFFICIENT_MODEL, "tests_passed"),
        "exploration": (CAPABLE_MODEL, "dimensions"),
        "critical": (CAPABLE_MODEL, "override"),
    }
    for scenario, (expected_model, expected_source) in expectations.items():
        result = await client.diagnose(scenario)
        if result.selected_model != expected_model:
            raise AssertionError(
                f"{scenario}: expected {expected_model}, got {result.selected_model}"
            )
        if expected_source not in result.decision_sources:
            stats = await client._get_json("/v1/stats")
            raise AssertionError(
                f"{scenario}: expected decision source {expected_source}, "
                f"got {result.decision_sources or 'no isolated stats delta'}; "
                f"official_stats={json.dumps(stats, sort_keys=True)}"
            )
        print(
            f"VERIFIED scenario={scenario} selected_model={result.selected_model} "
            f"decision_source={expected_source}"
        )


def main():
    server_binary = find_server()
    upstream = MockOpenAIServer()
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    switchyard_port = free_port()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="exitwatch-switchyard-") as folder:
            root = Path(folder)
            config = root / "routes.toml"
            config.write_text(
                f'''schema_version = 1

[llm_clients.local_mock]
format = "openai_chat"
base_url = "http://127.0.0.1:{upstream.server_port}/v1"

[targets.efficient]
id = "{EFFICIENT_MODEL}"
llm_client = "local_mock"

[targets.capable]
id = "{CAPABLE_MODEL}"
llm_client = "local_mock"

[routes.exitwatch_stage]
id = "{ROUTE}"
type = "stage_router"
efficient_target = "efficient"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.5
''',
                encoding="utf-8",
            )
            subprocess.run(
                [server_binary, "--config", str(config), "--dry-run"],
                check=True,
            )
            process = subprocess.Popen(
                [
                    server_binary,
                    "--config",
                    str(config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(switchyard_port),
                    "--routing-log-file",
                    str(root / "routing.jsonl"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            base_url = f"http://127.0.0.1:{switchyard_port}"
            wait_for_health(f"{base_url}/health", process)
            asyncio.run(verify(base_url))
            print(f"VERIFIED real switchyard-server requests={upstream.seen_models}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    main()
