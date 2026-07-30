#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
  if [[ -n "${nemo_pid:-}" ]] && kill -0 "$nemo_pid" 2>/dev/null; then
    kill "$nemo_pid" 2>/dev/null || true
    wait "$nemo_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

./scripts/run-nemo-agent.sh &
nemo_pid=$!

echo "Started NeMo agent on http://127.0.0.1:${NEMO_AGENT_PORT:-8010}"
echo "Starting Smart Facility app on http://127.0.0.1:${APP_PORT:-8000}"

./scripts/run.sh
