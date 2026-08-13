#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

nemo_host="${NEMO_AGENT_CHECK_HOST:-127.0.0.1}"
nemo_port="${NEMO_AGENT_PORT:-8010}"
nemo_start_timeout="${NEMO_START_TIMEOUT:-180}"

cleanup() {
  if [[ -n "${nemo_pid:-}" ]] && kill -0 "$nemo_pid" 2>/dev/null; then
    kill "$nemo_pid" 2>/dev/null || true
    wait "$nemo_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

./scripts/run-nemo-agent.sh &
nemo_pid=$!

echo "Starting NeMo agent on http://${nemo_host}:${nemo_port}"
elapsed=0
until curl --connect-timeout 2 -fsS "http://${nemo_host}:${nemo_port}/health" >/dev/null 2>&1; do
  if ! kill -0 "$nemo_pid" 2>/dev/null; then
    echo "NeMo Agent Toolkit exited during startup."
    wait "$nemo_pid" || true
    exit 1
  fi
  if (( elapsed >= nemo_start_timeout )); then
    echo "NeMo Agent Toolkit was not ready after ${nemo_start_timeout}s."
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

echo "NeMo Agent Toolkit is ready on http://${nemo_host}:${nemo_port}"
echo "Starting Smart Facility app on http://127.0.0.1:${APP_PORT:-8000}"

./scripts/run.sh
