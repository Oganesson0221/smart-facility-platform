#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data uploads evidence
if [[ -x ".app-venv/bin/python" ]]; then
  runtime_python=".app-venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  runtime_python=".venv/bin/python"
else
  runtime_python="python3"
fi
reload_args=()
if [[ "${APP_RELOAD:-false}" == "true" ]]; then
  reload_args=(--reload)
fi
exec "$runtime_python" -m uvicorn app.main:app \
  --host "${APP_HOST:-0.0.0.0}" \
  --port "${APP_PORT:-8000}" \
  "${reload_args[@]}"
