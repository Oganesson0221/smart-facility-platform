#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data uploads evidence
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$(pwd)/.yolo-config}"
mkdir -p "$YOLO_CONFIG_DIR"

can_run_python() {
  local candidate="$1"
  command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]] || return 1
  "$candidate" -V >/dev/null 2>&1
}

runtime_python=""
for candidate in ".venv/bin/python" ".app-venv/bin/python" "python3"; do
  if can_run_python "$candidate"; then
    runtime_python="$candidate"
    break
  fi
done

if [[ -z "$runtime_python" ]]; then
  echo "No working Python runtime found. Checked .app-venv, .venv, and python3."
  exit 1
fi

reload_args=()
if [[ "${APP_RELOAD:-false}" == "true" ]]; then
  reload_args=(--reload)
fi
exec "$runtime_python" -m uvicorn app.main:app \
  --host "${APP_HOST:-0.0.0.0}" \
  --port "${APP_PORT:-8000}" \
  "${reload_args[@]}"
