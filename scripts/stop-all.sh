#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

runtime_dir="${STACK_RUNTIME_DIR:-$(pwd)/.runtime}"
stop_timeout="${STACK_STOP_TIMEOUT:-30}"

process_start_time() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  sed 's/.*) //' "/proc/${pid}/stat" | awk '{print $20}'
}

stop_service() {
  local name="$1"
  local record="$runtime_dir/${name}.pid"
  local pid expected actual elapsed=0

  if [[ ! -r "$record" ]]; then
    echo "[skip] $name was not launched by start-all.sh"
    return
  fi

  read -r pid expected <"$record"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[stale] $name is no longer running"
    rm -f "$record"
    return
  fi

  actual="$(process_start_time "$pid" || true)"
  if [[ -z "$actual" || "$actual" != "$expected" ]]; then
    echo "[safe] $name PID $pid was reused; not stopping it"
    rm -f "$record"
    return
  fi

  echo "[stop] $name (PID $pid)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  while kill -0 "$pid" 2>/dev/null && (( elapsed < stop_timeout )); do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "[force] $name did not stop after ${stop_timeout}s"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$record"
}

# Stop consumers before their model dependencies.
stop_service "app"
stop_service "nemo-agent"
stop_service "switchyard"
stop_service "vision-vllm"
stop_service "text-vllm"

echo "Smart Facility Platform stopped."
