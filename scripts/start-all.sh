#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

runtime_dir="${STACK_RUNTIME_DIR:-$(pwd)/.runtime}"
log_dir="${STACK_LOG_DIR:-$(pwd)/logs}"
mkdir -p "$runtime_dir" "$log_dir"

llm_port="${LLM_VLLM_PORT:-8001}"
vision_port="${VISION_VLLM_PORT:-8002}"
nemo_port="${NEMO_AGENT_PORT:-8010}"
app_port="${APP_PORT:-8000}"

process_start_time() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  sed 's/.*) //' "/proc/${pid}/stat" | awk '{print $20}'
}

record_process() {
  local name="$1"
  local pid="$2"
  local started
  started="$(process_start_time "$pid")"
  printf '%s %s\n' "$pid" "$started" >"$runtime_dir/${name}.pid"
}

tracked_process_running() {
  local name="$1"
  local record="$runtime_dir/${name}.pid"
  local pid expected actual
  [[ -r "$record" ]] || return 1
  read -r pid expected <"$record"
  kill -0 "$pid" 2>/dev/null || return 1
  actual="$(process_start_time "$pid" || true)"
  [[ -n "$actual" && "$actual" == "$expected" ]]
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local timeout="$4"
  local elapsed=0

  while (( elapsed < timeout )); do
    if curl --connect-timeout 2 -fsS "$url" >/dev/null 2>&1; then
      echo "[ready] $name: $url"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[failed] $name exited during startup. Last log lines:"
      tail -n 30 "$log_dir/${name}.log" 2>/dev/null || true
      return 1
    fi
    if (( elapsed > 0 && elapsed % 30 == 0 )); then
      echo "[wait] $name is still loading (${elapsed}s/${timeout}s)"
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "[timeout] $name was not ready after ${timeout}s. See $log_dir/${name}.log"
  return 1
}

start_service() {
  local name="$1"
  local health_url="$2"
  local timeout="$3"
  shift 3

  if tracked_process_running "$name"; then
    local existing_pid
    read -r existing_pid _ <"$runtime_dir/${name}.pid"
    echo "[running] $name (PID $existing_pid)"
    wait_for_url "$name" "$health_url" "$existing_pid" "$timeout"
    return
  fi

  if curl --connect-timeout 2 -fsS "$health_url" >/dev/null 2>&1; then
    echo "[external] $name is already available; it will not be stopped by stop-all.sh"
    rm -f "$runtime_dir/${name}.pid"
    return
  fi

  rm -f "$runtime_dir/${name}.pid"
  echo "[start] $name -> $log_dir/${name}.log"
  nohup setsid "$@" >"$log_dir/${name}.log" 2>&1 </dev/null &
  local pid=$!
  record_process "$name" "$pid"
  wait_for_url "$name" "$health_url" "$pid" "$timeout"
}

echo "Starting Smart Facility Platform"
echo "Services load sequentially to limit peak disk I/O."
start_service "text-vllm" "http://127.0.0.1:${llm_port}/health" "${TEXT_START_TIMEOUT:-900}" ./scripts/run-vllm-llm.sh
start_service "vision-vllm" "http://127.0.0.1:${vision_port}/health" "${VISION_START_TIMEOUT:-1200}" ./scripts/run-vllm-vision.sh
start_service "nemo-agent" "http://127.0.0.1:${nemo_port}/health" "${NEMO_START_TIMEOUT:-180}" ./scripts/run-nemo-agent.sh
start_service "app" "http://127.0.0.1:${app_port}/api/health" "${APP_START_TIMEOUT:-180}" ./scripts/run.sh

echo
echo "Smart Facility Platform is ready: http://127.0.0.1:${app_port}"
printf '%-18s %-8s %s\n' "SERVICE" "PORT" "LOG"
printf '%-18s %-8s %s\n' "Qwen vLLM" "$llm_port" "$log_dir/text-vllm.log"
printf '%-18s %-8s %s\n' "Nemotron vLLM" "$vision_port" "$log_dir/vision-vllm.log"
printf '%-18s %-8s %s\n' "NeMo Agent" "$nemo_port" "$log_dir/nemo-agent.log"
printf '%-18s %-8s %s\n' "FastAPI" "$app_port" "$log_dir/app.log"
echo "Status: ./scripts/status-all.sh"
echo "Stop: ./scripts/stop-all.sh"

if [[ "${START_RUN_CHECKS:-false}" == "true" ]]; then
  ./scripts/check-local-ai.sh
fi
