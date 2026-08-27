#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

log_dir="${STACK_LOG_DIR:-$(pwd)/logs}"
show_logs=false
if [[ "${1:-}" == "--tail" ]]; then
  show_logs=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: ./scripts/status-all.sh [--tail]"
  exit 2
fi

status_for() {
  local url="$1"
  if curl --connect-timeout 1 --max-time 3 -fsS "$url" >/dev/null 2>&1; then
    printf 'ready'
  else
    printf 'down'
  fi
}

print_service() {
  local name="$1"
  local port="$2"
  local health_path="$3"
  local log_name="$4"
  local state
  state="$(status_for "http://127.0.0.1:${port}${health_path}")"
  printf '%-18s %-8s %-8s %s\n' "$name" "$port" "$state" "$log_dir/$log_name"
}

printf '%-18s %-8s %-8s %s\n' "SERVICE" "PORT" "STATUS" "LOG"
print_service "FastAPI" "${APP_PORT:-8000}" "/api/health" "app.log"
print_service "Qwen vLLM" "${LLM_VLLM_PORT:-8001}" "/health" "text-vllm.log"
print_service "Nemotron vLLM" "${VISION_VLLM_PORT:-8002}" "/health" "vision-vllm.log"
print_service "NeMo Agent" "${NEMO_AGENT_PORT:-8010}" "/health" "nemo-agent.log"

if [[ "$show_logs" == "true" ]]; then
  log_lines="${STATUS_LOG_LINES:-20}"
  for log_name in app.log text-vllm.log vision-vllm.log nemo-agent.log; do
    echo
    echo "==> $log_dir/$log_name <=="
    tail -n "$log_lines" "$log_dir/$log_name" 2>/dev/null || echo "No log file yet."
  done
fi
