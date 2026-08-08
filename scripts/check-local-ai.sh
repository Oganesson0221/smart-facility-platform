#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

check_health() {
  local name="$1"
  local url="$2"
  if curl -sf "$url" >/dev/null; then
    echo "[ok]   ${name}: ${url}"
  else
    echo "[fail] ${name}: ${url}"
  fi
}

check_models() {
  local name="$1"
  local url="$2"
  local api_key="${3:-}"
  if [[ -n "$api_key" ]]; then
    if curl -sf -H "Authorization: Bearer ${api_key}" "$url" >/dev/null; then
      echo "[ok]   ${name} models: ${url}"
    else
      echo "[fail] ${name} models: ${url}"
    fi
  else
    if curl -sf "$url" >/dev/null; then
      echo "[ok]   ${name} models: ${url}"
    else
      echo "[fail] ${name} models: ${url}"
    fi
  fi
}

check_nemo_tool_call() {
  local url="http://127.0.0.1:8010/v1/chat/completions"
  local image_path
  image_path="$(pwd)/lab_images/car4.jpg"
  local payload
  payload="$(printf '{"model":"%s","messages":[{"role":"system","content":"Do not answer directly. Call detect_image_objects exactly once with the supplied image_path and confidence_threshold=0.35. Return only the tool result."},{"role":"user","content":"Run detection for image_path=%s now."}],"temperature":0,"stream":false}' "${NEMO_AGENT_MODEL:-smart-facility-agent}" "$image_path")"
  local response
  if ! response="$(curl -sf -X POST -H "Content-Type: application/json" -d "$payload" "$url")"; then
    echo "[fail] NeMo tool call: ${url}"
    return
  fi
  if [[ "$response" == *'detections'* ]]; then
    echo "[ok]   NeMo detector tool call"
  else
    echo "[fail] NeMo responded but did not execute the detector tool"
  fi
}

check_health "App" "http://127.0.0.1:8000/api/health"
check_health "Text vLLM" "http://127.0.0.1:8001/health"
check_models "Text vLLM" "http://127.0.0.1:8001/v1/models" "${LLM_API_KEY:-}"
check_health "Vision vLLM" "http://127.0.0.1:8002/health"
check_models "Vision vLLM" "http://127.0.0.1:8002/v1/models" "${VISION_API_KEY:-${LLM_API_KEY:-}}"
check_health "NeMo" "http://127.0.0.1:8010/health"
check_nemo_tool_call
