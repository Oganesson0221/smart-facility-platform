#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

failures=0

pass() {
  printf '[ok]   %s\n' "$1"
}

fail() {
  printf '[fail] %s\n' "$1"
  failures=$((failures + 1))
}

openai_base_url() {
  local base="${1%/}"
  if [[ "$base" == */v1 ]]; then
    printf '%s' "$base"
  else
    printf '%s/v1' "$base"
  fi
}

check_health() {
  local name="$1"
  local url="$2"
  if curl --connect-timeout 2 --max-time 10 -fsS "$url" >/dev/null; then
    pass "${name}: ${url}"
  else
    fail "${name}: ${url}"
  fi
}

check_model() {
  local name="$1"
  local url="$2"
  local expected_model="$3"
  local api_key="${4:-}"
  local response
  local curl_args=(--connect-timeout 2 --max-time 10 -fsS)
  if [[ -n "$api_key" ]]; then
    curl_args+=(-H "Authorization: Bearer ${api_key}")
  fi
  if ! response="$(curl "${curl_args[@]}" "$url")"; then
    fail "${name} model list is unreachable: ${url}"
    return
  fi
  if printf '%s' "$response" | python3 -c \
    'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin).get("data", []); raise SystemExit(0 if expected in {str(item.get("id") or item.get("name")) for item in data if isinstance(item, dict)} else 1)' \
    "$expected_model"; then
    pass "${name} serves ${expected_model}"
  else
    fail "${name} is reachable but does not list ${expected_model}"
  fi
}

check_nemo_tool_call() {
  local base_url
  base_url="$(openai_base_url "${NEMO_AGENT_BASE_URL:-http://127.0.0.1:8010/v1}")"
  local url="${base_url}/chat/completions"
  local image_path="$(pwd)/tests/car4.jpeg"
  local payload response
  payload="$(printf '{"model":"%s","messages":[{"role":"system","content":"Do not answer directly. Call detect_image_objects exactly once with the supplied image_path and confidence_threshold=0.35. Return only the tool result."},{"role":"user","content":"Run detection for image_path=%s now."}],"temperature":0,"stream":false}' "${NEMO_AGENT_MODEL:-smart-facility-agent}" "$image_path")"
  local curl_args=(--connect-timeout 2 --max-time "${NEMO_AGENT_TIMEOUT_SECONDS:-120}" -fsS -X POST -H "Content-Type: application/json")
  if [[ -n "${NEMO_AGENT_API_KEY:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${NEMO_AGENT_API_KEY}")
  fi
  if ! response="$(curl "${curl_args[@]}" -d "$payload" "$url")"; then
    fail "NeMo detector tool call: ${url}"
    return
  fi
  if [[ "$response" == *'detections'* ]]; then
    pass "NeMo executed facility_object_detector"
  else
    fail "NeMo responded without a detector result"
  fi
}

check_switchyard_route() {
  local scenario="$1"
  local expected_model="$2"
  local response
  if ! response="$(curl --connect-timeout 2 --max-time "${SWITCHYARD_TIMEOUT_SECONDS:-180}" -fsS -X POST \
    "${app_url}/api/switchyard/diagnostics/${scenario}")"; then
    fail "Switchyard ${scenario} diagnostic request"
    return
  fi
  if printf '%s' "$response" | python3 -c \
    'import json,sys; expected=sys.argv[1]; payload=json.load(sys.stdin); raise SystemExit(0 if payload.get("selected_model") == expected and payload.get("decision_sources") else 1)' \
    "$expected_model"; then
    pass "Switchyard ${scenario} routes to ${expected_model} with an official decision source"
  else
    fail "Switchyard ${scenario} did not select ${expected_model} with routing evidence"
  fi
}

app_url="http://127.0.0.1:${APP_PORT:-8000}"
switchyard_url="${SWITCHYARD_BASE_URL:-http://127.0.0.1:4000}"
switchyard_url="${switchyard_url%/}"
[[ "$switchyard_url" == */v1 ]] && switchyard_url="${switchyard_url%/v1}"
llm_url="$(openai_base_url "${LLM_BASE_URL:-http://127.0.0.1:8001/v1}")"
vision_url="$(openai_base_url "${VISION_BASE_URL:-http://127.0.0.1:8002/v1}")"
nemo_url="$(openai_base_url "${NEMO_AGENT_BASE_URL:-http://127.0.0.1:8010/v1}")"

check_health "App" "${app_url}/api/health"
check_health "Text vLLM" "${llm_url%/v1}/health"
check_model "Text vLLM" "${llm_url}/models" "${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}" "${LLM_API_KEY:-}"
check_health "Vision vLLM" "${vision_url%/v1}/health"
check_model "Vision vLLM" "${vision_url}/models" "${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}" "${VISION_API_KEY:-${LLM_API_KEY:-}}"
check_health "Switchyard" "${switchyard_url}/health"
check_model "Switchyard route" "${switchyard_url}/v1/models" "${SWITCHYARD_MODEL:-switchyard/exitwatch-stage}" "${SWITCHYARD_API_KEY:-}"
check_health "NeMo Agent Toolkit" "${nemo_url%/v1}/health"
check_switchyard_route "routine" "${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
check_switchyard_route "critical" "${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
check_nemo_tool_call

if (( failures > 0 )); then
  printf '\n%d check(s) failed.\n' "$failures"
  exit 1
fi

printf '\nAll local AI checks passed.\n'
