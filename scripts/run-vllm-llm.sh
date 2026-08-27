#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env
export PATH="$(pwd)/.nemo-venv/bin:$(pwd)/.venv/bin:$HOME/.local/bin:${PATH}"

find_vllm() {
  local candidate
  for candidate in \
    "${VLLM_BIN:-}" \
    "$(command -v vllm 2>/dev/null || true)" \
    "$HOME/.local/bin/vllm" \
    ".venv/bin/vllm" \
    ".nemo-venv/bin/vllm"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

vllm_bin="$(find_vllm || true)"
if [[ -z "${vllm_bin:-}" ]]; then
  echo "No vLLM binary was found."
  echo "Install vllm in .nemo-venv, .venv, or ~/.local/bin, or set VLLM_BIN explicitly."
  echo "Recommended fix: ./scripts/setup.sh"
  exit 1
fi

if [[ "${ALLOW_HF_OFFLINE:-false}" != "true" ]]; then
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

host="${LLM_VLLM_HOST:-0.0.0.0}"
port="${LLM_VLLM_PORT:-8001}"
served_model_name="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
default_local_source="$(pwd)/models/${served_model_name}"
if [[ -n "${LLM_MODEL_SOURCE:-}" ]]; then
  model_source="$LLM_MODEL_SOURCE"
elif [[ -f "$default_local_source/config.json" ]]; then
  model_source="$default_local_source"
else
  model_source="$served_model_name"
fi
max_model_len="${LLM_MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${LLM_GPU_MEMORY_UTILIZATION:-0.35}"
tool_call_parser="${LLM_TOOL_CALL_PARSER:-}"
reasoning_parser="${LLM_REASONING_PARSER:-}"
llm_api_key="${LLM_API_KEY:-}"
if [[ "$llm_api_key" == hf_* ]]; then
  echo "Ignoring a Hugging Face token supplied as LLM_API_KEY." >&2
  echo "Use HF_TOKEN for downloads and leave the local vLLM API key empty." >&2
  llm_api_key=""
fi

looks_like_qwen25() {
  local value="${1:-}"
  value="${value,,}"
  [[ "$value" == *qwen2.5* || "$value" == *qwen-2.5* ]]
}

# Qwen 2.5 ships a Hermes-compatible tool-use chat template. NAT's
# tool_calling_agent sends tool_choice=auto, so vLLM must have both automatic
# tool selection and a parser enabled or every NAT request fails at runtime.
if [[ -z "${tool_call_parser}" ]] && {
  looks_like_qwen25 "$served_model_name" || looks_like_qwen25 "$model_source"
}; then
  tool_call_parser="hermes"
fi

args=(
  serve
  "$model_source"
  --host "$host"
  --port "$port"
  --served-model-name "$served_model_name"
  --trust-remote-code
)

if [[ -n "${tool_call_parser}" ]]; then
  args+=(--enable-auto-tool-choice)
fi

if [[ -n "${tool_call_parser}" ]]; then
  args+=(--tool-call-parser "$tool_call_parser")
fi

if [[ -n "${reasoning_parser}" ]]; then
  args+=(--reasoning-parser "$reasoning_parser")
fi

if [[ -n "$llm_api_key" ]]; then
  args+=(--api-key "$llm_api_key")
fi

if [[ -n "${VLLM_DOWNLOAD_DIR:-}" ]]; then
  args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
fi

if [[ -n "${max_model_len}" ]]; then
  args+=(--max-model-len "$max_model_len")
fi

if [[ -n "${gpu_memory_utilization}" ]]; then
  args+=(--gpu-memory-utilization "$gpu_memory_utilization")
fi

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="${VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE:-0}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
unset VLLM_SERVER_IP
echo "Starting text vLLM on port ${port} with max_model_len=${max_model_len} and gpu_memory_utilization=${gpu_memory_utilization}."
echo "Initial model load can take several minutes; the port will not answer until loading completes."
exec "$vllm_bin" "${args[@]}"
