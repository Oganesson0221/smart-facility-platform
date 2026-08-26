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
  echo "Recommended fix: ./scripts/setup-local-ai-runtime.sh"
  exit 1
fi

if [[ "${ALLOW_HF_OFFLINE:-false}" != "true" ]]; then
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

host="${VISION_VLLM_HOST:-0.0.0.0}"
port="${VISION_VLLM_PORT:-8002}"
served_model_name="${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
default_local_source="$(pwd)/models/${served_model_name}"
if [[ -n "${VISION_MODEL_SOURCE:-}" ]]; then
  model_source="$VISION_MODEL_SOURCE"
elif [[ -f "$default_local_source/config.json" ]]; then
  model_source="$default_local_source"
else
  model_source="$served_model_name"
fi
api_key="${VISION_API_KEY:-${LLM_API_KEY:-}}"
mm_limit_per_prompt="${VISION_MM_LIMIT_PER_PROMPT:-{\"image\":1}}"
max_model_len="${VISION_MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${VISION_GPU_MEMORY_UTILIZATION:-0.25}"
max_num_seqs="${VISION_MAX_NUM_SEQS:-8}"
max_num_batched_tokens="${VISION_MAX_NUM_BATCHED_TOKENS:-32768}"
reasoning_parser="${VISION_REASONING_PARSER:-nemotron_v3}"
tool_call_parser="${VISION_TOOL_CALL_PARSER:-qwen3_coder}"
kv_cache_dtype="${VISION_KV_CACHE_DTYPE:-fp8}"
moe_backend="${VISION_MOE_BACKEND:-}"

args=(
  serve
  "$model_source"
  --host "$host"
  --port "$port"
  --served-model-name "$served_model_name"
  --trust-remote-code
  --limit-mm-per-prompt "$mm_limit_per_prompt"
)

if [[ -n "${api_key}" ]]; then
  args+=(--api-key "$api_key")
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

if [[ "${served_model_name,,}" == *"nemotron"* || "${model_source,,}" == *"nemotron"* ]]; then
  [[ -n "${moe_backend}" ]] && args+=(--moe-backend "$moe_backend")
  args+=(--enable-prefix-caching)
  [[ -n "${max_num_seqs}" ]] && args+=(--max-num-seqs "$max_num_seqs")
  [[ -n "${max_num_batched_tokens}" ]] && args+=(--max-num-batched-tokens "$max_num_batched_tokens")
  [[ -n "${reasoning_parser}" ]] && args+=(--reasoning-parser "$reasoning_parser")
  args+=(--enable-auto-tool-choice)
  [[ -n "${tool_call_parser}" ]] && args+=(--tool-call-parser "$tool_call_parser")
  [[ -n "${kv_cache_dtype}" ]] && args+=(--kv-cache-dtype "$kv_cache_dtype")
fi

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="${VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE:-0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
unset VLLM_SERVER_IP
echo "Starting vision vLLM on port ${port} with max_model_len=${max_model_len} and gpu_memory_utilization=${gpu_memory_utilization}."
echo "Initial model load can take several minutes; the port will not answer until loading completes."
exec "$vllm_bin" "${args[@]}"
