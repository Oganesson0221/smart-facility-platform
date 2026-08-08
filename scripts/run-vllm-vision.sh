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

export HF_TOKEN="${HF_TOKEN:-${VISION_API_KEY:-${LLM_API_KEY:-}}}"

if [[ "${ALLOW_HF_OFFLINE:-false}" != "true" ]]; then
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

host="${VISION_VLLM_HOST:-0.0.0.0}"
port="${VISION_VLLM_PORT:-8002}"
served_model_name="${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
model_source="${VISION_MODEL_SOURCE:-$served_model_name}"
api_key="${VISION_API_KEY:-${LLM_API_KEY:-}}"
mm_limit_per_prompt="${VISION_MM_LIMIT_PER_PROMPT:-{\"image\":1}}"
max_model_len="${VISION_MAX_MODEL_LEN:-8192}"
gpu_memory_utilization="${VISION_GPU_MEMORY_UTILIZATION:-0.45}"

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
  args+=(--moe-backend triton)
fi

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="${VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE:-0}"
export VLLM_HAS_FLASHINFER_CUBIN="${VLLM_HAS_FLASHINFER_CUBIN:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
unset VLLM_SERVER_IP
echo "Starting vision vLLM on port ${port} with max_model_len=${max_model_len} and gpu_memory_utilization=${gpu_memory_utilization}."
echo "Initial model load can take several minutes; the port will not answer until loading completes."
exec "$vllm_bin" "${args[@]}"
