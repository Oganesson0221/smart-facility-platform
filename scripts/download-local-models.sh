#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

find_hf_cli() {
  local candidate
  for candidate in \
    "${HF_CLI_BIN:-}" \
    "$(command -v hf 2>/dev/null || true)" \
    ".app-venv/bin/hf" \
    ".venv/bin/hf" \
    ".nemo-venv/bin/hf" \
    "$(command -v huggingface-cli 2>/dev/null || true)" \
    ".app-venv/bin/huggingface-cli" \
    ".venv/bin/huggingface-cli" \
    ".nemo-venv/bin/huggingface-cli"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

hf_cli="$(find_hf_cli || true)"
if [[ -z "${hf_cli:-}" ]]; then
  echo "No Hugging Face CLI binary was found."
  echo "Run ./scripts/setup.sh first, or set HF_CLI_BIN explicitly."
  exit 1
fi

cache_dir="${MODEL_CACHE_DIR:-$(pwd)/models}"
mkdir -p "$cache_dir"

if [[ "${ALLOW_HF_OFFLINE:-false}" != "true" ]]; then
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

llm_model="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
vision_model="${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"

echo "Downloading $llm_model into $cache_dir/$llm_model"
"$hf_cli" download \
  "$llm_model" \
  --local-dir "$cache_dir/$llm_model"

echo "Downloading $vision_model into $cache_dir/$vision_model"
"$hf_cli" download \
  "$vision_model" \
  --local-dir "$cache_dir/$vision_model"
