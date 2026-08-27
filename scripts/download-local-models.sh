#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

mkdir -p logs
download_log="$(pwd)/logs/model-download.log"
exec > >(tee -a "$download_log") 2>&1

find_hf_cli() {
  local candidate
  for candidate in \
    "${HF_CLI_BIN:-}" \
    "$(command -v hf 2>/dev/null || true)" \
    ".venv/bin/hf" \
    ".nemo-venv/bin/hf" \
    "$(command -v huggingface-cli 2>/dev/null || true)" \
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
  echo "Install huggingface_hub in .venv or set HF_CLI_BIN explicitly."
  exit 1
fi

cache_dir="${MODEL_CACHE_DIR:-$(pwd)/models}"
mkdir -p "$cache_dir"

export HF_TOKEN="${HF_TOKEN:-${VISION_API_KEY:-${LLM_API_KEY:-}}}"

if [[ "${ALLOW_HF_OFFLINE:-false}" != "true" ]]; then
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

llm_model="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
vision_model="${VISION_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
max_workers="${HF_DOWNLOAD_MAX_WORKERS:-1}"

if [[ ! "$max_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "HF_DOWNLOAD_MAX_WORKERS must be a positive integer; got: $max_workers"
  exit 1
fi

echo "Model downloads are sequential with $max_workers file worker(s) per model."
echo "[1/2] Downloading $llm_model into $cache_dir/$llm_model"
"$hf_cli" download \
  "$llm_model" \
  --local-dir "$cache_dir/$llm_model" \
  --max-workers "$max_workers"

echo "[2/2] Downloading $vision_model into $cache_dir/$vision_model"
"$hf_cli" download \
  "$vision_model" \
  --local-dir "$cache_dir/$vision_model" \
  --max-workers "$max_workers"

echo "Model downloads complete. Log: $download_log"
