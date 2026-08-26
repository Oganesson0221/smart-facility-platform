#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

find_uv() {
  local candidate
  for candidate in \
    "${UV_BIN:-}" \
    "$(command -v uv 2>/dev/null || true)" \
    "$HOME/.local/bin/uv"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

uv_bin="$(find_uv || true)"
uv_cmd="${uv_bin:-uv}"

if [[ ! -x ".nemo-venv/bin/python" ]]; then
  echo "NeMo environment missing."
  echo "Expected setup:"
  echo "  1. Install Python 3.11 and uv on the host."
  echo "  2. Run: ${uv_cmd} venv --python 3.11 .nemo-venv"
  echo "  3. Run: ${uv_cmd} pip install --python .nemo-venv/bin/python -r requirements-nemo.txt"
  echo "  4. Run: ${uv_cmd} pip install --python .nemo-venv/bin/python vllm"
  exit 1
fi

python_minor="$(".nemo-venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_minor" != "3.11" ]]; then
  echo "Unsupported NeMo Python version: ${python_minor}"
  echo "The NeMo workflow in this repo must run from a dedicated Python 3.11 environment."
  exit 1
fi

missing_tools=()
[[ -x ".nemo-venv/bin/nat" ]] || missing_tools+=("nat")
[[ -x ".nemo-venv/bin/dotenv" ]] || missing_tools+=("dotenv")

if (( ${#missing_tools[@]} > 0 )); then
  echo "NeMo environment is incomplete."
  echo "Missing CLI tool(s): ${missing_tools[*]}"
  echo "Recommended fix: ./scripts/setup-local-ai-runtime.sh"
  echo "Reinstall with: ${uv_cmd} pip install --python .nemo-venv/bin/python -r requirements-nemo.txt"
  echo "Then install vllm with: ${uv_cmd} pip install --python .nemo-venv/bin/python vllm"
  exit 1
fi

if [[ "${SWITCHYARD_ENABLED:-true}" == "true" ]]; then
  switchyard_base="${SWITCHYARD_BASE_URL:-http://127.0.0.1:4000}"
  switchyard_base="${switchyard_base%/}"
  [[ "$switchyard_base" == */v1 ]] || switchyard_base="${switchyard_base}/v1"
  export NAT_LLM_BASE_URL="$switchyard_base"
  export NAT_LLM_MODEL="${SWITCHYARD_MODEL:-switchyard/exitwatch-stage}"
  export NAT_LLM_API_KEY="${SWITCHYARD_API_KEY:-EMPTY}"
else
  export NAT_LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
  export NAT_LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  export NAT_LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
fi

export SOP_DIRECTORY="${SOP_DIRECTORY:-$(pwd)/sops}"
export NAT_CONFIG_DIR="${NAT_CONFIG_DIR:-$(pwd)/.nat-config}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
exec .nemo-venv/bin/dotenv -f .env run -- \
  .nemo-venv/bin/nat serve \
    --config_file nemo_agent/config.yml \
    --host "${NEMO_AGENT_HOST:-0.0.0.0}" \
    --port "${NEMO_AGENT_PORT:-8010}"
