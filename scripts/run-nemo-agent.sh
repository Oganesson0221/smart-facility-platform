#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

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

export SOP_DIRECTORY="${SOP_DIRECTORY:-$(pwd)/sops}"
export NAT_CONFIG_DIR="${NAT_CONFIG_DIR:-$(pwd)/.nat-config}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
exec .nemo-venv/bin/dotenv -f .env run -- \
  .nemo-venv/bin/nat serve --config_file nemo_agent/config.yml
