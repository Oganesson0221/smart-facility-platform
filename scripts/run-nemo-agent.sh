#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x ".nemo-venv/bin/python" ]]; then
  echo "NeMo environment missing."
  echo "Expected setup:"
  echo "  1. Install Python 3.11 and uv on the host."
  echo "  2. Run: uv venv --python 3.11 .nemo-venv"
  echo "  3. Run: uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt"
  exit 1
fi

python_minor="$(".nemo-venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_minor" != "3.11" ]]; then
  echo "Unsupported NeMo Python version: ${python_minor}"
  echo "The NeMo workflow in this repo must run from a dedicated Python 3.11 environment."
  exit 1
fi

if [[ ! -x ".nemo-venv/bin/nat" || ! -x ".nemo-venv/bin/dotenv" ]]; then
  echo "NeMo environment is incomplete."
  echo "Reinstall with: uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt"
  exit 1
fi

export SOP_DIRECTORY="${SOP_DIRECTORY:-$(pwd)/sops}"
export NAT_CONFIG_DIR="${NAT_CONFIG_DIR:-$(pwd)/.nat-config}"
exec .nemo-venv/bin/dotenv -f .env run -- \
  .nemo-venv/bin/nat serve --config_file nemo_agent/config.yml
