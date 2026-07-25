#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x ".nemo-venv/bin/python" ]]; then
  echo "NeMo environment missing. Run: uv venv --python 3.11 .nemo-venv && uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt"
  exit 1
fi

export SOP_DIRECTORY="${SOP_DIRECTORY:-$(pwd)/sops}"
export NAT_CONFIG_DIR="${NAT_CONFIG_DIR:-$(pwd)/.nat-config}"
exec .nemo-venv/bin/dotenv -f .env run -- \
  .nemo-venv/bin/nat serve --config_file nemo_agent/config.yml
