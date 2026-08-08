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
if [[ -z "${uv_bin:-}" ]]; then
  echo "No uv binary was found."
  echo "Install uv or export UV_BIN=/absolute/path/to/uv."
  exit 1
fi

echo "Using uv binary: $uv_bin"

if [[ ! -d ".nemo-venv" ]]; then
  "$uv_bin" venv --python 3.11 .nemo-venv
fi

"$uv_bin" pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
"$uv_bin" pip install --python .nemo-venv/bin/python vllm

if [[ -d "third_party/sam2" ]]; then
  echo "Optional: install SAM 2 into the NeMo environment with:"
  echo "  $uv_bin pip install --python .nemo-venv/bin/python -e ./third_party/sam2"
fi

echo "Local AI runtime bootstrap complete."
echo "Next:"
echo "  1. ./scripts/run-vllm-llm.sh"
echo "  2. ./scripts/run-vllm-vision.sh"
echo "  3. ./scripts/run-nemo-agent.sh"
