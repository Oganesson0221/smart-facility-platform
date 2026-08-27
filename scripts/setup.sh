#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

download_models=false
for argument in "$@"; do
  case "$argument" in
    --download-models)
      download_models=true
      ;;
    -h|--help)
      echo "Usage: ./scripts/setup.sh [--download-models]"
      echo "  --download-models  Also download Qwen, Nemotron, and YOLO weights."
      exit 0
      ;;
    *)
      echo "Unknown option: $argument"
      echo "Usage: ./scripts/setup.sh [--download-models]"
      exit 2
      ;;
  esac
done

mkdir -p logs
setup_log="$(pwd)/logs/setup.log"
exec > >(tee -a "$setup_log") 2>&1

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
  echo "Install uv from https://docs.astral.sh/uv/ and rerun this command."
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example."
else
  echo "Keeping existing .env."
fi

sam_dir="third_party/sam2"
if [[ ! -f "$sam_dir/setup.py" ]]; then
  echo "The vendored SAM 2 runtime is unavailable at $sam_dir."
  echo "Restore the repository files and rerun setup."
  exit 1
fi

echo "[1/4] Installing application dependencies."
if [[ ! -x .app-venv/bin/python ]]; then
  "$uv_bin" venv --python 3.11 .app-venv
fi
"$uv_bin" pip install --python .app-venv/bin/python -r requirements.txt
"$uv_bin" pip install --python .app-venv/bin/python -r requirements-sam.txt
"$uv_bin" pip install --python .app-venv/bin/python -e ./third_party/sam2

echo "[2/4] Ensuring the SAM 2.1 tiny checkpoint is present."
checkpoint="third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
checkpoint_sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
mkdir -p "$(dirname "$checkpoint")"
if [[ ! -s "$checkpoint" ]]; then
  .app-venv/bin/hf download \
    facebook/sam2.1-hiera-tiny \
    sam2.1_hiera_tiny.pt \
    --local-dir "$(dirname "$checkpoint")" \
    --max-workers 1
else
  echo "SAM checkpoint already present."
fi

actual_checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
if [[ "$actual_checkpoint_sha256" != "$checkpoint_sha256" ]]; then
  echo "SAM checkpoint checksum verification failed."
  echo "Expected: $checkpoint_sha256"
  echo "Actual:   $actual_checkpoint_sha256"
  exit 1
fi
echo "SAM checkpoint checksum verified."

echo "[3/4] Installing NeMo Agent Toolkit and vLLM."
./scripts/setup-local-ai-runtime.sh
"$uv_bin" pip install --python .nemo-venv/bin/python -e ./third_party/sam2

echo "[4/4] Checking installed runtimes."
.app-venv/bin/python -c "import torch, sam2; print(f'application torch={torch.__version__}')"
.nemo-venv/bin/python -c "import torch, vllm; print(f'local-ai torch={torch.__version__} cuda={torch.cuda.is_available()} vllm={vllm.__version__}')"

if [[ "$download_models" == "true" ]]; then
  ./scripts/download-local-models.sh
  .app-venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
fi

echo
echo "Setup complete. Log: $setup_log"
if [[ "$download_models" == "false" ]]; then
  echo "Models were not downloaded. Run ./scripts/setup.sh --download-models when ready."
fi
echo "Start: ./scripts/start-all.sh"
