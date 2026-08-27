#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

download_models=false
sam_revision="${SAM2_GIT_REV:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
sam_url="${SAM2_GIT_URL:-https://github.com/facebookresearch/sam2.git}"
sam_checkpoint_url="${SAM2_TINY_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt}"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--download-models]

Installs the application, NeMo/vLLM runtime, SAM 2, and Switchyard.
Use --download-models to also fetch Qwen and Nemotron (about 36 GB total).
EOF
}

for argument in "$@"; do
  case "$argument" in
    --download-models) download_models=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    echo "$2" >&2
    exit 1
  }
}

validate_env_file() {
  local line line_number=0 failures=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    line="${line%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ ! "$line" =~ ^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*= ]]; then
      echo ".env:${line_number}: expected NAME=value, found invalid text" >&2
      failures=$((failures + 1))
    fi
  done < .env
  if (( failures > 0 )); then
    echo "Remove the invalid lines from .env and rerun setup." >&2
    exit 1
  fi
}

find_uv() {
  local candidate
  for candidate in "${UV_BIN:-}" "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv"; do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

require_command git "Install it with: sudo apt install -y git"
require_command curl "Install it with: sudo apt install -y curl"
require_command cargo "Install Rust from https://rustup.rs, reopen the shell, and retry."
uv_bin="$(find_uv || true)"
if [[ -z "$uv_bin" ]]; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
else
  echo "Keeping existing .env"
fi
validate_env_file
# Switchyard's config references API-key environment variables. They may be
# empty for local endpoints, but must exist while the config is validated.
# shellcheck disable=SC1091
source ./scripts/load-dotenv.sh
load_dotenv_file .env

mkdir -p data uploads evidence logs models third_party

echo "[1/6] Installing the FastAPI application environment"
[[ -x .app-venv/bin/python ]] || "$uv_bin" venv --python 3.11 .app-venv
"$uv_bin" pip install --python .app-venv/bin/python -r requirements.txt

echo "[2/6] Preparing pinned SAM 2 source"
if [[ ! -f third_party/sam2/pyproject.toml ]]; then
  if [[ -d third_party/sam2 ]] && [[ -n "$(find third_party/sam2 -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "third_party/sam2 exists but is not a usable SAM 2 checkout; move it aside and retry." >&2
    exit 1
  fi
  git clone "$sam_url" third_party/sam2
fi
current_sam_revision="$(git -C third_party/sam2 rev-parse HEAD 2>/dev/null || true)"
if [[ "$current_sam_revision" != "$sam_revision" ]]; then
  if [[ -n "$(git -C third_party/sam2 status --porcelain 2>/dev/null)" ]]; then
    echo "third_party/sam2 has local changes; refusing to change its revision." >&2
    exit 1
  fi
  git -C third_party/sam2 fetch origin "$sam_revision"
  git -C third_party/sam2 checkout --detach "$sam_revision"
fi
"$uv_bin" pip install --python .app-venv/bin/python -r requirements-sam.txt
"$uv_bin" pip install --python .app-venv/bin/python --no-build-isolation --no-deps -e ./third_party/sam2

sam_checkpoint="third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
if [[ ! -s "$sam_checkpoint" ]]; then
  echo "Downloading the SAM 2.1 tiny checkpoint"
  mkdir -p "$(dirname "$sam_checkpoint")"
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
    --output "$sam_checkpoint" "$sam_checkpoint_url"
fi

echo "[3/6] Installing NeMo Agent Toolkit and vLLM"
UV_BIN="$uv_bin" ./scripts/setup-local-ai-runtime.sh
"$uv_bin" pip install --python .nemo-venv/bin/python --no-build-isolation --no-deps -e ./third_party/sam2

echo "[4/6] Installing the pinned Switchyard server"
./scripts/setup-switchyard.sh
switchyard_bin="${SWITCHYARD_SERVER_BIN:-$(command -v switchyard-server 2>/dev/null || true)}"
if [[ -z "$switchyard_bin" && -x "$HOME/.cargo/bin/switchyard-server" ]]; then
  switchyard_bin="$HOME/.cargo/bin/switchyard-server"
fi
if [[ -z "$switchyard_bin" ]]; then
  echo "switchyard-server was installed but cannot be found." >&2
  exit 1
fi

echo "[5/6] Validating local configuration"
.app-venv/bin/python -c 'import cv2, numpy, sam2; from app.config import settings; assert int(numpy.__version__.split(".")[0]) < 2; print(f"App/CV imports OK: {settings.app_name}, NumPy {numpy.__version__}, OpenCV {cv2.__version__}")'
.nemo-venv/bin/python -c 'import vllm; import nat; print("NeMo/vLLM imports OK")'
"$switchyard_bin" \
  --config "${SWITCHYARD_CONFIG:-config/switchyard/routes.toml}" --dry-run

echo "[6/6] Model assets"
if [[ "$download_models" == "true" ]]; then
  ./scripts/download-local-models.sh
  .app-venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('YOLO weights ready')"
else
  echo "Skipped Qwen, Nemotron, and YOLO downloads. Run:"
  echo "  ./scripts/download-local-models.sh"
  echo "  .app-venv/bin/python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
fi

echo
echo "Setup complete. Next:"
echo "  ./scripts/start-all.sh"
echo "  ./scripts/check-local-ai.sh"
