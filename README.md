# Smart Facility Platform — NVIDIA GB10 Setup

ExitWatch runs locally with:

- YOLO11n for object detection
- SAM 2 for segmentation
- Qwen 2.5 7B for efficient agent work
- NVIDIA Nemotron 3 Nano Omni for vision and capable reasoning
- NVIDIA NeMo Agent Toolkit for tool orchestration
- NVIDIA NeMo Switchyard Stage Router for model selection

Run every command from the repository root.

## Architecture

```text
camera/image → YOLO → ROI filter → SAM 2 when needed
             → Nemotron validation → advisory incident → human review

FastAPI / NeMo Agent Toolkit
              ↓
Switchyard Stage Router :4000
          ↙             ↘
Qwen :8001          Nemotron :8002
efficient             capable
```

Frames that fail the inexpensive YOLO/ROI checks never reach an LLM. Images are
sent directly to the multimodal Nemotron endpoint, never to text-only Qwen.

## Fresh GB10 setup

### 1. System tools

Start with an NVIDIA GB10 system whose driver works:

```bash
nvidia-smi
sudo apt update
sudo apt install -y git curl build-essential
```

Install `uv` and Rust/Cargo if they are not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
uv --version
cargo --version
```

### 2. Configure ExitWatch

```bash
cp .env.example .env
```

Do not overwrite an existing `.env`. Review these values:

```dotenv
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VISION_MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
SWITCHYARD_ENABLED=true
NEMO_AGENT_ENABLED=true
```

`HF_TOKEN` is only for Hugging Face downloads. `LLM_API_KEY` and
`VISION_API_KEY` are optional local vLLM endpoint keys; do not reuse a Hugging
Face token for them.

Telegram is optional. Leave its variables blank for a local-only setup.

### 3. Install the application

```bash
uv venv --python 3.11 .app-venv
uv pip install --python .app-venv/bin/python -r requirements.txt
```

### 4. Install SAM 2

A fresh checkout contains the pinned SAM 2 gitlink but no submodule URL. Populate
it once:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
uv pip install --python .app-venv/bin/python -r requirements-sam.txt
uv pip install --python .app-venv/bin/python -e ./third_party/sam2
(
  cd third_party/sam2/checkpoints
  ./download_ckpts.sh
)
```

If `third_party/sam2/.git` already exists, skip the two clone/checkout commands.

### 5. Install NeMo, vLLM, and Switchyard

```bash
./scripts/setup-local-ai-runtime.sh
uv pip install --python .nemo-venv/bin/python -e ./third_party/sam2
./scripts/setup-switchyard.sh
switchyard-server --help
```

Switchyard is the official native server pinned by
`scripts/setup-switchyard.sh`. Validate its Stage Router configuration:

```bash
source ./scripts/load-dotenv.sh
load_dotenv_file .env
switchyard-server --config config/switchyard/routes.toml --dry-run
```

Expected output includes:

```text
server OK: switchyard/exitwatch-stage
```

### 6. Download local models

Authenticate with Hugging Face and download the configured Qwen and Nemotron
models:

```bash
.nemo-venv/bin/hf auth login
./scripts/download-local-models.sh
.app-venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Local assets should now include:

```text
models/Qwen/Qwen2.5-7B-Instruct/
models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/
third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
yolo11n.pt
```

## Run everything on the host (recommended)

```bash
./scripts/start-all.sh
```

Startup order is Qwen `:8001`, Nemotron `:8002`, Switchyard `:4000`, NeMo
Agent Toolkit `:8010`, then FastAPI `:8000`.

The GB10 defaults run YOLO/SAM on CPU while both LLMs use the unified GPU. The
verified Nemotron reservation is `VISION_GPU_MEMORY_UTILIZATION=0.25`; adjust it
in `.env` only if the hardware or model sizes change.

Open:

- Dashboard: <http://127.0.0.1:8000>
- Routing diagnostics: <http://127.0.0.1:8000/switchyard>

Verify the live stack:

```bash
./scripts/check-local-ai.sh
```

This checks all five services, both model IDs, routine and critical Switchyard
routing, and a real NeMo/YOLO detector call.

Stop everything:

```bash
./scripts/stop-all.sh
```

## Optional: Dockerize only FastAPI

Docker does not replace the model scripts. It packages FastAPI while the
GPU-heavy Qwen, Nemotron, Switchyard, and NeMo services remain on the GB10 host.
This option requires Docker Engine with the Compose plugin (`docker compose version`).

Start the AI services without the host FastAPI process:

```bash
START_APP=false ./scripts/start-all.sh
docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:8000/api/health
```

Compose uses Linux host networking so the container can reach the host services
at the same `127.0.0.1` URLs from `.env`. The image excludes secrets, model
directories, virtual environments, tests, and runtime data from its build
context. Persistent `data/`, `uploads/`, and `evidence/` directories are
mounted from the host.

Stop this layout with:

```bash
docker compose down
./scripts/stop-all.sh
```

## Tests

```bash
.app-venv/bin/python -m unittest discover -s tests -v
.app-venv/bin/python scripts/test-switchyard-integration.py
bash -n scripts/*.sh
docker compose config --quiet
```

The real-server integration test uses local mock OpenAI targets so it verifies
Switchyard itself without loading either large model.

## Troubleshooting

Inspect managed-service logs:

```bash
tail -n 100 logs/text-vllm.log
tail -n 100 logs/vision-vllm.log
tail -n 100 logs/switchyard.log
tail -n 100 logs/nemo-agent.log
tail -n 100 logs/app.log
```

If Nemotron reports insufficient free memory, confirm the Qwen and Nemotron
reservations in `.env`; the verified GB10 defaults are `0.35` and `0.25`.

The `evidence/` directory is required runtime storage for incident images and
Telegram attachments. Its generated contents can be archived or cleared only
when the corresponding incident history is no longer needed.
