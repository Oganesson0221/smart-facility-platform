# Smart Facility Platform

Local fire-exit monitoring for NVIDIA GB10: YOLO and SAM identify an
obstruction, a deterministic IoU gate avoids unnecessary model calls, Nemotron
reviews ambiguous images, and Telegram keeps a human operator in control.

## How routing works

```text
Camera/image → YOLO → SAM → mask/zone IoU
                              ├─ ≥ 70% → local SOP → Telegram (zero LLM tokens)
                              └─ < 70% → Nemotron vision → confirmed incident

General text → Switchyard Stage Router → Qwen (routine) or Nemotron (capable)
Telegram question → local incident/SOP retrieval → Qwen → operator reply
```

The system is advisory. It records evidence and notifies people; it does not
perform physical facility actions.

## Fresh GB10 setup

### 1. Prerequisites

Use Ubuntu on an NVIDIA GB10 with a working driver, internet access, and at
least 80 GB of free disk space:

```bash
nvidia-smi
sudo apt update
sudo apt install -y git curl build-essential
```

Install [uv](https://docs.astral.sh/uv/) and
[Rust](https://rustup.rs/) if needed, then open a new shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
uv --version
cargo --version
```

### 2. Configure

From the repository root:

```bash
cp .env.example .env
```

The defaults match the GB10 deployment. If Nemotron access requires a Hugging
Face token, put it in `HF_TOKEN` in `.env`. Do not reuse that token as a local
vLLM API key.

Telegram is optional. To enable it, set `TELEGRAM_BOT_TOKEN` and either
`TELEGRAM_ALERT_CHAT_ID` or `USER_ID`. If the GB10 cannot reach
`api.telegram.org`, set `TELEGRAM_PROXY_URL` or `TELEGRAM_API_BASE_URL`.

### 3. Install everything and download models

```bash
./scripts/setup.sh --download-models
```

This idempotent command creates `.app-venv` and `.nemo-venv`, installs the
application, NeMo Agent Toolkit, vLLM, pinned SAM 2, pinned Switchyard, the SAM
tiny checkpoint, Qwen, Nemotron, and YOLO weights. Qwen and Nemotron use about
36 GB together and can take a while to download.

To install dependencies without downloading the large models:

```bash
./scripts/setup.sh
```

Then download them later with:

```bash
./scripts/download-local-models.sh
.app-venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Expected local assets:

```text
models/Qwen/Qwen2.5-7B-Instruct/
models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/
third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
yolo11n.pt
```

## Start, verify, and stop

```bash
./scripts/start-all.sh
./scripts/check-local-ai.sh
```

Initial model loading can take several minutes. The scripts start and track:

| Service | Port | Purpose |
|---|---:|---|
| FastAPI dashboard | 8000 | UI and browser-safe APIs |
| Qwen vLLM | 8001 | Efficient text and Telegram RAG |
| Nemotron vLLM | 8002 | Capable text and multimodal validation |
| Switchyard | 4000 | Internal Stage Router |
| NeMo Agent Toolkit | 8010 | Tool orchestration |

Open:

- Dashboard: <http://127.0.0.1:8000>
- Routing architecture and statistics: <http://127.0.0.1:8000/switchyard>

Stop only processes launched by `start-all.sh`:

```bash
./scripts/stop-all.sh
```

### Access from a laptop over SSH

Forward only the dashboard port; its Switchyard API links are proxied through
FastAPI:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@GB10_HOST
```

Then open <http://127.0.0.1:8000>. Port 4000 does not need a separate tunnel.

## Useful configuration

The main defaults live in `.env.example`:

- `LLM_GPU_MEMORY_UTILIZATION=0.35`
- `VISION_GPU_MEMORY_UTILIZATION=0.38`
- `VISION_VALIDATION_IOU_THRESHOLD=0.70`
- `TELEGRAM_QUERY_MODEL=Qwen/Qwen2.5-7B-Instruct`

`HF_TOKEN` is only for downloading gated Hugging Face models. Keep
`LLM_API_KEY` and `VISION_API_KEY` empty for the default localhost servers;
putting a Hugging Face token there can expose it in vLLM startup logs.

Only traffic through Switchyard appears in its token table. Telegram Qwen calls
are visible in Qwen vLLM metrics, and ambiguous image calls are visible in
Nemotron vLLM metrics.

## Logs and troubleshooting

```bash
tail -n 100 logs/text-vllm.log
tail -n 100 logs/vision-vllm.log
tail -n 100 logs/switchyard.log
tail -n 100 logs/nemo-agent.log
tail -n 100 logs/app.log
```

Common failures:

- **Model download denied:** accept the model terms on Hugging Face and set a
  valid `HF_TOKEN`.
- **CUDA memory error:** lower one GPU memory-utilization value in `.env`; keep
  their combined reservation below the memory available after model loading.
- **Telegram timeout:** configure `TELEGRAM_PROXY_URL` or a reachable Bot API
  mirror. The local Qwen answer path can work even while Telegram delivery is
  blocked.
- **Stale PID:** rerun `start-all.sh`; it verifies both PID and process start
  time before trusting a runtime record.

## Tests

Run tests without contacting the live NeMo detector from mocked CV fixtures:

```bash
NEMO_AGENT_ORCHESTRATE_CV=false \
  .app-venv/bin/python -m unittest discover -s tests -v
.app-venv/bin/python scripts/test-switchyard-integration.py
bash -n scripts/*.sh
```

The Switchyard integration test starts the real pinned server against local mock
OpenAI targets; it does not load the large models.

## Optional FastAPI container

The recommended deployment runs everything on the host. To containerize only
FastAPI while keeping Qwen, Nemotron, Switchyard, and NeMo on the GB10 host:

```bash
START_APP=false ./scripts/start-all.sh
docker compose up --build -d
curl -fsS http://127.0.0.1:8000/api/health
```

Stop that layout with:

```bash
docker compose down
./scripts/stop-all.sh
```
