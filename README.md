# Smart Facility Platform

Local fire-exit monitoring for NVIDIA GB10. YOLO detects possible
obstructions, SAM 2 measures their position inside the clearance zone,
Nemotron reviews the visual evidence, and the dashboard and optional Telegram
bot keep a human operator in control.

> This system is advisory. It records evidence and notifies people; it does not
> perform physical facility actions.

## How it works

```text
Camera, image, or video
        |
        v
YOLO detection -> SAM 2 segmentation -> zone and duration rules
        |                                      |
        | clear                                | candidate
        v                                      v
   no incident                    Nemotron visual validation
                                                   |
                                                   v
                              incident + evidence + local SOP
                                                   |
                                      dashboard + Telegram
```

NeMo Agent Toolkit uses the local Qwen server to orchestrate the detector,
segmenter, validation, scene-assessment, and SOP tools. All models and
application data stay on the GB10.

## Fresh GB10 setup

Use Ubuntu with a working NVIDIA driver, internet access, and at least 80 GB
of free disk space.

### 1. Install prerequisites

```bash
nvidia-smi
sudo apt update
sudo apt install -y git curl build-essential
```

Install `uv` if needed, then open a new shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm the required tools:

```bash
nvidia-smi
uv --version
git --version
curl --version
```

You do not need to install Python 3.11 system-wide. `uv` provisions the
correct Python version inside the project environments.

### 2. Clone and configure

```bash
git clone --depth 1 https://github.com/Oganesson0221/smart-facility-platform.git
cd smart-facility-platform
cp .env.example .env
```

If Hugging Face requires access approval for Nemotron, accept its model terms
using your own account and set `HF_TOKEN` in `.env`. Never commit `.env`.

Telegram is optional. To enable it, set `TELEGRAM_BOT_TOKEN` and either
`TELEGRAM_ALERT_CHAT_ID` or `USER_ID` in `.env`.

### 3. Install everything

```bash
./scripts/setup.sh --download-models
```

This idempotent command:

1. creates `.app-venv` with Python 3.11;
2. installs the minimal SAM 2 runtime vendored from pinned commit
   `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
3. downloads only the SAM 2.1 tiny checkpoint;
4. creates `.nemo-venv` with NeMo Agent Toolkit and vLLM;
5. downloads Qwen, then Nemotron, then YOLO.

The two large Hugging Face models are downloaded sequentially. Each model uses
one file worker by default to avoid bursts of simultaneous disk writes. To use
more download concurrency on faster storage, change
`HF_DOWNLOAD_MAX_WORKERS` in `.env`.

Qwen and Nemotron use roughly 35 GB together and can take a while to download.
Progress is saved in `logs/model-download.log`, and interrupted Hugging Face
downloads can be resumed by running the same setup command again.

To install dependencies first and download models later:

```bash
./scripts/setup.sh
.nemo-venv/bin/hf auth login
./scripts/setup.sh --download-models
```

Expected local assets:

```text
models/Qwen/Qwen2.5-7B-Instruct/
models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/
third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
yolo11n.pt
```

These model files, both environments, `.env`, logs, uploads, and runtime data
are ignored by Git. Everyone who clones the repository creates their own local
copies.

## Start, verify, and stop

Start the full stack:

```bash
./scripts/start-all.sh
```

Startup is intentionally sequential: Qwen loads first, then Nemotron, then
NeMo Agent Toolkit, and finally FastAPI. Each service must become healthy
before the next starts, reducing simultaneous model reads and making failures
easy to identify.

Check every port and its log file:

```bash
./scripts/status-all.sh
```

Show the latest lines from all service logs:

```bash
./scripts/status-all.sh --tail
```

Run the full health check, including model identities and a real NeMo detector
tool call:

```bash
./scripts/check-local-ai.sh
```

Open:

- Dashboard: <http://127.0.0.1:8000>
- API documentation: <http://127.0.0.1:8000/docs>

Stop only processes launched by `start-all.sh`:

```bash
./scripts/stop-all.sh
```

## Services and logs

| Service | Port | Log |
| --- | ---: | --- |
| FastAPI dashboard | `8000` | `logs/app.log` |
| Qwen vLLM | `8001` | `logs/text-vllm.log` |
| Nemotron vLLM | `8002` | `logs/vision-vllm.log` |
| NeMo Agent Toolkit | `8010` | `logs/nemo-agent.log` |

Setup output is saved to `logs/setup.log`. Model-download output is saved to
`logs/model-download.log`.

## Access from another computer

Forward only the dashboard port:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@GB10_HOST
```

Then open <http://127.0.0.1:8000> locally. The browser does not need direct
access to the model or NeMo ports.

## Important configuration

The tested GB10 defaults live in `.env.example`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DETECTOR_PROVIDER` | `yolo` | Local object detector |
| `SAM_MODEL_SIZE` | `tiny` | SAM 2 model variant |
| `CV_SHARED_GPU_SAFE_MODE` | `true` | Keeps auto-selected YOLO and SAM work off CUDA while both vLLM servers are active |
| `HF_DOWNLOAD_MAX_WORKERS` | `1` | Files downloaded concurrently within one model |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.35` | Qwen vLLM memory reservation |
| `VISION_GPU_MEMORY_UTILIZATION` | `0.45` | Nemotron vLLM memory reservation |
| `VISION_ATTENTION_BACKEND` | `TRITON_ATTN` | GB10-compatible Nemotron attention backend |
| `VALIDATE_FIRE_EXIT_INCIDENTS_WITH_VISION` | `true` | Sends obstruction candidates to Nemotron |

Keep `LLM_API_KEY` and `VISION_API_KEY` empty for the default localhost
servers. `HF_TOKEN` is only a model-download credential.

## Troubleshooting

- **SAM is missing:** restore `third_party/sam2` from Git and rerun
  `./scripts/setup.sh`.
- **Model download is denied:** accept the Nemotron terms and set a valid
  `HF_TOKEN` in `.env`.
- **A download stops:** rerun `./scripts/setup.sh --download-models`; completed
  files are reused.
- **A service does not start:** run `./scripts/status-all.sh --tail`.
- **CUDA is out of memory:** lower one of the vLLM memory-utilization values in
  `.env`.
- **Telegram times out:** confirm the GB10 can reach `api.telegram.org`.
  Telegram is not required for dashboard operation.

## Development checks

```bash
.app-venv/bin/python -m unittest discover -s tests -v
.app-venv/bin/python -m compileall -q app tests
bash -n scripts/*.sh
```
