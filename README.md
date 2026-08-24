# Smart Facility Platform — Full NVIDIA Setup

This repository uses the following local models:

- `Qwen/Qwen2.5-7B-Instruct` for NeMo Agent Toolkit control and tool calling
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` for image reasoning
- `yolo11n.pt` for object detection
- `sam2.1_hiera_tiny.pt` for segmentation

Run all commands below from the repository root.

## 1. Prerequisites

You need:

- Linux with an NVIDIA GPU
- a working NVIDIA driver and `nvidia-smi`
- Python 3.11
- `uv`, `git`, and `curl`
- enough GPU memory, RAM, and disk space for Qwen and Nemotron
- a Hugging Face account with access to the NVIDIA Nemotron model

Check the required tools:

```bash
nvidia-smi
python3.11 --version
uv --version
git --version
```

## 2. Configure the project

Create `.env` once:

```bash
cp .env.example .env
```

If `.env` already exists, do not copy it again. The supplied configuration uses
Qwen, NVIDIA Nemotron, YOLO, SAM 2, NeMo Agent Toolkit, and local vLLM servers.
Do not commit `.env` because it may contain access tokens.

Create and install the application environment:

```bash
uv venv --python 3.11 .app-venv
uv pip install --python .app-venv/bin/python -r requirements.txt
```

## 3. Install SAM 2

If `third_party/sam2` is empty, clone the repository version pinned by this
project:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Install SAM 2 in the application environment:

```bash
uv pip install --python .app-venv/bin/python -r requirements-sam.txt
uv pip install --python .app-venv/bin/python -e ./third_party/sam2
```

Download the SAM checkpoints:

```bash
cd third_party/sam2/checkpoints
./download_ckpts.sh
cd ../../..
```

The configured tiny checkpoint is:

```text
third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

## 4. Install NeMo Agent Toolkit and vLLM

Create the separate local-AI environment:

```bash
./scripts/setup-local-ai-runtime.sh
uv pip install --python .nemo-venv/bin/python -e ./third_party/sam2
```

This installs NeMo Agent Toolkit, the project NeMo plugin, vLLM, and SAM 2 in
`.nemo-venv`. On Python 3.11, the setup script also applies the upstream
FlashInfer deferred-annotation fix required by current vLLM releases.
The launchers leave vLLM compilation mode at its architecture-aware default,
which is required on NVIDIA GB10/aarch64 systems.

## 5. Download all models locally

Authenticate with Hugging Face:

```bash
.nemo-venv/bin/hf auth login
```

Download Qwen and NVIDIA Nemotron:

```bash
./scripts/download-local-models.sh
```

Download the YOLO checkpoint before startup:

```bash
.app-venv/bin/python -c "from ultralytics import YOLO; YOLO(\"yolo11n.pt\")"
```

The downloaded files are stored at:

```text
models/Qwen/Qwen2.5-7B-Instruct/
models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/
yolo11n.pt
third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

The vLLM launchers automatically use the two directories under `models/`.
Model weights are ignored by Git.

## 6. Start the full NVIDIA stack

```bash
./scripts/start-all.sh
```

Startup order:

1. Qwen vLLM on `8001`
2. NVIDIA Nemotron vLLM on `8002`
3. NeMo Agent Toolkit on `8010`
4. FastAPI and the dashboard on `8000`

The first startup can take several minutes. Logs are written to `logs/`.
When startup completes, open <http://127.0.0.1:8000>.

## 7. Verify and stop

Verify the model identities, services, and a real NeMo detector call:

```bash
./scripts/check-local-ai.sh
```

Stop the managed stack:

```bash
./scripts/stop-all.sh
```

If startup fails, inspect the service logs:

```bash
ls logs
tail -n 100 logs/text-vllm.log
tail -n 100 logs/vision-vllm.log
tail -n 100 logs/nemo-agent.log
tail -n 100 logs/app.log
```
