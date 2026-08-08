# Smart Facility Platform

Local-first fire-exit monitoring with pretrained Ultralytics YOLO, optional
Segment Anything refinement, polygon-based clearance rules, local multimodal
validation, NVIDIA NeMo Agent Toolkit orchestration, Telegram alerts, and a
browser dashboard.

The app still supports:

- `demo` detector mode with no model download
- `grounding_dino` for the existing open-vocabulary path
- polygon-free image reasoning through the local vision-language model

## Framework Overview

This repo now runs as a layered NeMo-orchestrated system:

1. `FastAPI app` on port `8000`
   - handles uploads, camera config, incidents, dashboard APIs, and Telegram hooks
   - keeps deterministic geometry, tracking, persistence, incident storage, and evidence generation local

2. `NeMo Agent Toolkit` on port `8010`
   - is the orchestration layer for AI calls
   - can call local tools for:
     - YOLO detection
     - SAM segmentation
     - scene reasoning
     - fire-exit validation
     - SOP retrieval

3. `vLLM text server` on port `8001`
   - serves the control LLM for the NeMo tool-calling workflow
   - default model: `google/gemma-4-12B-it`

4. `vLLM vision server` on port `8002`
   - serves the multimodal validation and scene model
   - default model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

5. `Local CV runtime`
   - Ultralytics YOLO remains the first-stage detector
   - SAM 2 remains the segmentation/refinement layer
   - both are exposed as NeMo tools, with local fallback if the NeMo runtime is unavailable

6. `SQLite + evidence storage`
   - incidents are stored in `data/`
   - uploads and generated evidence are stored locally in `uploads/` and `evidence/`

## Architecture At A Glance

```text
Browser UI
  -> FastAPI app : uploads, incidents, camera config, health, dashboard
     -> NeMo Agent Toolkit : tool-calling orchestration on port 8010
        -> vLLM text server : google/gemma-4-12B-it on port 8001
        -> smart facility tools
           -> YOLO detector : yolo11n.pt
           -> SAM segmenter : sam2.1_hiera_tiny when enabled
           -> multimodal vision endpoint : nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 on port 8002
           -> SOP retrieval : local Markdown files under sops/
     -> SQLite, uploads/, evidence/
```

## Default Model Map

| Layer | Default runtime name | Role |
| --- | --- | --- |
| NeMo workflow name | `smart-facility-agent` | OpenAI-compatible workflow endpoint exposed by NeMo |
| Text control model | `google/gemma-4-12B-it` | Tool-calling control LLM used by NeMo |
| Vision model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | Multimodal scene and fire-exit validation |
| Detector | `yolo11n.pt` | First-stage object detection |
| Segmenter | `sam2.1_hiera_tiny` | Mask refinement for relevant detections when SAM is enabled |
| SOP source | `sops/*.md` | Grounded operating procedures and response guidance |

`smart-facility-agent` is the NeMo workflow identifier, not a separate model
checkpoint. The actual control model behind it is configured in
`nemo_agent/config.yml` as `llms.local_facility_llm`.

## Updated fire-exit workflow

Image or sampled video frame
→ YOLO detects an object
→ relevant class filtering
→ fire-exit zone pre-check
→ YOLO box prompts SAM
→ SAM generates an object mask
→ mask is compared with the fire-exit polygon
→ persistence is checked for video
→ the local multimodal model optionally validates
→ NeMo orchestrates SOP retrieval and response steps
→ incident is stored and sent to the dashboard and Telegram

When `SAM_ENABLED=false`, the existing YOLO bounding-box overlap path remains
active and backward compatible.

## Why YOLO and SAM are both used

- YOLO identifies and localises candidate objects quickly.
- SAM refines the YOLO box into an object mask for better overlap measurements.
- Geometry stays deterministic and auditable.
- The local multimodal model is reserved for contextual validation after deterministic checks pass.
- NeMo now orchestrates scene reasoning, fire-exit validation, and SOP-aware response generation.

## Fallback Behavior

- If the NeMo server is offline, the app falls back to the local SOP summary path and the UI shows the NeMo runtime as offline.
- If SAM is unavailable and `SAM_FAIL_OPEN=true`, the app keeps the YOLO box-overlap path active.
- If the vision model is unavailable and `VISION_VALIDATION_FAIL_CLOSED=false`, the deterministic decision path can still continue.

## Limitations

- Pretrained YOLO may not recognise every obstruction type.
- SAM cannot identify an unknown object without a prompt.
- SAM accuracy depends on the quality of the YOLO box.
- The fire-exit polygon must be configured correctly.
- Thresholds still need calibration against real site footage.
- The system remains advisory and human-in-the-loop.

## Install

Host prerequisites:

```bash
python3 --version
```

- Main app: Python 3.10 or 3.11 recommended
- NeMo workflow: Python 3.11 required
- Local model serving: a vLLM-compatible or other OpenAI-compatible local endpoint

Base application environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional Grounding DINO / shared GPU extras:

```bash
pip install -r requirements-nvidia.txt
```

Optional SAM 2 extras:

```bash
pip install -r requirements-sam.txt
git clone https://github.com/facebookresearch/sam2.git ./third_party/sam2
pip install -e ./third_party/sam2
```

`requirements-sam.txt` intentionally does not download checkpoints. Install the
official SAM 2 package separately because the official project ships configs and
extensions outside the base app repo.

The official SAM 2 documentation currently requires Python `>=3.10`,
`torch>=2.5.1`, and `torchvision>=0.20.1`. If your environment is older than
that, upgrade `torch` and `torchvision` before installing the official SAM 2
package.

Download a SAM 2.1 checkpoint from the official repo when needed:

```bash
cd ./third_party/sam2/checkpoints
./download_ckpts.sh
cd /home/admin/Documents/smar-facility-platform
```

NeMo workflow environment:

```bash
python3.11 --version
uv --version
uv venv --python 3.11 .nemo-venv
uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
uv pip install --python .nemo-venv/bin/python vllm
```

The NeMo environment intentionally has its own dependency set. It should not
reuse the main app pins blindly. Current `nvidia-nat` releases require both
`pydantic>=2.11` and `numpy>=2.3`, so the NeMo environment cannot share the
older app-era pins such as `pydantic==2.7.1` or `numpy==1.23.5`. The
repository now keeps the NeMo-specific pins in `requirements-nemo.txt`
compatible with the current NeMo runtime.

If you want SAM to run inside the NeMo-orchestrated tool path as well, install
the official SAM 2 package into the NeMo environment after cloning `third_party/sam2`:

```bash
uv pip install --python .nemo-venv/bin/python -e ./third_party/sam2
```

## Configure

Copy the example file and adjust values as needed:

```bash
cp .env.example .env
```

Recommended fire-exit settings:

```dotenv
DETECTOR_PROVIDER=yolo
YOLO_MODEL_PATH=yolo11n.pt
YOLO_CONFIDENCE_THRESHOLD=0.35
YOLO_IMAGE_SIZE=640
YOLO_DEVICE=auto

SAM_ENABLED=true
SAM_PROVIDER=sam2
SAM_MODEL_SIZE=tiny
SAM_CHECKPOINT_PATH=/absolute/path/to/sam2.1_hiera_tiny.pt
SAM_DEVICE=auto
SAM_USE_FP16=true
SAM_MIN_YOLO_CONFIDENCE=0.35
SAM_ONLY_FOR_ZONE_CANDIDATES=true
SAM_BOUNDARY_MARGIN_PIXELS=20
SAM_MASK_SIMPLIFICATION_EPSILON=2.0
SAM_PROMPT_BOX_EXPAND_RATIO=0.03
SAM_FAIL_OPEN=true

FIRE_EXIT_OBSTRUCTION_CLASSES=car,truck,bus,motorcycle,bicycle,chair
INCLUDE_PERSON_AS_OBSTRUCTION=false
PERSON_MINIMUM_DURATION_SECONDS=15

MINIMUM_OBJECT_INTRUSION_RATIO=0.25
MINIMUM_EXIT_BLOCKAGE_RATIO=0.05
MINIMUM_DURATION_SECONDS=5

VALIDATE_FIRE_EXIT_INCIDENTS_WITH_VISION=true
VISION_VALIDATION_FAIL_CLOSED=false

LLM_ENABLED=true
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=google/gemma-4-12B-it
LLM_MODEL_SOURCE=/absolute/path/to/models/google/gemma-4-12B-it
LLM_MAX_MODEL_LEN=8192
LLM_GPU_MEMORY_UTILIZATION=0.35
LLM_API_KEY=
LLM_TOOL_CALL_PARSER=gemma4
LLM_REASONING_PARSER=gemma4
GEMMA4_CHAT_TEMPLATE=/absolute/path/to/tool_chat_template_gemma4.jinja

NEMO_AGENT_ENABLED=true
NEMO_AGENT_BASE_URL=http://127.0.0.1:8010/v1
NEMO_AGENT_MODEL=smart-facility-agent
NEMO_AGENT_REQUIRED=true
NEMO_AGENT_ORCHESTRATE_CV=true
NEMO_AGENT_ORCHESTRATE_VISION=true

VISION_ENABLED=true
VISION_BASE_URL=http://127.0.0.1:8002/v1
VISION_MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
VISION_MODEL_SOURCE=/absolute/path/to/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
VISION_MAX_MODEL_LEN=8192
VISION_GPU_MEMORY_UTILIZATION=0.45
VISION_API_KEY=
```

Token notes:

- `VISION_API_KEY` is now accepted as the Hugging Face token fallback by the model download and vLLM startup scripts.
- If you already placed your Hugging Face token in `VISION_API_KEY`, you do not need to move it just to download the models.
- `LLM_API_KEY` is still available if you want to protect the local text vLLM endpoint with an API key.
- `HF_TOKEN` can also be exported explicitly if you prefer not to reuse either application key variable.
- The shell scripts no longer `source` `.env` directly, so values like `APP_NAME=Smart Facility Platform` are safe.
- Keep `LLM_MODEL` and `VISION_MODEL` as the logical served model names that the app and NeMo request.
- Use `LLM_MODEL_SOURCE` and `VISION_MODEL_SOURCE` only when vLLM should load weights from an absolute local directory.
- Keep `LLM_MAX_MODEL_LEN` and `VISION_MAX_MODEL_LEN` modest for local dual-model serving. Extremely large values such as `262144` can reserve tens of GiB of KV cache and prevent the text and vision servers from staying up together.
- Keep `LLM_GPU_MEMORY_UTILIZATION` and `VISION_GPU_MEMORY_UTILIZATION` split conservatively when both servers share one GPU. The repo defaults target local dual-model serving rather than single-model maximum throughput.
- If you switch the text control model away from Gemma 4, set `LLM_TOOL_CALL_PARSER` and `LLM_REASONING_PARSER` to the vLLM parser required by that model.

Set the SAM checkpoint path explicitly:

```bash
export SAM_CHECKPOINT_PATH=/absolute/path/to/sam2.1_hiera_tiny.pt
```

Enable or disable SAM at runtime:

```bash
export SAM_ENABLED=true
export SAM_ENABLED=false
```

## Run

### Quick Start

If you already have `.env` and `VISION_API_KEY` populated, the normal startup order is:

1. Create the Python environments.
2. Install app, NeMo, and optional SAM dependencies.
3. Download the local model weights.
4. Start the text vLLM server.
5. Start the vision vLLM server.
6. Start the NeMo Agent Toolkit server.
7. Start the FastAPI app.
8. Open `http://127.0.0.1:8000`.

### Full Runbook

From the repo root:

1. Create and install the main app environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-nvidia.txt
pip install -r requirements-sam.txt
```

2. Install the official SAM 2 package if you want SAM enabled:

```bash
git clone https://github.com/facebookresearch/sam2.git ./third_party/sam2
pip install -e ./third_party/sam2
```

3. Create the NeMo environment:

```bash
uv venv --python 3.11 .nemo-venv
uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
```

If SAM should also execute through the NeMo tool workflow:

```bash
uv pip install --python .nemo-venv/bin/python -e ./third_party/sam2
```

The repo requirements do not install `vllm` for you automatically. The two
`run-vllm-*.sh` launchers expect a real `vllm` binary in one of these places:

- `command -v vllm`
- `.venv/bin/vllm`
- `.nemo-venv/bin/vllm`
- `$HOME/.local/bin/vllm`

4. Verify `.env` has these minimum values:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
VISION_BASE_URL=http://127.0.0.1:8002/v1
NEMO_AGENT_BASE_URL=http://127.0.0.1:8010/v1
VISION_API_KEY=your_huggingface_token
NEMO_AGENT_ORCHESTRATE_CV=true
NEMO_AGENT_ORCHESTRATE_VISION=true
```

5. Download the local checkpoints:

```bash
./scripts/download-local-models.sh
```

If that command already completed successfully on this machine, skip this step
and start the servers.

6. Start the text vLLM server:

```bash
./scripts/run-vllm-llm.sh
```

7. Start the vision vLLM server in a second terminal:

```bash
./scripts/run-vllm-vision.sh
```

When serving `google/gemma-4-12B-it` through vLLM for NeMo tool orchestration,
set `GEMMA4_CHAT_TEMPLATE` to the Gemma 4 tool-use chat template recommended by
the current vLLM documentation.

8. Start the NeMo Agent Toolkit server in a third terminal:

```bash
./scripts/run-nemo-agent.sh
```

9. Start the FastAPI app in a fourth terminal:

```bash
./scripts/run.sh
```

10. Open the dashboard:

```text
http://127.0.0.1:8000
```

### Optional Combined Start

If the two vLLM servers are already running, you can still use the combined app launcher:

```bash
./scripts/run-stack.sh
```

This starts:

- the NeMo Agent Toolkit server
- the FastAPI app

It does not start the two vLLM model servers for you.

### Recovery for the exact errors you saw

If `uv` is installed under `$HOME/.local/bin` but not on your shell `PATH`, either run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

or use the full path explicitly:

```bash
$HOME/.local/bin/uv --version
```

If you see `No vLLM binary was found`, install the runtime into `.nemo-venv`:

```bash
$HOME/.local/bin/uv venv --python 3.11 .nemo-venv
$HOME/.local/bin/uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
$HOME/.local/bin/uv pip install --python .nemo-venv/bin/python vllm
```

You can also use the bundled bootstrap script:

```bash
./scripts/setup-local-ai-runtime.sh
```

If `./scripts/run-vllm-vision.sh` fails with:

```text
argument --limit-mm-per-prompt: Value image=1 cannot be converted ...
```

use the current launcher from this repo. vLLM `0.26.0` expects JSON for that
flag, so the effective value should be:

```bash
export VISION_MM_LIMIT_PER_PROMPT='{"image":1}'
./scripts/run-vllm-vision.sh
```

If `./scripts/run-nemo-agent.sh` reports that `nat` is missing, the NeMo
dependency install did not complete. Re-run:

```bash
$HOME/.local/bin/uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
```

If the NeMo server starts but logs `ModuleNotFoundError: No module named 'app'`,
use the current launcher from this repo. It now exports the repository root on
`PYTHONPATH` before starting NAT so the installed smart-facility plugin can
import `app.services.nemo_tools`.

After that, rerun:

```bash
./scripts/run-vllm-llm.sh
./scripts/run-vllm-vision.sh
./scripts/run-nemo-agent.sh
./scripts/run.sh
```

### Using model folders outside this repo

You can keep the downloaded weights anywhere on disk. The NeMo workflow does
not require the model folders to stay under `./models`.

To download into a shared model directory instead of this repo:

```bash
MODEL_CACHE_DIR=/srv/local-models ./scripts/download-local-models.sh
```

To serve already-downloaded local folders while keeping stable model names for
the app and NeMo:

```dotenv
LLM_MODEL=google/gemma-4-12B-it
LLM_MODEL_SOURCE=/srv/local-models/google/gemma-4-12B-it

VISION_MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
VISION_MODEL_SOURCE=/srv/local-models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
```

`LLM_MODEL` and `VISION_MODEL` are the served names exposed by the local
OpenAI-compatible endpoints. `LLM_MODEL_SOURCE` and `VISION_MODEL_SOURCE` are
the on-disk folders that vLLM loads.

## Detector modes

- `DETECTOR_PROVIDER=yolo`: pretrained Ultralytics YOLO first-stage detection
- `DETECTOR_PROVIDER=grounding_dino`: existing Grounding DINO provider
- `DETECTOR_PROVIDER=demo`: zero-download deterministic test provider

## Image analysis behavior

- Image with polygon: YOLO runs first, SAM optionally refines relevant
  zone candidates, geometry decides overlap, the multimodal model validates
  only after deterministic checks pass, and the incident goes to NeMo and Telegram.
- Image without polygon: the existing YOLO-grounded full-scene reasoning path
  stays active, but the NeMo workflow now orchestrates the multimodal scene assessment.

## Video analysis behavior

- Frames are sampled with `VIDEO_SAMPLE_FPS`.
- YOLO remains the first-stage detector.
- The existing IoU tracker is reused.
- The same track must remain blocking the zone for the configured duration.
- Each qualifying track emits at most one incident per analysis job.
- SAM runs only on relevant zone candidates, not every detection.

## Test the app

Run the unit tests:

```bash
YOLO_CONFIG_DIR=$(pwd)/.yolo-config .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

These tests mock YOLO, SAM, the multimodal model, Telegram, and the NeMo path.
They do not download checkpoints, require CUDA, call vLLM, call NeMo, or call Telegram.

Test an uploaded image through the fire-exit API:

```bash
curl -X POST http://127.0.0.1:8000/api/analyse/image \
  -F file=@tests/car2.jpg \
  -F camera_id=building-a-exit-south \
  -F 'exit_zone=[[250,100],[850,100],[850,680],[250,680]]'
```

Test an uploaded video through the fire-exit API:

```bash
curl -X POST http://127.0.0.1:8000/api/analyse/video \
  -F file=@tests/Download.mp4 \
  -F camera_id=building-a-exit-south \
  -F 'exit_zone=[[250,100],[850,100],[850,680],[250,680]]'
```

Test an uploaded image without a polygon and preserve the existing scene path:

```bash
curl -X POST http://127.0.0.1:8000/api/analyse/scene \
  -F file=@tests/car3.jpg \
  -F camera_id=building-a-exit-south
```

## Local models and NeMo

Polygon-free image reasoning and optional fire-exit confirmation now default
to the NeMo Agent Toolkit workflow when `NEMO_AGENT_ORCHESTRATE_VISION=true`.
The NeMo workflow can call local tools for YOLO detection, SAM segmentation,
multimodal validation, and SOP retrieval, while the underlying text and vision
models stay behind OpenAI-compatible local endpoints configured by `LLM_*` and
`VISION_*`.

For fire-exit image analysis, the effective path is:

1. FastAPI receives the image.
2. NeMo can orchestrate YOLO detection.
3. NeMo can orchestrate SAM refinement for candidate boxes.
4. The app performs deterministic polygon overlap and persistence checks.
5. NeMo can orchestrate multimodal validation with the local vision endpoint.
6. NeMo retrieves the relevant SOP.
7. The app stores the incident and sends Telegram alerts.

The fire-exit SOP is stored in:

```text
sops/fire_exit_obstruction.md
```

### What counts as "in the NeMo Agent Toolkit workflow"

A raw model directory is not "in NeMo" by itself. In this repo, a model is part
of the NeMo workflow only after all of the following are true:

1. The model is loaded behind the local endpoint or tool that the workflow uses.
2. The workflow can reach that endpoint or tool at runtime.
3. The configured model name matches what the endpoint is actually serving.

Concretely in this repo:

- The control LLM is part of the NeMo workflow through `llms.local_facility_llm`
  in `nemo_agent/config.yml`.
- The multimodal vision model is part of the NeMo workflow through the
  `facility_scene_assessor` and `facility_fire_exit_validator` tools.
- YOLO and SAM participate through the `facility_object_detector` and
  `facility_segmenter` tools. Their raw weights are local assets, not separate
  NeMo LLM entries.

### How to verify the stack

If you did not set API keys on the local endpoints:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
curl http://127.0.0.1:8000/api/health
```

If you protected the endpoints with API keys, include the appropriate
`Authorization: Bearer ...` header.

The `GET /api/health` response should show:

- `llm.reachable: true`
- `llm.model_available: true`
- `vision.reachable: true`
- `vision.model_available: true`
- `nemo_agent.reachable: true`

The relevant switches should also stay enabled:

- `NEMO_AGENT_ENABLED=true`
- `NEMO_AGENT_ORCHESTRATE_CV=true`
- `NEMO_AGENT_ORCHESTRATE_VISION=true`

If `llm.model_available` or `vision.model_available` is `false`, the model may
exist on disk but it is not yet part of the active NeMo-served runtime.

## API

- `POST /api/analyse/image`
- `POST /api/analyse/scene`
- `POST /api/analyse/video`
- `GET /api/jobs/{id}`
- `GET /api/incidents`
- `GET /api/incidents/{id}`
- `POST /api/incidents/{id}/acknowledge`
- `POST /api/incidents/{id}/false-alarm`
- `POST /api/incidents/{id}/close`
- `GET /api/cameras`
- `POST /api/cameras`
- `PUT /api/cameras/{id}`
- `GET /api/health`

Swagger docs are available at <http://127.0.0.1:8000/docs>.
