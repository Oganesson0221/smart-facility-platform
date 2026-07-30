# Smart Facility Platform

Local-first fire-exit monitoring with pretrained Ultralytics YOLO, polygon-based
clearance rules, optional local vision confirmation, NVIDIA NeMo SOP grounding,
Telegram alerts, and a browser dashboard.

The app still supports:

- `demo` detector mode with no model download
- `grounding_dino` for the existing open-vocabulary path
- polygon-free image reasoning through the local vision-language model

## Fire-exit setup

1. Start the application.
2. Upload a reference image or video.
3. Draw the fire-exit clearance polygon.
4. Save the camera configuration.
5. Select the YOLO detector.
6. Analyse the image or video.

## Why a polygon is used

The camera is fixed, so the fire-exit door and required clearance area are also
fixed. The polygon is configured once and reused as the operational clearance
zone. It represents the space that must remain clear, which can be larger than
the visible door itself.

## Token-efficient design

- YOLO processes sampled frames, not every pixel through the vision-language model.
- Deterministic geometry checks whether a detected object intrudes into the saved
  fire-exit clearance zone and how much of the zone it blocks.
- Tracking confirms persistence before an incident is created.
- Gemma 3 is called only after YOLO, overlap, and persistence checks pass.
- Only a cropped evidence region is sent for vision validation.
- NeMo is invoked only after the probable fire-exit incident is confirmed.

## Runtime flow

Image or video upload
→ saved fire-exit clearance polygon
→ YOLO object detection
→ relevant class filtering
→ overlap metrics
→ tracking and persistence for video
→ optional cropped vision validation
→ NeMo SOP retrieval and grounded recommendation
→ incident creation
→ dashboard and Telegram notification

## Install

Host prerequisites:

```bash
python3 --version
```

- Main app: Python 3.10 or 3.11 recommended
- NeMo workflow: Python 3.11 required
- Optional local vision path: `ollama` installed on the host

Application environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 is not supported for the NeMo workflow in this repo. If you create
`.venv` with Python 3.12, `pip install -r requirements-nemo.txt` can fail during
build isolation with `ModuleNotFoundError: No module named 'distutils'`.

Optional Grounding DINO extras:

```bash
pip install -r requirements-nvidia.txt
```

NeMo workflow environment:

```bash
python3.11 --version
uv --version
uv venv --python 3.11 .nemo-venv
uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
```

If `python3.11` or `uv` is missing, install those first on the host. The GB10
machine shown in the error log is currently missing both.

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

FIRE_EXIT_OBSTRUCTION_CLASSES=car,truck,bus,motorcycle,bicycle
INCLUDE_PERSON_AS_OBSTRUCTION=false
PERSON_MINIMUM_DURATION_SECONDS=15

MINIMUM_OBJECT_INTRUSION_RATIO=0.25
MINIMUM_EXIT_BLOCKAGE_RATIO=0.05
MINIMUM_DURATION_SECONDS=5

VALIDATE_FIRE_EXIT_INCIDENTS_WITH_VISION=true
VISION_VALIDATION_FAIL_CLOSED=false
```

## Run with pretrained YOLO

Start Ollama:

```bash
ollama serve
ollama pull gemma3:27b
ollama pull qwen3:32b
```

If `ollama` is not installed, those commands will fail with `Command 'ollama'
not found`. Install it on the host before enabling local vision validation.

`VISION_MODEL` must point to a vision-capable Ollama model such as
`gemma3:27b` or `qwen3-vl:30b`. `qwen3:32b` is text-only and should remain under
`LLM_MODEL`, not `VISION_MODEL`.

Start the full stack:

```bash
./scripts/run-stack.sh
```

Open <http://127.0.0.1:8000>.

If you need to run the processes separately, start the NeMo workflow first:

```bash
./scripts/run-nemo-agent.sh
```

Start the app:

```bash
./scripts/run.sh
```

Open <http://127.0.0.1:8000>.

## Detector modes

- `DETECTOR_PROVIDER=yolo`: pretrained Ultralytics YOLO first-stage detection
- `DETECTOR_PROVIDER=grounding_dino`: existing Grounding DINO provider
- `DETECTOR_PROVIDER=demo`: zero-download deterministic test provider

If Ultralytics is unavailable and `DETECTOR_PROVIDER=demo`, the app still starts.
If `DETECTOR_PROVIDER=yolo` is selected without the dependency, the app raises a
clear runtime error instead of failing with an obscure import stack trace.

## Image analysis behavior

- Image with polygon: YOLO + geometry rules create a fire-exit obstruction
  incident when a relevant object blocks the configured zone.
- Image without polygon: the app keeps the existing full-scene vision reasoning
  path and does not require the fire-exit workflow.

## Video analysis behavior

- Frames are sampled with `VIDEO_SAMPLE_FPS`.
- YOLO runs on sampled frames only.
- The existing tracker is reused.
- The same track must remain inside the fire-exit zone for the configured
  duration before an incident is created.
- Each qualifying track emits at most one incident per analysis job.

Job messages progress through:

- `Loading video`
- `Detecting objects`
- `Tracking obstruction`
- `Validating incident`
- `Retrieving SOP`
- `Sending alert`
- `Completed`

## Local models and NeMo

Polygon-free image reasoning and optional fire-exit confirmation run through the
local vision endpoint configured by `VISION_*`. SOP-grounded summaries and
actions run through the existing NVIDIA NeMo Agent Toolkit workflow and the
local `smart_facility_sop` tool.

The fire-exit SOP is stored in:

```text
sops/fire_exit_obstruction.md
```

The workflow remains advisory and human-in-the-loop.

## Tests

Run the relevant tests with:

```bash
python3 -m unittest tests.test_core tests.test_telegram
```

These tests mock YOLO, Gemma, Telegram, and the NeMo path. They do not download
weights, require a GPU, or call live services.

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
