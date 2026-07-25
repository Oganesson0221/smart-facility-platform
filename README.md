# Smart Facility Platform

Local-first fire-exit obstruction monitoring with NVIDIA Grounding DINO support,
incident tracking, SOP-grounded summaries, Telegram alerts, and a browser dashboard.

The default `demo` detector has no model download and makes the complete workflow
testable on any machine. Set `DETECTOR_PROVIDER=grounding_dino` on the GPU host to
use NVIDIA/IDEA Grounding DINO through Hugging Face Transformers.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run.sh
```

Open <http://127.0.0.1:8000>. The first start creates the local SQLite store and a
sample camera. Draw an exit polygon on an uploaded image, then select **Analyse
image**. For an immediate end-to-end test, select **Use synthetic demo frame**.

For uploaded images without an exit polygon, Smart Facility Platform uses a local
vision-language model to assess visible safety, access, signage, and parking
violations. A positive assessment creates an incident, adds a visible violation
banner to the image, runs the SOP workflow, and sends the annotated image to
Telegram. With Ollama installed, pull a vision-capable model:

```bash
ollama pull qwen3-vl:30b
```

Draw an exit polygon when you want deterministic protected-exit overlap rules.

## NVIDIA Grounding DINO

Install the GPU/model extras in the same environment:

```bash
pip install -r requirements-nvidia.txt
```

Then configure `.env`:

```dotenv
DETECTOR_PROVIDER=grounding_dino
GROUNDING_DINO_MODEL=IDEA-Research/grounding-dino-tiny
DEVICE=cuda
DETECTION_PROMPT=vehicle . car . truck . motorcycle . trolley . pallet . cardboard box . large object . person .
```

The model is loaded lazily on the first analysis. Its weights are downloaded from
Hugging Face the first time, so pre-download them if the production host is
offline. This prototype samples uploaded video at a configurable rate. The
detector interface is intentionally isolated so it can later be replaced with a
TAO-exported TensorRT engine or DeepStream pipeline.

## Local models and NVIDIA NeMo Agent Toolkit

Image understanding runs through local Ollama. Agentic SOP selection and response
generation run through an NVIDIA NeMo Agent Toolkit 1.8 `tool_calling_agent`,
backed by any local OpenAI-compatible model. The tested local setup is:

```dotenv
VISION_ENABLED=true
VISION_BASE_URL=http://127.0.0.1:11434
VISION_MODEL=qwen3-vl:30b

LLM_ENABLED=true
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3:32b
LLM_API_KEY=

NEMO_AGENT_ENABLED=true
NEMO_AGENT_BASE_URL=http://127.0.0.1:8010/v1
NEMO_AGENT_MODEL=smart-facility-agent
NEMO_AGENT_REQUIRED=true
```

Start Ollama, the NeMo workflow, and the app in separate terminals:

```bash
ollama serve
ollama pull qwen3-vl:30b
ollama pull qwen3:32b

uv venv --python 3.11 .nemo-venv
uv pip install --python .nemo-venv/bin/python -r requirements-nemo.txt
./scripts/run-nemo-agent.sh

uv venv --python 3.11 .app-venv
uv pip install --python .app-venv/bin/python -r requirements.txt
./scripts/run.sh
```

No cloud model API key is required. The NeMo plugin in `nemo_agent/` registers a
local `smart_facility_sop` tool. The workflow must call this tool before it
recommends an incident action. If `NEMO_AGENT_REQUIRED=true`, an unavailable
workflow fails closed instead of silently bypassing NeMo.

You can also point `LLM_BASE_URL` at local vLLM or NVIDIA NIM. Keep the NeMo
runtime in its separate Python 3.11+ environment to avoid dependency conflicts.

## Local AI Jupyter lab

[`smart_facility_local_ai_lab.ipynb`](smart_facility_local_ai_lab.ipynb) is an
independent, step-by-step version of the local vision and agent workflow. It does
not use Telegram. Put source images in `lab_images/`, set `IMAGE_NAME` in the
notebook, and run the cells in order. Annotated images and JSON reports are
written to `lab_outputs/`.

Install and launch the lab with:

```bash
source .app-venv/bin/activate
python -m pip install -r requirements-lab.txt
jupyter lab
```

The NeMo server and Ollama must be running as described above. The notebook
checks both runtimes before inference and records the local model and framework
used for each stage in its final result.

## Telegram setup

1. Create a bot with Telegram `@BotFather`.
2. Put `TELEGRAM_BOT_TOKEN` in `.env` and leave
   `TELEGRAM_POLLING_ENABLED=true` for a local installation.
3. Start the app, then send `/start` to `@SmartFacilityAssistant_bot`. Every chat
   that starts the bot is stored as an active alert subscriber; `/stop` opts out.
4. `USER_ID` or `TELEGRAM_ALERT_CHAT_ID` can provide a direct fallback chat ID.

Polling handles `/start`, `/stop`, acknowledge, and false-alarm callbacks without
a public URL. If you prefer webhooks, set `TELEGRAM_POLLING_ENABLED=false`, expose
the API over HTTPS, and set:

```bash
curl -X POST \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -d "url=${PUBLIC_BASE_URL}/api/telegram/webhook" \
  -d "secret_token=${TELEGRAM_WEBHOOK_SECRET}"
```

The bot sends the annotated image and provides **Acknowledge**, **False alarm**,
and **View incident** buttons. Actions update the same SQLite incident record
used by the dashboard. The host must allow outbound HTTPS to
`api.telegram.org:443`.

## Configuration

All settings are documented in [.env.example](.env.example). Do not commit the
real `.env`; it is ignored by Git. Important settings:

- `DETECTOR_PROVIDER`: `demo` or `grounding_dino`
- `DEVICE`: `cuda`, `cpu`, or `auto`
- `VIDEO_SAMPLE_FPS`: video inference sample rate
- `MINIMUM_DURATION_SECONDS`: persistent obstruction threshold
- `LLM_*`: local vLLM/NIM endpoint
- `NEMO_AGENT_*`: local NeMo Agent Toolkit workflow endpoint and fail-closed mode
- `VISION_*`: local Ollama vision model used for polygon-free image reasoning
- `TELEGRAM_*`: bot polling/webhook and optional fallback destination

## API

- `POST /api/analyse/image` — multipart image, camera, polygon
- `POST /api/analyse/scene` — local vision-language assessment without a polygon
- `POST /api/analyse/video` — enqueue multipart video analysis
- `GET /api/jobs/{id}` — video job progress
- `GET /api/incidents` and `GET /api/incidents/{id}`
- `POST /api/incidents/{id}/acknowledge`
- `POST /api/incidents/{id}/false-alarm`
- `POST /api/incidents/{id}/close`
- `GET/POST /api/cameras`
- `PUT /api/cameras/{id}`
- `POST /api/telegram/test`
- `POST /api/telegram/webhook`
- `GET /api/health`
- WebSocket `/api/events`

Swagger documentation is at <http://127.0.0.1:8000/docs>.

## Production path

The MVP is human-in-the-loop: it only records, recommends, and notifies. Before
production, use PostgreSQL/object storage, authenticated roles, a task queue,
TLS, retention policy, manually validated camera polygons, a TAO-fine-tuned
detector exported to TensorRT, and DeepStream for multi-camera RTSP ingestion.
