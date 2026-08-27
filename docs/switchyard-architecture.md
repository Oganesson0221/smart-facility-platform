# Switchyard architecture in Smart Facility Platform

## Request flow

```mermaid
flowchart LR
    User[Dashboard / API client] --> App[FastAPI app :8000]
    App --> NeMo[NeMo Agent Toolkit :8010]

    NeMo -->|LLM orchestration request| SY[Switchyard :4000]
    SY --> Stage{Stage Router<br/>efficient_first<br/>threshold 0.5}
    Stage -->|fall_open or tests_passed| Qwen[Qwen 2.5 7B :8001]
    Stage -->|override / critical signals| NemotronText[Nemotron Omni 30B :8002]
    Qwen --> SY
    NemotronText --> SY
    SY -->|OpenAI response plus<br/>x-model-router-selected-model| NeMo

    NeMo -->|tool call| Tools[YOLO + SAM]
    Tools --> IoU{SAM mask / exit-zone IoU}
    IoU -->|at least 70%| Alert[Incident + deterministic SOP + Telegram]
    IoU -->|below 70% or unavailable| NemotronVision[Nemotron Omni 30B :8002]
    NemotronVision -->|confirmed| Alert
    NemotronVision -->|rejected| NoAlert[No incident / no alert]

    SY --> Stats[/v1/stats and /metrics]
    SY --> Log[switchyard-routing.jsonl]
```

Only requests that enter Switchyard on port 4000 appear in Switchyard token and
request statistics. Ordinary grounded SOP/RAG answers use this path: the Stage
Router normally selects Qwen for routine generation and can override to Nemotron
when conversation or tool-result signals warrant the capable tier.

Telegram operator free-text questions are intentionally more constrained. The app
retrieves incident and SOP context locally, then sends the grounded prompt directly
to `TELEGRAM_QUERY_MODEL` (Qwen by default). This prevents words inside incident
records from unnecessarily escalating a normal operator question to Nemotron. These
query tokens are accounted by the Qwen vLLM server rather than Switchyard.

The vision gate is application-level deterministic routing, separate from Switchyard.
When SAM mask/zone IoU is at least `VISION_VALIDATION_IOU_THRESHOLD` (default
`0.70`), the incident uses deterministic local SOP text and proceeds to Telegram
without any Qwen or Nemotron request. Lower IoU or a missing SAM IoU escalates
directly to multimodal Nemotron on port 8002. Those vision tokens are visible
through vLLM rather than Switchyard statistics.

## OpenAI-compatible routed call

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "switchyard/exitwatch-stage",
    "messages": [{"role": "user", "content": "Summarize the incident."}],
    "max_tokens": 128
  }'
```

The client requests the route ID, not a backend model. Switchyard examines the
conversation and tool-result signals, selects a target, forwards the translated request,
and returns a normal OpenAI-compatible response. The response header
`x-model-router-selected-model` identifies the backend that served it.

## Live observability

- Dashboard: `http://127.0.0.1:8000/switchyard`
- Model and token statistics through the app: `/api/switchyard/stats`
- Prometheus metrics through the app: `/api/switchyard/metrics`
- Configured route IDs through the app: `/api/switchyard/models`
- Per-request durable records: `logs/switchyard-routing.jsonl`

The `/api/switchyard/*` endpoints proxy the internal port 4000 service through the
FastAPI app. They therefore work through the same SSH port forwarding used for port
8000; a laptop browser does not need separate access to port 4000.

`/v1/stats` is cumulative until the server restarts or
`POST /v1/stats/reset` is called.
