# Smart Facility Platform: Slide-Ready System Overview

This document is meant to help turn the project into presentation slides.
It explains what the system does, how the pipeline works, which local models
are involved, where SOP guidance lives, and what the current limitations are.

## 1. What This Project Is

Smart Facility Platform is a local-first safety monitoring system for fire exits
and restricted access areas.

At a high level, it does four things:

1. Detects visible objects in uploaded images or videos.
2. Decides whether those objects represent a safety issue.
3. Creates an incident with evidence, summary, and recommended action.
4. Sends alerts to Telegram and allows follow-up SOP questions through the bot.

The current implementation is designed around fire-exit obstruction scenarios,
especially vehicles and other large objects that block access.

## 2. Core Idea

The project does not send every image straight to a large model.

Instead, it uses a staged pipeline:

1. YOLO runs first as the fast object detector.
2. Geometry rules or scene grounding reduce false positives.
3. A local vision model reasons about the grounded evidence.
4. SOP retrieval and LLM guidance produce the operator-facing response.

This keeps the system more explainable, more efficient, and easier to debug
than a pure end-to-end multimodal model.

## 3. Main Operating Modes

The system currently has two main analysis modes.

### A. Polygon-Based Fire-Exit Monitoring

This is the structured safety-monitoring mode.

Use case:
- Fixed camera
- Known fire-exit location
- Known clearance zone

Flow:
1. A user uploads an image or video.
2. A polygon is drawn around the fire-exit clearance zone.
3. YOLO detects objects.
4. The system checks whether a detected object overlaps the protected zone.
5. For video, tracking and persistence are used before creating an incident.
6. The incident is optionally validated by the local vision model.
7. The SOP/LLM layer produces the final summary and recommended action.
8. The incident is shown in the UI and can be sent to Telegram.

### B. Scene Reasoning Mode

This is the polygon-free image reasoning mode.

Use case:
- Single uploaded image
- No manual zone drawn
- Need a quick safety assessment

Flow:
1. YOLO runs first on the full image.
2. The UI shows what YOLO detected and the confidence scores.
3. The grounded detections are passed to the local vision model.
4. The vision model decides whether a violation exists.
5. The system creates an incident only if the result matches real YOLO detections.
6. The final summary and action are shown in the UI and can be sent to Telegram.

## 4. End-to-End Runtime Flow

A simplified end-to-end flow looks like this:

```text
Image or video upload
-> YOLO detection
-> Object filtering / grounding
-> Zone logic or scene reasoning
-> Incident creation
-> SOP retrieval
-> Local LLM / NeMo response
-> Annotated evidence
-> Telegram alert
-> Telegram follow-up questions
```

## 5. Main Components

| Component | Purpose | Main file(s) |
| --- | --- | --- |
| FastAPI app | API and dashboard backend | `app/main.py`, `app/api.py` |
| Object detection | YOLO / Grounding DINO / demo detector | `app/services/cv/detector.py` |
| Geometry + tracking | Zone overlap, blockage, persistence | `app/services/cv/roi_engine.py`, `app/services/cv/tracker.py` |
| Scene orchestration | Incident creation and grounding | `app/services/processing.py` |
| Vision reasoning | Local image-based reasoning | `app/services/scene_reasoning.py` |
| Summary generation | SOP-grounded LLM response | `app/services/llm.py`, `app/services/agent.py` |
| SOP retrieval | Finds relevant SOP files | `app/services/sop.py` |
| Telegram alerts + bot | Sends alerts and handles chat | `app/services/telegram.py` |
| Frontend UI | Visual analysis, incident viewer, workflow display | `app/static/app.js`, `app/static/styles.css` |

## 6. Detection Layer

The active first-stage detector in the main flow is Ultralytics YOLO.

Current repo default:
- Model: `yolo11n.pt`
- Detector provider: `yolo`

What YOLO does here:
- Finds objects quickly
- Returns labels, confidence scores, and bounding boxes
- Feeds those detections into the rest of the pipeline

Recent implementation details:
- Scene uploads are YOLO-first
- The UI now shows YOLO detections before vision reasoning
- Class-agnostic NMS and duplicate-box suppression are used to reduce cases
  like one vehicle being labeled as both `car` and `truck`

## 7. Vision Reasoning Layer

After YOLO, the local vision model evaluates whether the detected objects
actually support a violation.

Current repo default:
- Vision model: `gemma3:27b`
- Vision endpoint: local Ollama

The local vision model does not invent boxes in the current scene path.
It must reason from:
- the image
- the grounded YOLO detections
- visible evidence

This makes the result easier to explain:
- YOLO says what exists
- the vision model says what the detected object means in context

## 8. SOP and LLM Layer

Once an incident exists, the system produces a human-readable explanation and
recommended action.

There are two SOP sources in the repo:

1. Incident SOP markdown files in:
   - `sops/fire_exit_obstruction.md`
   - `sops/vehicle_blocking_exit.md`
   - `sops/restricted_parking.md`
   - `sops/escalation_matrix.md`

2. Telegram bot reference text in:
   - `sops/telegram_assistant_reference.txt`

The Telegram assistant reference file is the one to mention in slides if you
want to explain how the bot answers follow-up operator questions.

The LLM layer uses:
- incident fields
- SOP excerpts
- scene detections
- visible evidence

It then generates:
- incident summary
- recommended action

## 9. Telegram Alerting

When a valid incident is created, the system can send:
- the annotated image
- incident summary
- confidence
- incident ID
- incident time
- recommended action

The Telegram bot can also do more than alerting.

It now supports interactive follow-up questions such as:
- "What should I do next for the latest incident?"
- "What should I do next for INC-20260730-042724-593?"
- "What does the SOP say?"

The assistant answers from the local SOP reference file and the available
incident context.

## 10. UI Behavior

The browser UI now makes the pipeline visible.

For scene uploads, the user sees:
1. YOLO detection step
2. Local vision reasoning step
3. Structured result sections

The result card now separates:
- Workflow
- Assessment
- Confidence
- Incident time
- YOLO detections
- Visible evidence
- Recommended first action

This is useful for slides because it demonstrates that the system is not a
black box. The user can see the intermediate steps.

## 11. Data Stored Per Incident

Each incident can store:
- incident ID
- facility
- zone
- event type
- object type
- confidence
- timestamps
- evidence image
- metadata
- SOP title
- recommended action
- Telegram delivery status

For scene incidents, the raw source image is also preserved so the system can
re-ground or re-annotate later.

## 12. Why This Design Is Useful

This architecture is useful because it separates responsibilities:

- YOLO is fast and good at object localization
- geometry logic is deterministic and auditable
- the vision model handles semantic interpretation
- SOP retrieval constrains recommendations
- Telegram closes the loop with the operator

That makes the system:
- cheaper to run than sending everything to a multimodal model
- easier to test
- easier to debug
- more defensible in a safety setting

## 13. Current Strengths

- Local-first runtime
- Structured incident records
- Clear operator-facing UI
- Telegram alert delivery with annotated images
- Telegram assistant for SOP follow-up questions
- Human-in-the-loop response model
- Good support for vehicle-based exit obstruction scenarios

## 14. Current Limitations

Important limitations to mention honestly in slides:

1. The quality of object detection depends on the chosen YOLO model.
2. `yolo11n.pt` does not support every object type.
3. Some classes are available, but others are not.
4. Ladder is not currently a supported YOLO class in `yolo11n.pt`.
5. Chair is now included in the default blocked classes/config, but reliable
   incident behavior still depends on actual YOLO detection quality.
6. The Telegram assistant is grounded in local SOP text, so if the SOP text is
   incomplete, the answer quality is limited by that source.
7. The system is advisory. It recommends human action; it does not autonomously
   dispatch emergency services or move objects.

## 15. Good Slide Narrative

If you want to make slides, a clean storyline is:

### Slide 1: Problem

Fire exits and restricted access areas can be blocked without immediate response.

### Slide 2: Solution

A local-first safety monitoring platform that detects, reasons, logs, alerts,
and guides operators.

### Slide 3: Architecture

Show:

```text
Camera / Upload
-> YOLO
-> Rules / Grounding
-> Vision model
-> SOP / LLM
-> Telegram + UI
```

### Slide 4: Detection Pipeline

Explain why YOLO is first:
- fast
- box-based
- cheap
- interpretable

### Slide 5: Reasoning Layer

Explain that the local vision model interprets the grounded detection instead of
inventing its own structure.

### Slide 6: Incident and SOP Layer

Explain that the system does not stop at detection. It creates an incident,
retrieves the SOP, and recommends the next action.

### Slide 7: Telegram Workflow

Explain:
- annotated alert image
- incident details
- operator can ask follow-up SOP questions

### Slide 8: Demo Example

Use one example:
- upload car blocking fire exit
- YOLO detects car
- local vision confirms obstruction
- incident is created
- Telegram alert is sent
- user asks the bot: "What should I do next?"

### Slide 9: Limitations

Be explicit:
- not every object class is supported
- behavior depends on detector quality
- operator remains in control

### Slide 10: Future Work

Possible next steps:
- better open-vocabulary detection
- ladder and pallet specialization
- stronger temporal reasoning for CCTV streams
- richer SOP knowledge base
- multi-camera correlation

## 16. Key Files to Mention in a Presentation

If someone asks where things are implemented:

- Main app: `app/main.py`
- API routes: `app/api.py`
- Detection: `app/services/cv/detector.py`
- Processing pipeline: `app/services/processing.py`
- Vision reasoning: `app/services/scene_reasoning.py`
- SOP retrieval: `app/services/sop.py`
- LLM response generation: `app/services/llm.py`
- Telegram alerts and bot: `app/services/telegram.py`
- Telegram SOP text source: `sops/telegram_assistant_reference.txt`

## 17. One-Sentence Summary

Smart Facility Platform is a local-first computer vision and operator guidance
system that uses YOLO for detection, local vision for contextual reasoning, SOPs
for action grounding, and Telegram for alerting and follow-up assistance.
