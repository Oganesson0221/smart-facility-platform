import base64
import json
from typing import Any

import httpx

from app.config import settings


SCENE_PROMPT = """Inspect this facility image using only visible evidence.
Identify important objects and read any visible safety, access, parking, or
warning signs. Decide whether a rule or unsafe condition is visibly present.
Do not assume that an exit, restriction, or hazard exists unless the image
supports it.

Return one JSON object with exactly these fields:
- violation: boolean
- category: short string
- summary: one clear sentence
- evidence: array of short visible observations
- confidence: number from 0 to 1
- visible_objects: array of important object names
- annotations: array of objects with "label" and "box". Each box must be
  [x1,y1,x2,y2] using normalized coordinates from 0 to 1000. Add boxes only
  for the objects or signs that directly support the violation.
"""


def _parse_assessment(content: str) -> dict[str, Any]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Vision model did not return a JSON object")

    evidence = result.get("evidence", [])
    visible_objects = result.get("visible_objects", [])
    if not isinstance(evidence, list) or not isinstance(visible_objects, list):
        raise ValueError("Vision model returned invalid evidence")

    confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
    annotations = []
    for item in result.get("annotations", [])[:8]:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            coordinates = [max(0.0, min(1000.0, float(value))) for value in box]
        except (TypeError, ValueError):
            continue
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        annotations.append(
            {"label": str(item.get("label") or "evidence"), "box": coordinates}
        )
    return {
        "violation": bool(result.get("violation", False)),
        "category": str(result.get("category", "General")),
        "summary": str(result.get("summary", "No assessment was returned.")),
        "evidence": [str(item) for item in evidence[:8]],
        "confidence": confidence,
        "visible_objects": [str(item) for item in visible_objects[:12]],
        "annotations": annotations,
        "model": settings.vision_model,
        "local": True,
    }


async def assess_scene(image_bytes: bytes) -> dict[str, Any]:
    if not settings.vision_enabled:
        raise RuntimeError("Automatic scene reasoning is disabled")

    payload = {
        "model": settings.vision_model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": SCENE_PROMPT,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "options": {"temperature": 0},
    }
    try:
        async with httpx.AsyncClient(timeout=settings.vision_timeout_seconds) as client:
            response = await client.post(
                f"{settings.vision_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        return _parse_assessment(response.json()["message"]["content"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local vision model '{settings.vision_model}' is unavailable or returned "
            "an invalid response"
        ) from exc
