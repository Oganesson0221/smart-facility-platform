import json

import httpx

from app.config import settings
from app.services.sop import SOPResult


SYSTEM_PROMPT = """You are the Smart Facility Platform safety incident assistant.
Use only the supplied incident fields and SOP excerpts. Never invent safety
instructions. Return JSON with exactly two string keys: summary and
recommended_action. Keep each under 60 words."""


def _fallback(incident: dict, sops: list[SOPResult]) -> tuple[str, str]:
    fallback_action = "Confirm the location and notify Facilities Security."
    if sops:
        actions = sops[0].metadata.get("initial_actions", [])
        if actions:
            steps = [
                f"{index}. {str(action).rstrip('.') }."
                for index, action in enumerate(actions[:2], start=1)
            ]
            escalation_minutes = sops[0].metadata.get("escalation_after_minutes")
            escalation_team = sops[0].metadata.get(
                "escalation_team", "the duty facility manager"
            )
            if escalation_minutes:
                steps.append(
                    f"{len(steps) + 1}. If unresolved after {escalation_minutes} "
                    f"minutes, escalate to {escalation_team}."
                )
            fallback_action = "\n".join(steps)
    fallback_summary = (
        f"{incident['object_type'].replace('_', ' ').title()} detected at "
        f"{incident['zone']} in {incident['facility']} with "
        f"{incident['confidence']:.0%} confidence."
    )
    if incident.get("overlap", 0):
        fallback_summary = fallback_summary.rstrip(".") + (
            f" and {incident['overlap']:.0%} protected-zone overlap."
        )
    return fallback_summary, fallback_action


def _parse_result(content: str, fallback: tuple[str, str]) -> tuple[str, str]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw.startswith("{"):
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    result = json.loads(raw)
    return (
        str(result.get("summary") or fallback[0]),
        str(result.get("recommended_action") or fallback[1]),
    )


async def create_grounded_summary(incident: dict, sops: list[SOPResult]) -> tuple[str, str]:
    fallback = _fallback(incident, sops)
    excerpts = [
        {"title": sop.title, "metadata": sop.metadata, "content": sop.content[:3000]}
        for sop in sops
    ]
    user_content = json.dumps({"incident": incident, "sops": excerpts}, default=str)

    if settings.nemo_agent_enabled:
        payload = {
            "model": settings.nemo_agent_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        headers = {}
        if settings.nemo_agent_api_key:
            headers["Authorization"] = f"Bearer {settings.nemo_agent_api_key}"
        try:
            async with httpx.AsyncClient(
                timeout=settings.nemo_agent_timeout_seconds
            ) as client:
                response = await client.post(
                    f"{settings.nemo_agent_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _parse_result(content, fallback)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            if settings.nemo_agent_required:
                raise RuntimeError("NVIDIA NeMo Agent Toolkit workflow is unavailable")

    if not settings.llm_enabled:
        return fallback

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_result(content, fallback)
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback
