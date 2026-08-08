import json
import logging

import httpx

from app.config import settings
from app.services.openai_compat import chat_completions
from app.services.openai_compat import coerce_text_content
from app.services.openai_compat import extract_chat_message_content
from app.services.openai_compat import list_models
from app.services.sop import SOPResult


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Smart Facility Platform safety incident assistant.
Use only the supplied incident fields and SOP excerpts. Never invent safety
instructions. If scene_assessment or scene_detections are supplied, ground the
summary in those detected objects and visible observations. Return JSON with
exactly two string keys: summary and
recommended_action. Keep each under 60 words."""

TELEGRAM_ASSISTANT_PROMPT = """You are the Smart Facility Telegram safety assistant.
Use only the supplied local SOP reference, matched SOP excerpts, and incident
fields. Never invent procedures. Answer the user's question directly in plain
text. If the user asks what to do next, give concise numbered steps. If the
question is ambiguous, ask for the incident ID or say you are using the latest
incident context if one was supplied. Keep the reply under 120 words."""


async def nemo_agent_runtime_status() -> dict[str, object]:
    if not settings.nemo_agent_enabled:
        return {"enabled": False, "reachable": False, "detail": "disabled"}

    try:
        await chat_completions(
            base_url=settings.nemo_agent_base_url,
            api_key=settings.nemo_agent_api_key,
            timeout_seconds=min(8.0, settings.nemo_agent_timeout_seconds),
            model=settings.nemo_agent_model,
            messages=[
                {"role": "system", "content": "Return the word ok."},
                {"role": "user", "content": "ok"},
            ],
            temperature=0,
        )
        return {"enabled": True, "reachable": True, "detail": "ready"}
    except httpx.HTTPStatusError as exc:
        return {
            "enabled": True,
            "reachable": False,
            "detail": f"HTTP {exc.response.status_code}",
        }
    except httpx.HTTPError as exc:
        return {
            "enabled": True,
            "reachable": False,
            "detail": str(exc),
        }


async def llm_runtime_status() -> dict[str, object]:
    if not settings.llm_enabled:
        return {"enabled": False, "reachable": False, "model_available": False}

    timeout = min(5.0, settings.llm_timeout_seconds)
    try:
        models = await list_models(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout_seconds=timeout,
        )
    except httpx.HTTPError as exc:
        return {
            "enabled": True,
            "reachable": False,
            "model_available": False,
            "detail": str(exc),
            "available_models": [],
        }

    return {
        "enabled": True,
        "reachable": True,
        "model_available": settings.llm_model in models,
        "available_models": models,
        "detail": (
            "ready"
            if settings.llm_model in models
            else f"Configured model '{settings.llm_model}' is not available"
        ),
    }


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
    summary_hint = str(incident.get("summary_hint") or "").strip()
    fallback_summary = (
        summary_hint
        if summary_hint
        else (
            f"{incident['object_type'].replace('_', ' ').title()} detected at "
            f"{incident['zone']} in {incident['facility']} with "
            f"{incident['confidence']:.0%} confidence."
        )
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


def _fallback_telegram_answer(
    question: str,
    incident: dict | None,
    sop_reference_path: str,
    sops: list[SOPResult],
) -> str:
    if incident:
        recommended = str(incident.get("recommended_action") or "").strip()
        incident_id = str(incident.get("incident_id") or incident.get("id") or "latest incident")
        incident_time = str(incident.get("created_at") or incident.get("first_seen") or "unknown time")
        zone = str(incident.get("zone") or "unknown location")
        if recommended:
            return (
                f"Using {incident_id} at {zone} on {incident_time} and the local SOP reference "
                f"({sop_reference_path}):\n{recommended}"
            )
        if sops:
            actions = sops[0].metadata.get("initial_actions", [])
            if actions:
                steps = [
                    f"{index}. {str(action).rstrip('.') }."
                    for index, action in enumerate(actions[:3], start=1)
                ]
                return (
                    f"Using {incident_id} at {zone} on {incident_time} and the local SOP reference "
                    f"({sop_reference_path}):\n" + "\n".join(steps)
                )
    if sops:
        actions = sops[0].metadata.get("initial_actions", [])
        if actions:
            steps = [
                f"{index}. {str(action).rstrip('.') }."
                for index, action in enumerate(actions[:3], start=1)
            ]
            return (
                f"Based on the local SOP reference ({sop_reference_path}) and {sops[0].title}:\n"
                + "\n".join(steps)
            )
    normalized_question = question.strip() or "your request"
    return (
        f"I can answer using the local SOP reference at {sop_reference_path}. "
        f"For incident-specific guidance, send an incident ID like INC-YYYYMMDD-HHMMSS-000 or "
        f"ask what to do next for the latest incident. Current question: {normalized_question}"
    )


async def create_grounded_summary_direct(
    incident: dict,
    sops: list[SOPResult],
) -> tuple[str, str]:
    fallback = _fallback(incident, sops)
    excerpts = [
        {"title": sop.title, "metadata": sop.metadata, "content": sop.content[:3000]}
        for sop in sops
    ]
    user_content = json.dumps({"incident": incident, "sops": excerpts}, default=str)

    if settings.nemo_agent_enabled:
        try:
            payload = await chat_completions(
                base_url=settings.nemo_agent_base_url,
                api_key=settings.nemo_agent_api_key,
                timeout_seconds=settings.nemo_agent_timeout_seconds,
                model=settings.nemo_agent_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
            )
            content = extract_chat_message_content(payload)
            return _parse_result(content, fallback)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "NeMo agent workflow unavailable; falling back to local summary path: %s",
                exc,
            )

    if not settings.llm_enabled:
        return fallback

    try:
        payload = await chat_completions(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        content = extract_chat_message_content(payload)
        return _parse_result(content, fallback)
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


async def answer_sop_question_direct(
    question: str,
    incident: dict | None,
    sop_reference_path: str,
    sop_reference_text: str,
    sops: list[SOPResult],
) -> str:
    fallback = _fallback_telegram_answer(question, incident, sop_reference_path, sops)
    excerpts = [
        {"title": sop.title, "metadata": sop.metadata, "content": sop.content[:2000]}
        for sop in sops
    ]
    user_content = json.dumps(
        {
            "question": question,
            "incident": incident,
            "sop_reference_path": sop_reference_path,
            "sop_reference_text": sop_reference_text[:12000],
            "matched_sops": excerpts,
        },
        default=str,
    )

    if settings.nemo_agent_enabled:
        try:
            payload = await chat_completions(
                base_url=settings.nemo_agent_base_url,
                api_key=settings.nemo_agent_api_key,
                timeout_seconds=settings.nemo_agent_timeout_seconds,
                model=settings.nemo_agent_model,
                messages=[
                    {"role": "system", "content": TELEGRAM_ASSISTANT_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
            )
            content = coerce_text_content(extract_chat_message_content(payload))
            return content or fallback
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # The Telegram assistant must still answer from the local SOP reference
            # even when the NeMo workflow is unavailable.
            pass

    if not settings.llm_enabled:
        return fallback

    try:
        payload = await chat_completions(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": TELEGRAM_ASSISTANT_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        content = coerce_text_content(extract_chat_message_content(payload))
        return content or fallback
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


async def create_grounded_summary(
    incident: dict,
    sops: list[SOPResult],
) -> tuple[str, str]:
    return await create_grounded_summary_direct(incident, sops)


async def answer_sop_question(
    question: str,
    incident: dict | None,
    sop_reference_path: str,
    sop_reference_text: str,
    sops: list[SOPResult],
) -> str:
    return await answer_sop_question_direct(
        question,
        incident,
        sop_reference_path,
        sop_reference_text,
        sops,
    )
