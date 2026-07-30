import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import ROOT


@dataclass
class SOPResult:
    title: str
    content: str
    source: str
    metadata: dict
    score: float


_VEHICLE_OBJECT_TYPES = {"vehicle", "car", "truck", "motorcycle", "bus", "van"}
TELEGRAM_SOP_REFERENCE = ROOT / "sops" / "telegram_assistant_reference.txt"


def _parse_sop(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, frontmatter, content = text.split("---", 2)
        return yaml.safe_load(frontmatter) or {}, content.strip()
    return {}, text


def _object_family(value: str) -> str:
    normalized = value.strip().lower()
    return "vehicle" if normalized in _VEHICLE_OBJECT_TYPES else normalized


def search_sops(event_type: str, facility: str, object_type: str, limit: int = 2) -> list[SOPResult]:
    query_tokens = set(re.findall(r"[a-z0-9]+", f"{event_type} {facility} {object_type}".lower()))
    results: list[SOPResult] = []
    for path in (ROOT / "sops").glob("*.md"):
        metadata, content = _parse_sop(path)
        searchable = f"{metadata} {content}".lower()
        score = float(sum(1 for token in query_tokens if token in searchable))
        expected_event = event_type.strip().lower()
        expected_object = object_type.strip().lower()
        metadata_event = str(metadata.get("event_type", "")).strip().lower()
        metadata_object = str(metadata.get("object_type", "")).strip().lower()
        if metadata_event:
            if metadata_event == expected_event:
                score += 12
            elif metadata_event in expected_event or expected_event in metadata_event:
                score += 4
            else:
                score -= 4
        if metadata_object:
            if metadata_object == expected_object:
                score += 8
            elif _object_family(metadata_object) == _object_family(expected_object):
                score += 6
            else:
                score -= 3
        facility_value = str(metadata.get("facility", "")).lower()
        if facility_value and facility_value not in ("all", facility.lower()):
            score -= 2
        elif facility_value:
            score += 2
        results.append(
            SOPResult(
                title=metadata.get("title", path.stem.replace("_", " ").title()),
                content=content,
                source=f"sops/{path.name}",
                metadata=metadata,
                score=score,
            )
        )
    return sorted(results, key=lambda item: item.score, reverse=True)[:limit]


def load_telegram_sop_reference() -> tuple[str, str]:
    path = TELEGRAM_SOP_REFERENCE
    return str(path.relative_to(ROOT)), path.read_text(encoding="utf-8").strip()
