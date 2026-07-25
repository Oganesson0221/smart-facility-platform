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


def _parse_sop(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, frontmatter, content = text.split("---", 2)
        return yaml.safe_load(frontmatter) or {}, content.strip()
    return {}, text


def search_sops(event_type: str, facility: str, object_type: str, limit: int = 2) -> list[SOPResult]:
    query_tokens = set(re.findall(r"[a-z0-9]+", f"{event_type} {facility} {object_type}".lower()))
    results: list[SOPResult] = []
    for path in (ROOT / "sops").glob("*.md"):
        metadata, content = _parse_sop(path)
        searchable = f"{metadata} {content}".lower()
        score = sum(1 for token in query_tokens if token in searchable)
        facility_value = str(metadata.get("facility", "")).lower()
        if facility_value and facility_value not in ("all", facility.lower()):
            score -= 2
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
