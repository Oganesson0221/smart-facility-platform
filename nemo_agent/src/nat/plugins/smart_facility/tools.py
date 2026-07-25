from pathlib import Path

from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import register_function


class FacilitySOPConfig(FunctionBaseConfig, name="smart_facility_sop"):
    sop_directory: str = "./sops"


@register_function(config_type=FacilitySOPConfig)
async def build_facility_sop(config: FacilitySOPConfig, _builder: Builder):
    async def lookup_safety_sop(event_type: str, object_type: str) -> str:
        """Retrieve local facility SOPs relevant to a detected safety violation."""
        root = Path(config.sop_directory).resolve()
        if not root.is_dir():
            return "No SOP directory is available."
        query_terms = {
            term.lower()
            for term in f"{event_type} {object_type}".replace("_", " ").split()
            if len(term) > 2
        }
        ranked: list[tuple[int, Path, str]] = []
        for path in root.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            lowered = content.lower()
            score = sum(lowered.count(term) for term in query_terms)
            ranked.append((score, path, content))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        if not ranked:
            return "No matching local SOP was found."
        return "\n\n".join(
            f"SOURCE: {path.name}\n{content[:5000]}"
            for _, path, content in ranked[:2]
        )

    yield FunctionInfo.from_fn(
        lookup_safety_sop,
        description=(
            "Retrieve local safety procedures. Always call this before recommending "
            "an action for a facility incident."
        ),
    )
