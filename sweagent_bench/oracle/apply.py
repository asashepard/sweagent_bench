"""Apply structured edits to AGENTS.md via mechanical section editing.

No LLM calls. Parses markdown sections, inserts/removes bullets,
enforces character budget by trimming longest sections. Deterministic,
cannot fail, cannot produce reasoning contamination.
"""
from __future__ import annotations

import re

from sweagent_bench.kb.agents_md import AGENTS_MD_CHAR_BUDGET
from sweagent_bench.oracle.schema import Edit


def apply_edits(
    agents_md: str, edits: list[Edit], model: str = "", *, timeout_s: int = 120,
) -> tuple[str, dict]:
    """Mechanically apply edits to AGENTS.md. Always succeeds."""
    if not edits:
        return agents_md, {"accepted": True, "edits_applied": 0, "over_budget_trimmed": False}

    sections = _parse_sections(agents_md)

    applied = 0
    for edit in edits:
        action = edit.action.strip().lower()
        content = edit.content.strip()
        if not content:
            continue
        if action == "remove":
            _remove_from_section(sections, edit.section, content)
        else:  # add, modify, strengthen → insert bullet
            _add_to_section(sections, edit.section, content)
        applied += 1

    result = _render(sections)
    trimmed = False
    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = _trim_to_budget(sections, AGENTS_MD_CHAR_BUDGET)
        trimmed = True

    return result, {"accepted": True, "edits_applied": applied, "over_budget_trimmed": trimmed}


# ---------------------------------------------------------------------------
# Section parser / renderer
# ---------------------------------------------------------------------------

def _parse_sections(md: str) -> list[dict]:
    """Split markdown into ordered [{title, lines}] blocks."""
    sections: list[dict] = []
    cur: dict = {"title": "", "lines": []}
    for line in md.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            if cur["title"] or cur["lines"]:
                sections.append(cur)
            cur = {"title": line, "lines": []}
        else:
            cur["lines"].append(line)
    if cur["title"] or cur["lines"]:
        sections.append(cur)
    return sections


def _render(sections: list[dict]) -> str:
    parts: list[str] = []
    for sec in sections:
        if sec["title"]:
            parts.append(sec["title"])
        parts.extend(sec["lines"])
    return "\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# Edit operations
# ---------------------------------------------------------------------------

# Canonical aliases — maps common LLM-generated section names to the
# standard heading used in the static KB so edits merge instead of sprawling.
_SECTION_ALIASES: dict[str, str] = {
    "code change requirements": "operating mode",
    "code change protocol": "operating mode",
    "change protocol": "operating mode",
    "workflow": "operating mode",
    "exploration strategy": "operating mode",
    "localization": "operating mode",
    "dependency tracing": "operating mode",
    "general": "operating mode",
    "guardrail": "guardrails",
    "guard rails": "guardrails",
    "safety": "guardrails",
    "constraints": "guardrails",
    "validation": "repo priors",
    "testing": "repo priors",
    "test strategy": "repo priors",
    "conventions": "repo priors",
    "integration risk": "repo priors",
    "high-impact hubs": "repo priors",
    "entry points": "repo priors",
}


def _normalize_section_name(name: str) -> str:
    """Resolve aliases to canonical section names."""
    key = name.lower().strip()
    return _SECTION_ALIASES.get(key, key)


def _find_section(sections: list[dict], name: str) -> dict | None:
    target = _normalize_section_name(name)
    for sec in sections:
        title_text = re.sub(r"^#{1,3}\s+", "", sec["title"]).strip().lower()
        # exact match or substring
        if title_text == target or target in title_text:
            return sec
    return None


def _add_to_section(sections: list[dict], name: str, content: str) -> None:
    sec = _find_section(sections, name)
    if sec is None:
        sections.append({"title": f"## {name}", "lines": [f"- {content}"]})
    else:
        sec["lines"].append(f"- {content}")


def _remove_from_section(sections: list[dict], name: str, content: str) -> None:
    sec = _find_section(sections, name)
    if sec is None:
        return
    target = " ".join(content.lower().split())
    sec["lines"] = [
        line for line in sec["lines"]
        if target not in " ".join(line.lower().split())
    ]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def _trim_to_budget(sections: list[dict], budget: int) -> str:
    """Remove last bullet from longest section until under budget."""
    while True:
        rendered = _render(sections)
        if len(rendered) <= budget:
            return rendered
        # find section with most content chars
        content_secs = [s for s in sections if s["lines"] and any(l.strip() for l in s["lines"])]
        if not content_secs:
            # headers alone exceed budget — hard truncate
            return rendered[:budget]
        longest = max(content_secs, key=lambda s: sum(len(l) for l in s["lines"]))
        if not longest["lines"]:
            return rendered[:budget]
        longest["lines"].pop()
