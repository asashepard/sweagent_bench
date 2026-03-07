"""Apply structured edits to AGENTS.md via mechanical section editing.

No LLM calls. Parses markdown sections, inserts/removes bullets,
enforces character budget by trimming longest sections. Deterministic,
cannot fail, cannot produce reasoning contamination.

Edits are constrained to canonical static-KB sections. Edits targeting
unknown sections are remapped via aliases or dropped. Edits containing
benchmark/pipeline boilerplate are filtered out.
"""
from __future__ import annotations

import re

from sweagent_bench.kb.agents_md import AGENTS_MD_CHAR_BUDGET
from sweagent_bench.oracle.schema import Edit


# ---------------------------------------------------------------------------
# Canonical section allowlist + alias remapping
# ---------------------------------------------------------------------------

# Sections that exist in the static KB (v0). Edits MUST resolve to one of
# these after alias remapping, or they are dropped.
_CANONICAL_SECTIONS = {
    "operating mode",
    "repo priors",
    "high-impact hubs",
    "entry points",
    "import chains",
    "validation",
    "integration risk",
    "conventions",
    "guardrails",
}

# Maps common LLM-invented section names to the canonical heading they
# belong to, so edits merge instead of creating new generic sections.
_SECTION_ALIASES: dict[str, str] = {
    # → operating mode
    "code change requirements": "operating mode",
    "code change protocol": "operating mode",
    "change protocol": "operating mode",
    "change process": "operating mode",
    "workflow": "operating mode",
    "exploration strategy": "operating mode",
    "localization": "operating mode",
    "dependency tracing": "operating mode",
    "general": "operating mode",
    "regression fix workflow": "operating mode",
    "regression handling": "operating mode",
    "patch requirements": "operating mode",
    # → guardrails
    "guardrail": "guardrails",
    "guard rails": "guardrails",
    "safety": "guardrails",
    "constraints": "guardrails",
    "fallback behavior": "guardrails",
    # → validation
    "testing": "validation",
    "test strategy": "validation",
    "validation requirements": "validation",
    "validation protocol": "validation",
    # → repo priors (parent)
    "conventions": "conventions",
    "integration risk": "integration risk",
    "high-impact hubs": "high-impact hubs",
    "entry points": "entry points",
    "import chains": "import chains",
}

# ---------------------------------------------------------------------------
# Content filter — reject benchmark/pipeline boilerplate
# ---------------------------------------------------------------------------

# If ANY of these patterns appear in the edit content (case-insensitive),
# the edit is rejected before insertion.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"patch_len",
        r"patch_source",
        r"fallback_single_shot",
        r"non[_\-\s]?zero\s+diff",
        r"execution\s+summary\s+format",
        r"placeholder\s+fields",
        r"runner_status",
        r"runner_error",
        r"runner_token",
        r"token_usage",
        r"elapsed_s",
        r"show\s+(validation|test)\s+output",
        r"report\s+pass/fail\s+status",
        r"include\s+(validation|test)\s+output",
        r"explicit\s+pass/fail",
        r"confirm.*regression.*resolved\s+before\s+submission",
    ]
]


def _is_boilerplate(content: str) -> bool:
    """Return True if content matches benchmark/pipeline boilerplate."""
    return any(pat.search(content) for pat in _BOILERPLATE_PATTERNS)


def _normalize_section_name(name: str) -> str:
    """Resolve aliases to canonical section names."""
    key = name.lower().strip()
    return _SECTION_ALIASES.get(key, key)


def _is_canonical(name: str) -> bool:
    """Return True if the normalized name maps to a canonical section."""
    return _normalize_section_name(name) in _CANONICAL_SECTIONS


# ---------------------------------------------------------------------------
# Edit priority — repo-specific sections consume budget first
# ---------------------------------------------------------------------------

# Sections whose content is inherently repo-specific.
_REPO_SPECIFIC_SECTIONS = {
    "high-impact hubs",
    "entry points",
    "import chains",
    "integration risk",
    "conventions",
}


def _edit_priority(edit: Edit) -> int:
    """Lower value = higher priority. Repo-specific first, generic last."""
    canon = _normalize_section_name(edit.section)
    if canon in _REPO_SPECIFIC_SECTIONS:
        return 0
    if canon == "validation":
        return 1
    return 2  # operating mode, guardrails, repo priors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_edits(
    agents_md: str, edits: list[Edit], model: str = "", *, timeout_s: int = 120,
) -> tuple[str, dict]:
    """Mechanically apply edits to AGENTS.md. Always succeeds.

    All valid edits are applied unconditionally. If the result exceeds
    AGENTS_MD_CHAR_BUDGET, bullets are trimmed from the lowest-priority
    (most generic) sections first, preserving repo-specific content.
    """
    if not edits:
        return agents_md, {
            "accepted": True, "edits_applied": 0,
            "edits_dropped": 0, "budget_trimmed": 0,
        }

    sections = _parse_sections(agents_md)

    # Gate pass: filter boilerplate and non-canonical edits.
    applied = 0
    dropped = 0
    for edit in edits:
        action = edit.action.strip().lower()
        content = edit.content.strip()
        if not content:
            continue
        if action != "remove" and _is_boilerplate(content):
            dropped += 1
            continue
        if not _is_canonical(edit.section):
            dropped += 1
            continue
        if action == "remove":
            _remove_from_section(sections, edit.section, content)
        else:
            _add_to_section(sections, edit.section, content)
        applied += 1

    result = _render(sections)
    budget_trimmed = 0
    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result, budget_trimmed = _trim_to_budget(sections, AGENTS_MD_CHAR_BUDGET)

    return result, {
        "accepted": True, "edits_applied": applied,
        "edits_dropped": dropped, "budget_trimmed": budget_trimmed,
    }


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
        # Section doesn't exist in the document — drop silently.
        # Canonical-section gate already ran in apply_edits().
        return
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

def _section_priority(sec: dict) -> int:
    """Priority of a parsed section. Higher value = shed first."""
    title_text = re.sub(r"^#{1,3}\s+", "", sec["title"]).strip().lower()
    if title_text in _REPO_SPECIFIC_SECTIONS:
        return 0   # shed last
    if title_text == "validation":
        return 1
    return 2       # operating mode, guardrails — shed first


def _trim_to_budget(sections: list[dict], budget: int) -> tuple[str, int]:
    """Pop bullets from generic sections first until under budget.

    Returns (rendered_str, bullets_removed).
    """
    removed = 0
    while True:
        rendered = _render(sections)
        if len(rendered) <= budget:
            return rendered, removed
        # Collect sections that have strippable content, sorted so
        # lowest-priority (most generic) come first.
        strippable = [
            s for s in sections
            if s["lines"] and any(l.strip() for l in s["lines"])
        ]
        if not strippable:
            return rendered[:budget], removed
        # Among the lowest-priority tier, pick the section with
        # the most content so trimming converges quickly.
        strippable.sort(key=lambda s: (-_section_priority(s), -sum(len(l) for l in s["lines"])))
        target = strippable[0]
        if not target["lines"]:
            return rendered[:budget], removed
        target["lines"].pop()
        removed += 1
