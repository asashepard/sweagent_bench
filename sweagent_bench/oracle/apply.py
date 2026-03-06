"""Apply structured edits to AGENTS.md via LLM merging."""
from __future__ import annotations

import re

from sweagent_bench.kb.agents_md import AGENTS_MD_CHAR_BUDGET
from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.schema import Edit

_APPLY_SYSTEM = """\
You are an editor for AGENTS.md files — concise instruction documents
that guide a coding assistant's behavior on a specific repository.

You will be given the current AGENTS.md and a list of edits to apply.
Each edit specifies a section, an action (add/modify/strengthen/remove),
and content.

Rules:
- Apply ALL edits faithfully.
- "add" → insert the content into the specified section (create section if needed).
- "modify" → replace or rephrase the closest matching rule in that section.
- "strengthen" → make an existing rule more specific/forceful.
- "remove" → delete the matching rule from that section.
- You have explicit liberty to delete, rewrite, or reorganize existing AGENTS.md content when needed to apply edits well.
- Prefer concise guidance; remove low-value or redundant static details when they conflict with stronger guidance.
- Also resolve direct conflicts with existing guidance:
    - If an old rule directly contradicts a newly edited rule, keep the newly edited rule and remove the conflicting old rule.
    - If an old rule is a weaker duplicate of a newly strengthened rule, keep only the stronger/newer phrasing.
    - Do not invent unrelated new rules while resolving conflicts.
    - Keep conflict cleanup local to touched sections when possible.
- Keep the final AGENTS.md under {char_budget} characters.
- Preserve clear markdown readability.
- Output ONLY the updated AGENTS.md. No commentary."""

_APPLY_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

EDITS TO APPLY:
{edits}

Output the updated AGENTS.MD."""


def apply_edits(
    agents_md: str, edits: list[Edit], model: str, *, timeout_s: int = 120,
) -> str:
    if not edits:
        return agents_md

    edits_text = "\n".join(
        f"- [{e.action.upper()}] Section: {e.section} — {e.content}"
        for e in edits
    )

    messages = [
        {"role": "system", "content": _APPLY_SYSTEM.format(char_budget=AGENTS_MD_CHAR_BUDGET)},
        {"role": "user", "content": _APPLY_USER.format(agents_md=agents_md, edits=edits_text)},
    ]
    raw = chat_completion(model=model, messages=messages, temperature=0.2, max_tokens=2048, timeout_s=timeout_s)

    result = raw.strip()
    result = re.sub(r"^```(?:markdown)?\s*", "", result)
    result = re.sub(r"\s*```$", "", result)
    result = _remove_exact_duplicate_bullets(result)

    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = result[:AGENTS_MD_CHAR_BUDGET - 20] + "\n\n[... truncated]"

    return result


def _remove_exact_duplicate_bullets(agents_md: str) -> str:
    """Remove exact duplicate bullet lines while preserving order/format."""
    out_lines: list[str] = []
    seen_bullets: set[str] = set()
    for line in agents_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            normalized = " ".join(stripped.split())
            if normalized in seen_bullets:
                continue
            seen_bullets.add(normalized)
        out_lines.append(line)
    return "\n".join(out_lines)
