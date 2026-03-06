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

_CLEANUP_SYSTEM = """\
You clean LLM output into final AGENTS.md content.

Return ONLY the final AGENTS.md markdown.
Do not include analysis, role/task/constraint text, planning steps, or code fences.
Keep it under {char_budget} characters.
"""

_CLEANUP_USER = """\
Extract and return only the final AGENTS.md content from this text.

TEXT:
---
{text}
---

Return only AGENTS.md markdown.
"""

_LEADING_META_PATTERNS = [
    r"^\s*thinking\s+process\s*:",
    r"^\s*analysis\s*:",
    r"^\s*reasoning\s*:",
    r"^\s*analyze\s+the\s+request\s*:",
    r"^\s*role\s*:",
    r"^\s*task\s*:",
    r"^\s*constraints\s*:",
    r"^\s*solution\s*:",
    r"^\s*\[\.\.\.\s*truncated\s*\]\s*$",
]


def _preview(text: str, max_len: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_agents_markdown(text: str) -> str:
    if not text.strip():
        return ""

    patterns = [
        r"^#\s*AGENTS\\.md",
        r"^#\s*AGENTS",
        r"^##\s*Operating Mode",
        r"^##\s*Repo Priors",
        r"^##\s*Guardrails",
    ]
    starts: list[int] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            starts.append(match.start())

    if starts:
        return text[min(starts):].strip()
    return text.strip()


def _strip_leading_meta_preamble(text: str) -> str:
    lines = text.splitlines()
    kept_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept_start = idx + 1
            continue
        if any(re.match(pat, stripped, flags=re.IGNORECASE) for pat in _LEADING_META_PATTERNS):
            kept_start = idx + 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            kept_start = idx + 1
            continue
        break

    return "\n".join(lines[kept_start:]).strip()


def _sanitize_candidate(text: str) -> str:
    out = _strip_code_fences(text)
    out = _strip_think_blocks(out)
    out = _strip_leading_meta_preamble(out)
    out = _extract_agents_markdown(out)
    out = _remove_exact_duplicate_bullets(out)
    return out.strip()


def _find_invalid_reason(agents_md: str) -> str | None:
    if not agents_md.strip():
        return "empty output after sanitization"

    head = "\n".join(agents_md.splitlines()[:24])
    for pat in _LEADING_META_PATTERNS:
        if re.search(pat, head, flags=re.IGNORECASE | re.MULTILINE):
            return f"contains meta/editor preamble matching: {pat}"

    if len(agents_md) > AGENTS_MD_CHAR_BUDGET:
        return f"exceeds character budget: {len(agents_md)} > {AGENTS_MD_CHAR_BUDGET}"
    return None


def _make_stage_snapshot(stage: str, text: str, invalid_reason: str | None) -> dict:
    return {
        "stage": stage,
        "length": len(text),
        "invalid_reason": invalid_reason,
        "preview": _preview(text),
    }


def _cleanup_with_llm(text: str, model: str, *, timeout_s: int) -> str:
    messages = [
        {"role": "system", "content": _CLEANUP_SYSTEM.format(char_budget=AGENTS_MD_CHAR_BUDGET)},
        {"role": "user", "content": _CLEANUP_USER.format(text=text)},
    ]
    return chat_completion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        timeout_s=timeout_s,
    )


def apply_edits(
    agents_md: str, edits: list[Edit], model: str, *, timeout_s: int = 120,
) -> tuple[str, dict]:
    if not edits:
        noop_trace = [_make_stage_snapshot("noop", agents_md, None)]
        return agents_md, {
            "accepted": True,
            "reason": None,
            "raw_output": "",
            "sanitized_output": agents_md,
            "cleanup_output": "",
            "cleaned_output": agents_md,
            "stage": "noop",
            "used_cleanup": False,
            "trace": noop_trace,
        }

    edits_text = "\n".join(
        f"- [{e.action.upper()}] Section: {e.section} — {e.content}"
        for e in edits
    )

    messages = [
        {"role": "system", "content": _APPLY_SYSTEM.format(char_budget=AGENTS_MD_CHAR_BUDGET)},
        {"role": "user", "content": _APPLY_USER.format(agents_md=agents_md, edits=edits_text)},
    ]
    raw = chat_completion(model=model, messages=messages, temperature=0.2, max_tokens=2048, timeout_s=timeout_s)

    trace: list[dict] = [_make_stage_snapshot("raw", raw, None)]
    sanitized = _sanitize_candidate(raw)
    invalid_reason_sanitized = _find_invalid_reason(sanitized)
    trace.append(_make_stage_snapshot("sanitized", sanitized, invalid_reason_sanitized))

    invalid_reason = invalid_reason_sanitized
    cleanup_raw = ""
    cleaned = sanitized
    stage = "sanitized"
    used_cleanup = False

    if invalid_reason:
        used_cleanup = True
        cleanup_raw = _cleanup_with_llm(raw, model, timeout_s=timeout_s)
        trace.append(_make_stage_snapshot("cleanup_raw", cleanup_raw, None))
        cleaned = _sanitize_candidate(cleanup_raw)
        invalid_reason = _find_invalid_reason(cleaned)
        trace.append(_make_stage_snapshot("cleaned", cleaned, invalid_reason))
        stage = "cleanup"

    if invalid_reason:
        return agents_md, {
            "accepted": False,
            "reason": invalid_reason,
            "raw_output": raw,
            "sanitized_output": sanitized,
            "cleanup_output": cleanup_raw,
            "cleaned_output": cleaned,
            "stage": stage,
            "used_cleanup": used_cleanup,
            "trace": trace,
        }

    return cleaned, {
        "accepted": True,
        "reason": None,
        "raw_output": raw,
        "sanitized_output": sanitized,
        "cleanup_output": cleanup_raw,
        "cleaned_output": cleaned,
        "stage": stage,
        "used_cleanup": used_cleanup,
        "trace": trace,
    }


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
