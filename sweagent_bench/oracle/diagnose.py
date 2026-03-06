"""Diagnostics aggregation — proposes structured edits to AGENTS.md."""
from __future__ import annotations

import json
import re

from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.schema import Edit, ProbeResult

_DIAGNOSE_SYSTEM = """\
You are an expert AGENTS.md editor. You will be given the current AGENTS.md
and diagnostic probe outcomes. Your goal is to improve future assistant
behavior by proposing targeted edits.

Output a JSON array of edit objects, each with:
- "section": which AGENTS.md section to edit
- "action": one of "add", "modify", "strengthen", "remove"
- "content": the specific text to add, modify, or strengthen

Rules:
- Keep edits specific and actionable.
- Focus on reusable repo-level exploration guidance (localize, trace dependencies, validate).
- Prefer minimal/scoped behavior over broad changes.
- Avoid one-off file paths/commands and speculative semantics changes.
- Prefer "modify"/"strengthen" over many new "add" rules.
- Edits are optional: return [] when confidence is low or guidance is already strong.
- Return at most 6 edits.
- Keep the total AGENTS.md under 3,000 characters.
- Output ONLY the JSON array."""

_DIAGNOSE_USER = """\
CURRENT AGENTS.MD:
---
{agents_md}
---

PROBE DIAGNOSTICS:
{diagnostics}

Propose edits to improve AGENTS.md for future iterations."""


def diagnose_failures(
    agents_md: str, results: list[ProbeResult],
    model: str, *, timeout_s: int = 120,
) -> list[Edit]:
    diagnostic_lines: list[str] = []
    for pr in results:
        diagnostic_lines.append(f"- Probe {pr.probe_id}: {pr.task}")
        for review in pr.behavior_reviews:
            diagnostic_lines.append(
                f"  * Behavior: {review.behavior} | "
                f"Assessment: {review.assessment} | "
                f"Evidence: {review.evidence} | "
                f"Improvement: {review.improvement}"
            )
        if pr.overall_notes:
            diagnostic_lines.append(f"  * Overall: {pr.overall_notes}")
        for edit in pr.proposed_edits:
            diagnostic_lines.append(
                f"  * ProposedEdit: {edit.action}@{edit.section}: {edit.content}"
            )

    if not diagnostic_lines:
        return []

    diagnostics_text = "\n".join(diagnostic_lines)
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": _DIAGNOSE_USER.format(
            agents_md=agents_md, diagnostics=diagnostics_text,
        )},
    ]
    raw = chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=2048, timeout_s=timeout_s)
    return _parse_edits(raw)


def _parse_edits(raw: str) -> list[Edit]:
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(arr, list):
        return []

    valid_actions = {"add", "modify", "strengthen", "remove"}
    edits: list[Edit] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "")).strip()
        action = str(item.get("action", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not section or not content:
            continue
        if action not in valid_actions:
            action = "add"
        edits.append(Edit(section=section, action=action, content=content))

    return edits
