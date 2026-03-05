"""LLM-as-judge evaluator for micro-test probes."""
from __future__ import annotations

import json
import re

from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.schema import BehaviorReview, Edit, Probe, ProbeResult

SYSTEM_PROMPT_CHAR_BUDGET = 60_000

_SIMULATE_SYSTEM = """\
You are a coding assistant helping with the repository described below.
Follow the AGENTS.md guidelines when answering.

{agents_md}"""

_JUDGE_SYSTEM = """\
You are an evaluator/editor for AGENTS.md quality.
You will be given a TASK, the assistant RESPONSE, and EXPECTED BEHAVIORS.

Assess each behavior with one of: "strong", "partial", "missing".

Return a JSON object with this exact shape:
{
    "behavior_reviews": [
        {
            "behavior": "...",
            "assessment": "strong|partial|missing",
            "evidence": "short evidence from response",
            "improvement": "what AGENTS.md should add/change"
        }
    ],
    "proposed_edits": [
        {"section": "...", "action": "add|modify|strengthen|remove", "content": "..."}
    ],
    "overall_notes": "short summary"
}

Rules:
- Prefer concrete, testable edits over vague advice.
- Proposed edits are optional.
- If behavior is already strong enough, return an empty "proposed_edits": [].
- If behavior has multiple independent gaps, propose multiple edits.
- Only propose edits you are confident will materially improve future behavior.
- Prefer "modify"/"strengthen" over many new "add" rules.
- Keep edits reusable and repo-level; avoid one-off file-path or one-off command prescriptions.
- Return at most 3 proposed edits per probe.
- Do NOT force edits just to fill the list.
- Output ONLY valid JSON."""

_JUDGE_USER = """\
TASK:
{task}

RESPONSE:
{response}

EXPECTED BEHAVIORS:
{behaviors}

Produce behavior_reviews and proposed_edits JSON."""


def simulate_response(agents_md: str, probe: Probe, model: str, *, timeout_s: int = 120) -> str:
    system = _SIMULATE_SYSTEM.format(agents_md=agents_md[:SYSTEM_PROMPT_CHAR_BUDGET])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": probe.task},
    ]
    return chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=1024, timeout_s=timeout_s)


def review_probe(
    task: str, response: str, expected_behaviors: list[str],
    model: str, *, timeout_s: int = 120,
) -> tuple[list[BehaviorReview], list[Edit], str]:
    behaviors_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(expected_behaviors))
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER.format(
            task=task, response=response, behaviors=behaviors_text,
        )},
    ]
    raw = chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=2048, timeout_s=timeout_s)
    return _parse_review(raw, expected_behaviors)


def _parse_review(raw: str, expected_behaviors: list[str]) -> tuple[list[BehaviorReview], list[Edit], str]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_reviews(expected_behaviors)
        else:
            return _fallback_reviews(expected_behaviors)

    if not isinstance(obj, dict):
        return _fallback_reviews(expected_behaviors)

    reviews_raw = obj.get("behavior_reviews", [])
    edits_raw = obj.get("proposed_edits", [])
    overall_notes = str(obj.get("overall_notes", "")).strip()

    reviews: list[BehaviorReview] = []
    for i, b in enumerate(expected_behaviors):
        if i < len(reviews_raw) and isinstance(reviews_raw[i], dict):
            item = reviews_raw[i]
            assessment = str(item.get("assessment", "partial")).strip().lower()
            if assessment not in {"strong", "partial", "missing"}:
                assessment = "partial"
            reviews.append(BehaviorReview(
                behavior=str(item.get("behavior", b)),
                assessment=assessment,
                evidence=str(item.get("evidence", "")).strip(),
                improvement=str(item.get("improvement", "")).strip(),
            ))
            continue
        reviews.append(BehaviorReview(
            behavior=b, assessment="missing",
            evidence="Review missing",
            improvement="Add explicit instruction for this behavior to AGENTS.md.",
        ))

    valid_actions = {"add", "modify", "strengthen", "remove"}
    edits: list[Edit] = []
    for item in edits_raw if isinstance(edits_raw, list) else []:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "")).strip() or "General"
        action = str(item.get("action", "add")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if action not in valid_actions:
            action = "add"
        edits.append(Edit(section=section, action=action, content=content))

    return reviews, edits, overall_notes


def _fallback_reviews(expected_behaviors: list[str]) -> tuple[list[BehaviorReview], list[Edit], str]:
    reviews = [
        BehaviorReview(
            behavior=b, assessment="missing",
            evidence="Failed to parse reviewer output.",
            improvement="Add explicit AGENTS.md guidance for this behavior.",
        )
        for b in expected_behaviors
    ]
    return reviews, [], "Parse error in reviewer output"


def evaluate_probe(agents_md: str, probe: Probe, model: str, *, timeout_s: int = 120) -> ProbeResult:
    response = simulate_response(agents_md, probe, model, timeout_s=timeout_s)
    behavior_reviews, proposed_edits, overall_notes = review_probe(
        task=probe.task, response=response,
        expected_behaviors=probe.expected_behaviors,
        model=model, timeout_s=timeout_s,
    )
    return ProbeResult(
        probe_id=probe.id, task=probe.task, response=response,
        behavior_reviews=behavior_reviews, proposed_edits=proposed_edits,
        overall_notes=overall_notes,
    )
