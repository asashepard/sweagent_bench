"""LLM-as-judge evaluator for micro-test probes."""
from __future__ import annotations

import json
import re

from sweagent_bench.generation.sweagent_runner import generate_patch_with_sweagent
from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.schema import BehaviorReview, Edit, Probe, ProbeResult

SYSTEM_PROMPT_CHAR_BUDGET = 60_000

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
- Use a strict rubric: evidence-first localization, dependency tracing, minimal scoped edits, targeted validation.
- Penalize speculative breadth (broad refactors, multi-location edits without evidence, invented behavior changes).
- Proposed edits are optional; return [] if behavior is already strong.
- Keep edits reusable and repo-level; avoid one-off file paths/commands.
- Prefer "modify"/"strengthen" over many new "add" rules.
- Return at most 3 proposed edits.
- Output ONLY valid JSON."""

_JUDGE_USER = """\
TASK:
{task}

RESPONSE:
{response}

EXPECTED BEHAVIORS:
{behaviors}

Produce behavior_reviews and proposed_edits JSON."""


def run_probe_with_runner(
    agents_md: str,
    probe: Probe,
    model: str,
    *,
    repo: str,
    commit: str,
    probe_timeout_s: int,
    probe_max_steps: int,
    api_base: str | None,
) -> tuple[str, dict]:
    instance = {
        "instance_id": f"oracle_probe__{probe.id}",
        "repo": repo,
        "base_commit": commit,
        "problem_statement": probe.task,
    }
    run_meta = generate_patch_with_sweagent(
        instance=instance,
        model=model,
        guidance_text=agents_md[:SYSTEM_PROMPT_CHAR_BUDGET],
        timeout_s=max(30, int(probe_timeout_s)),
        max_steps=max(1, int(probe_max_steps)),
        traj_dir=None,
        api_base=api_base,
    )

    patch = str(run_meta.get("patch", "") or "")
    patch_preview = ""
    if patch.strip():
        preview_lines = patch.splitlines()[:80]
        patch_preview = "\nPATCH_PREVIEW:\n```diff\n" + "\n".join(preview_lines) + "\n```"

    response = (
        "RUNNER_EXECUTION_SUMMARY\n"
        f"status: {run_meta.get('status')}\n"
        f"error: {run_meta.get('error')}\n"
        f"patch_source: {run_meta.get('patch_source')}\n"
        f"patch_len: {len(patch)}\n"
        f"elapsed_s: {float(run_meta.get('elapsed_s', 0.0) or 0.0):.2f}\n"
        f"stall_detected: {bool(run_meta.get('stall_detected', False))}\n"
        f"stall_type: {run_meta.get('stall_type')}\n"
        f"stall_repeat_count: {int(run_meta.get('stall_repeat_count', 0) or 0)}\n"
        f"no_bash_block_count: {int(run_meta.get('no_bash_block_count', 0) or 0)}\n"
        f"empty_bash_block_count: {int(run_meta.get('empty_bash_block_count', 0) or 0)}\n"
        f"repeated_command_stall_count: {int(run_meta.get('repeated_command_stall_count', 0) or 0)}\n"
        f"fallback_single_shot_used: {bool(run_meta.get('fallback_single_shot_used', False))}"
        f"{patch_preview}"
    )
    return response, run_meta


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


def evaluate_probe(
    agents_md: str,
    probe: Probe,
    model: str,
    *,
    repo: str,
    commit: str,
    timeout_s: int = 120,
    probe_timeout_s: int = 300,
    probe_max_steps: int = 8,
    api_base: str | None = None,
) -> ProbeResult:
    response, run_meta = run_probe_with_runner(
        agents_md,
        probe,
        model,
        repo=repo,
        commit=commit,
        probe_timeout_s=probe_timeout_s,
        probe_max_steps=probe_max_steps,
        api_base=api_base,
    )
    behavior_reviews, proposed_edits, overall_notes = review_probe(
        task=probe.task, response=response,
        expected_behaviors=probe.expected_behaviors,
        model=model, timeout_s=timeout_s,
    )
    run_status = str(run_meta.get("status", "") or "")
    run_patch_source = str(run_meta.get("patch_source", "") or "")
    run_patch_len = len(str(run_meta.get("patch", "") or ""))
    run_note = (
        f"runner_status={run_status}; "
        f"runner_patch_source={run_patch_source}; "
        f"runner_patch_len={run_patch_len}"
    )
    if overall_notes:
        overall_notes = f"{overall_notes} | {run_note}"
    else:
        overall_notes = run_note

    return ProbeResult(
        probe_id=probe.id, task=probe.task, response=response,
        behavior_reviews=behavior_reviews, proposed_edits=proposed_edits,
        overall_notes=overall_notes,
    )
