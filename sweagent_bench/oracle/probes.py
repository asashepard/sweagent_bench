"""LLM-generated probe generation for continuous AGENTS.md tuning."""
from __future__ import annotations

import hashlib
import json
import re

from sweagent_bench.kb.schema import RepoKB
from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.oracle.schema import Probe

MAX_PROBES_PER_ROUND = 10

_PROBE_SYSTEM = """\
Generate probe tasks to evaluate whether AGENTS.md improves repo self-exploration.

Return ONLY a JSON array of probe objects with this shape:
[
    {{
    "task": "short realistic coding-assistant user request",
    "expected_behaviors": ["behavior 1", "behavior 2"],
    "rationale": "why this probe is useful"
    }}
]

Rules:
- Return exactly {max_probes} probes, diverse across bug-fix and test-failure tasks.
- Tasks must be concrete coding requests, not AGENTS/meta questions.
- Each probe must include 2-4 expected behaviors.
- Expected behaviors must emphasize: evidence-first localization, dependency tracing, minimal scoped edits, and targeted validation.
- Every task must be patchable and executable by a tool-using coding runner (inspect files, run commands, propose code diff).
- Avoid pure advisory/navigation-only tasks that do not naturally end in a code diff.
- Avoid duplicates with prior tasks.
- Exactly {max_probes} probes.
"""

_PROBE_USER = """\
Repository: {repo} @ {commit}

CURRENT AGENTS.MD:
---
{agents_md}
---

REPO KB SNIPPET:
---
{kb_text}
---

PRIOR PROBE TASKS (avoid duplicates):
{prior_tasks}

Generate probes now.

Additional requirements for this batch:
- Include realistic technical context (module/function/test hints) when possible.
- Prefer scoped fixes; avoid broad refactors.
- Phrase each task as a request to fix a concrete failing behavior or test regression."""


def _make_probe_id(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:10]


def _fallback_probe_task_pool(repo: str) -> list[str]:
    return [
        f"A regression in {repo} breaks a high-traffic request path; localize the root cause and submit a minimal patch with targeted validation.",
        f"A recent change in {repo} introduced a failing test in core behavior; identify the exact break and provide a focused fix diff.",
        f"A bug in {repo} causes incorrect output under edge inputs; trace dependency flow and patch only the root condition.",
        f"A command-line entry flow in {repo} now behaves incorrectly after refactor; isolate the failing branch and fix with minimal scope.",
        f"An import or initialization sequence in {repo} regressed; find the offending code path and submit a narrowly scoped patch.",
        f"A configuration handling path in {repo} is now inconsistent; reproduce with targeted checks and apply a precise fix.",
        f"A serialization/parsing path in {repo} regressed for specific input shape; localize the parser boundary and patch minimally.",
        f"A validation guard in {repo} became too strict or too permissive; trace call sites and repair the guard condition only.",
        f"A routing/dispatch decision in {repo} selects the wrong handler in a corner case; identify and fix the selector logic.",
        f"A caching/state path in {repo} returns stale or invalid results; pinpoint invalidation logic and apply a focused correction.",
        f"A version-compatibility branch in {repo} regressed after recent updates; localize conditional logic and patch minimally.",
        f"A file/path normalization utility in {repo} now mishandles edge paths; reproduce quickly and fix only normalization behavior.",
        f"An error-handling path in {repo} now swallows or misreports critical exceptions; patch the smallest relevant try/except logic.",
        f"A lifecycle hook sequence in {repo} runs in the wrong order and causes a test failure; correct ordering with minimal edits.",
        f"A dedup/filter step in {repo} is now too aggressive; locate comparison logic and restore intended behavior with targeted changes.",
    ]


def _fallback_probes(kb: RepoKB, limit: int) -> list[Probe]:
    tasks = _fallback_probe_task_pool(kb.repo)
    probes: list[Probe] = []
    for task in tasks[:limit]:
        probes.append(Probe(
            id=_make_probe_id(task), task=task,
            expected_behaviors=[
                "Localizes likely files/functions before editing",
                "Applies a minimal scoped code change",
                "Runs targeted validation relevant to the change",
            ],
            rationale="Fallback probe due to generation parse/error.",
        ))
    return probes


def generate_probes(
    kb: RepoKB, model: str, agents_md: str, *,
    prior_probes: list[Probe] | None = None,
    timeout_s: int = 120, max_probes: int = MAX_PROBES_PER_ROUND,
) -> list[Probe]:
    prior_probes = prior_probes or []
    prior_tasks = "\n".join(f"- {p.task}" for p in prior_probes)
    if not prior_tasks:
        prior_tasks = "- (none)"

    kb_text = kb.render_truncated(char_budget=12_000)
    messages = [
        {"role": "system", "content": _PROBE_SYSTEM.format(max_probes=max_probes)},
        {"role": "user", "content": _PROBE_USER.format(
            repo=kb.repo, commit=kb.commit, agents_md=agents_md,
            kb_text=kb_text, prior_tasks=prior_tasks,
        )},
    ]

    raw = chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=2048, timeout_s=timeout_s)

    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return _fallback_probes(kb, max_probes)
        try:
            arr = json.loads(match.group())
        except json.JSONDecodeError:
            return _fallback_probes(kb, max_probes)

    if not isinstance(arr, list):
        return _fallback_probes(kb, max_probes)

    seen_tasks = {p.task.strip() for p in prior_probes}
    probes: list[Probe] = []
    for item in arr:
        if len(probes) >= max_probes:
            break
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        if not task or task in seen_tasks:
            continue
        expected_behaviors = item.get("expected_behaviors", [])
        if not isinstance(expected_behaviors, list):
            expected_behaviors = []
        expected_behaviors = [str(b).strip() for b in expected_behaviors if str(b).strip()]
        if not expected_behaviors:
            expected_behaviors = [
                "Response reflects AGENTS.md repository guidance",
                "Response includes actionable implementation/testing advice",
            ]
        rationale = str(item.get("rationale", "")).strip()
        probes.append(Probe(
            id=_make_probe_id(task), task=task,
            expected_behaviors=expected_behaviors[:4], rationale=rationale,
        ))
        seen_tasks.add(task)

    if len(probes) < max_probes:
        for task in _fallback_probe_task_pool(kb.repo):
            task = task.strip()
            if not task or task in seen_tasks:
                continue
            probes.append(Probe(
                id=_make_probe_id(task),
                task=task,
                expected_behaviors=[
                    "Localizes likely files/functions before editing",
                    "Applies a minimal scoped code change",
                    "Runs targeted validation relevant to the change",
                ],
                rationale="Fallback top-up to enforce fixed probe count.",
            ))
            seen_tasks.add(task)
            if len(probes) >= max_probes:
                break

    if not probes:
        return _fallback_probes(kb, max_probes)

    return probes[:max_probes]
