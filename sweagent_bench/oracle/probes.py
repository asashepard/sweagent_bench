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
You are generating probe tasks to stress-test and improve AGENTS.md guidance
for a coding assistant working on a repository.

Return ONLY a JSON array of probe objects with this shape:
[
    {{
    "task": "short realistic coding-assistant user request",
    "expected_behaviors": ["behavior 1", "behavior 2"],
    "rationale": "why this probe is useful"
    }}
]

Rules:
- Generate diverse probes (architecture, testing, code navigation, change safety).
- Prefer concrete repo-aware tasks over generic style prompts.
- Avoid duplicates with prior probe tasks.
- Provide 2-4 expected behaviors per probe.
- Maximum {max_probes} probes.
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

Generate probes now."""


def _make_probe_id(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:10]


def _fallback_probes(kb: RepoKB, limit: int) -> list[Probe]:
    tasks = [
        f"I need to make a risky change in {kb.repo}. What files and tests should I check first?",
        f"I am adding a new feature in {kb.repo}. What repo-specific conventions should I follow?",
        f"I changed a core module in {kb.repo}. How should I validate blast radius before submission?",
    ]
    probes: list[Probe] = []
    for task in tasks[:limit]:
        probes.append(Probe(
            id=_make_probe_id(task), task=task,
            expected_behaviors=[
                "Identifies impacted files or integration points",
                "Recommends concrete validation/test steps",
                "Uses repository-specific guidance from AGENTS.md",
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

    raw = chat_completion(model=model, messages=messages, temperature=0.7, max_tokens=2048, timeout_s=timeout_s)

    text = raw.strip()
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

    if not probes:
        return _fallback_probes(kb, max_probes)

    return probes
