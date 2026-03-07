"""Deterministic AGENTS.md renderer from RepoKB."""
from __future__ import annotations

from sweagent_bench.kb.schema import RepoKB

AGENTS_MD_CHAR_BUDGET = 3000
MAX_HUB_RULES = 2
MAX_ENTRY_RULES = 2
MAX_INTEGRATION_RULES = 2


def _extract_hub_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    for line in kb.architecture.splitlines():
        if line.startswith("|") and "File" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                file = parts[0]
                in_degree = parts[1]
                rules.append(
                    f"- `{file}` — hub ({in_degree} importers)."
                )
    return rules[:MAX_HUB_RULES]


def _extract_entry_point_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    in_ep_section = False
    for line in kb.architecture.splitlines():
        if "Entry Points" in line:
            in_ep_section = True
            continue
        if in_ep_section and line.startswith("|") and "File" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 4:
                file = parts[0]
                kind = parts[1]
                rules.append(f"- `{file}` ({kind}).")
        elif in_ep_section and line.startswith("#"):
            break
    return rules[:MAX_ENTRY_RULES]


def _extract_convention_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    seen_lower: set[str] = set()
    for line in kb.conventions.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            # Deduplicate near-identical convention bullets
            norm = content.lower()
            if norm in seen_lower:
                continue
            seen_lower.add(norm)
            if "docstring" in norm:
                rules.append(f"- Docstring: {content}.")
            elif "type hint" in norm:
                rules.append(f"- {content}.")
            elif "linter" in norm or "formatter" in norm:
                rules.append(f"- {content}.")
            else:
                rules.append(f"- {content}")
    return rules[:3]


def _extract_test_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    for line in kb.context.splitlines():
        stripped = line.strip()
        if "test command" in stripped.lower():
            cmd = stripped.split("`")
            if len(cmd) >= 2:
                rules.append(f"- Run `{cmd[1]}` to verify your changes.")
        elif "test director" in stripped.lower():
            rules.append(f"- {stripped.lstrip('- ')}")
        elif "conftest" in stripped.lower():
            rules.append(f"- {stripped.lstrip('- ')}")
        elif "fixture" in stripped.lower():
            rules.append(f"- {stripped.lstrip('- ')}")
    return rules[:3]


def _extract_integration_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    in_integration = False
    for line in kb.context.splitlines():
        stripped = line.strip()
        if stripped.startswith("### Integration Points"):
            in_integration = True
            continue
        if in_integration and stripped.startswith("### "):
            break
        if in_integration and stripped.startswith("- "):
            rules.append(f"- {stripped[2:]}")
    return rules[:MAX_INTEGRATION_RULES]


def _extract_import_chain_rules(kb: RepoKB) -> list[str]:
    """Extract import chain rules from the context section."""
    rules: list[str] = []
    in_chains = False
    for line in kb.context.splitlines():
        stripped = line.strip()
        if stripped.startswith("### Import Chains"):
            in_chains = True
            continue
        if in_chains and stripped.startswith("### "):
            break
        if in_chains and stripped.startswith("- "):
            rules.append(f"- Chain: {stripped[2:]}")
    return rules[:2]


def _base_workflow_rules() -> list[str]:
    return [
        "- Verify repo priors with targeted reads before editing.",
        "- Localize, trace deps, then apply minimal scoped edit.",
        "- Run the smallest relevant test first, broaden only if needed.",
    ]


def _guardrail_rules() -> list[str]:
    return [
        "- No speculative changes or broad refactors without evidence.",
        "- Every touched file must tie to the diagnosed path.",
    ]


def _procedural_scaffold_rules() -> list[str]:
    """Generic, repo-agnostic procedural standards.

    These rules form a stronger fixed baseline for the static_kb control.
    They are NOT issue-specific, benchmark-specific, or oracle-derived.
    The oracle_tuned condition is free to exceed or refine these.
    """
    return [
        "- Reproduce the failure before editing when possible.",
        "- Read target files and nearby callers before patching.",
        "- Keep first patch minimal; inspect call sites if public API changes.",
        "- Require evidence from file reads or command output — no fabricated edits.",
        "- Patches must be syntactically complete; remove unused imports.",
    ]


def render_agents_md(kb: RepoKB) -> str:
    """Render a compact, exploration-first AGENTS.md from RepoKB. Deterministic."""
    lines: list[str] = []

    lines.append(f"# AGENTS.md — {kb.repo}")
    lines.append("## Operating Mode")
    lines.extend(_base_workflow_rules())

    lines.append("## Procedural Standards")
    lines.extend(_procedural_scaffold_rules())

    hub_rules = _extract_hub_rules(kb)
    ep_rules = _extract_entry_point_rules(kb)
    integration_rules = _extract_integration_rules(kb)
    test_rules = _extract_test_rules(kb)
    conv_rules = _extract_convention_rules(kb)

    chain_rules = _extract_import_chain_rules(kb)

    has_priors = hub_rules or ep_rules or integration_rules or test_rules or conv_rules or chain_rules
    if has_priors:
        lines.append("## Repo Priors")
    if hub_rules:
        lines.append("### High-Impact Hubs")
        lines.extend(hub_rules)
    if ep_rules:
        lines.append("### Entry Points")
        lines.extend(ep_rules)
    if chain_rules:
        lines.append("### Import Chains")
        lines.extend(chain_rules)
    if test_rules:
        lines.append("### Validation")
        lines.extend(test_rules)
    if integration_rules:
        lines.append("### Integration Risk")
        lines.extend(integration_rules)
    if conv_rules:
        lines.append("### Conventions")
        lines.extend(conv_rules)

    lines.append("## Guardrails")
    lines.extend(_guardrail_rules())

    result = "\n".join(lines) + "\n"

    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = result[:AGENTS_MD_CHAR_BUDGET - 20] + "\n[... truncated]"

    return result
