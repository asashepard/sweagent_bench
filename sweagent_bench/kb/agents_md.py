"""Deterministic AGENTS.md renderer from RepoKB."""
from __future__ import annotations

from sweagent_bench.kb.schema import RepoKB

AGENTS_MD_CHAR_BUDGET = 3000
MAX_HUB_RULES = 3
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
                importers = parts[2]
                rules.append(
                    f"- `{file}` is a high-impact hub ({in_degree} importers); likely impact surface: {importers}."
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
                classification = parts[2]
                rules.append(f"- `{file}` is a key entry point ({kind}, {classification}).")
        elif in_ep_section and line.startswith("#"):
            break
    return rules[:MAX_ENTRY_RULES]


def _extract_convention_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    for line in kb.conventions.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if "docstring" in content.lower():
                rules.append(f"- Follow {content} for all new functions/classes.")
            elif "type hint" in content.lower():
                rules.append(f"- {content} — maintain this level in new code.")
            elif "linter" in content.lower() or "formatter" in content.lower():
                rules.append(f"- {content} — check compliance before submitting.")
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
    return rules[:2]


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


def _base_workflow_rules() -> list[str]:
    return [
        "- Treat repo facts below as priors; verify with targeted file reads before editing.",
        "- Two-phase workflow: (1) localize and trace dependencies, (2) apply minimal scoped edit.",
        "- Keep first patch limited to one likely module/hub unless evidence requires expansion.",
        "- Run the smallest relevant test/check first, then broaden only if needed.",
    ]


def _guardrail_rules() -> list[str]:
    return [
        "- Avoid speculative behavior changes and broad refactors unless strongly justified by evidence.",
        "- Avoid unrelated file edits; each touched file should be tied to the diagnosed path.",
        "- If multiple plausible fixes exist, run one discriminating command/test before editing.",
    ]


def render_agents_md(kb: RepoKB) -> str:
    """Render a compact, exploration-first AGENTS.md from RepoKB. Deterministic."""
    sections: list[str] = []

    sections.append(f"# AGENTS.md — {kb.repo}\n")

    sections.append("## Operating Mode\n")
    sections.append("\n".join(_base_workflow_rules()))

    hub_rules = _extract_hub_rules(kb)
    ep_rules = _extract_entry_point_rules(kb)
    integration_rules = _extract_integration_rules(kb)
    test_rules = _extract_test_rules(kb)
    conv_rules = _extract_convention_rules(kb)

    if hub_rules or ep_rules or integration_rules or test_rules or conv_rules:
        sections.append("\n## Repo Priors\n")
    if hub_rules:
        sections.append("### High-Impact Hubs")
        sections.append("\n".join(hub_rules))
    if ep_rules:
        sections.append("\n### Entry Points")
        sections.append("\n".join(ep_rules))
    if test_rules:
        sections.append("\n### Validation")
        sections.append("\n".join(test_rules))
    if integration_rules:
        sections.append("\n### Integration Risk")
        sections.append("\n".join(integration_rules))
    if conv_rules:
        sections.append("\n### Conventions")
        sections.append("\n".join(conv_rules))

    sections.append("\n## Guardrails\n")
    sections.append("\n".join(_guardrail_rules()))

    result = "\n".join(sections)

    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = result[:AGENTS_MD_CHAR_BUDGET - 20] + "\n\n[... truncated]"

    return result
