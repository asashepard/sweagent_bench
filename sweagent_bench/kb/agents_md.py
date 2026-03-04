"""Deterministic AGENTS.md renderer from RepoKB."""
from __future__ import annotations

from sweagent_bench.kb.schema import RepoKB

AGENTS_MD_CHAR_BUDGET = 3200


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
                    f"- Hub file `{file}` ({in_degree} importers). "
                    f"Changes here affect: {importers}. "
                    f"Run full test suite after editing."
                )
    return rules[:6]


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
                rules.append(f"- Entry point: `{file}` ({kind}, {classification})")
        elif in_ep_section and line.startswith("#"):
            break
    return rules[:5]


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
    return rules[:5]


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
    return rules[:3]


def _extract_cluster_rules(kb: RepoKB) -> list[str]:
    rules: list[str] = []
    for line in kb.context.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "integration" in stripped.lower():
            rules.append(stripped)
        elif "bridge" in stripped.lower() or "broad impact" in stripped.lower():
            continue
        elif stripped.startswith("- ") and "cluster" in stripped.lower():
            rules.append(stripped)
    return rules[:3]


def render_agents_md(kb: RepoKB) -> str:
    """Render an actionable AGENTS.md from the RepoKB. Deterministic."""
    sections: list[str] = []

    sections.append(f"# AGENTS.md — {kb.repo}\n")

    hub_rules = _extract_hub_rules(kb)
    if hub_rules:
        sections.append("## Hub Safety\n")
        sections.append("\n".join(hub_rules))

    ep_rules = _extract_entry_point_rules(kb)
    if ep_rules:
        sections.append("\n## Entry Points\n")
        sections.append("\n".join(ep_rules))

    conv_rules = _extract_convention_rules(kb)
    if conv_rules:
        sections.append("\n## Conventions\n")
        sections.append("\n".join(conv_rules))

    test_rules = _extract_test_rules(kb)
    if test_rules:
        sections.append("\n## Testing\n")
        sections.append("\n".join(test_rules))

    cluster_rules = _extract_cluster_rules(kb)
    if cluster_rules:
        sections.append("\n## Integration\n")
        sections.append("\n".join(cluster_rules))

    result = "\n".join(sections)

    if len(result) > AGENTS_MD_CHAR_BUDGET:
        result = result[:AGENTS_MD_CHAR_BUDGET - 20] + "\n\n[... truncated]"

    return result
