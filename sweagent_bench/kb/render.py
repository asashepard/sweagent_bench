"""Section renderers for the RepoKB artifact."""
from __future__ import annotations

from sweagent_bench.probes.schema import (
    ClusterResults,
    Conventions,
    EntryPoints,
    ImportGraph,
    SymbolIndex,
    TestInfo,
)

ARCHITECTURE_LINE_BUDGET = 200
SYMBOL_MAP_LINE_BUDGET = 300
CONTEXT_LINE_BUDGET = 200


def _cap_lines(text: str, budget: int) -> str:
    lines = text.splitlines()
    if len(lines) <= budget:
        return text
    return "\n".join(lines[:budget]) + "\n[... truncated]"


def render_architecture(graph: ImportGraph, entries: EntryPoints) -> str:
    parts: list[str] = []

    if graph.hubs:
        parts.append("### Hub Modules (highest in-degree)\n")
        parts.append("| File | In-degree | Top Importers | Key Exports |")
        parts.append("|------|-----------|---------------|-------------|")
        for hub in graph.hubs[:12]:
            importers_str = ", ".join(hub.importers[:3])
            exports_str = ", ".join(hub.exports[:3])
            parts.append(
                f"| {hub.file} | {hub.in_degree} "
                f"| {importers_str} | {exports_str} |"
            )
        parts.append("")

    if entries.entries:
        parts.append("### Entry Points\n")
        parts.append("| File | Kind | Classification | Confidence | Detail |")
        parts.append("|------|------|----------------|------------|--------|")
        for ep in entries.entries[:10]:
            detail = ep.detail[:60] if ep.detail else ""
            parts.append(
                f"| {ep.file} | {ep.kind} | {ep.classification} "
                f"| {ep.confidence} | {detail} |"
            )
        parts.append("")

    return _cap_lines("\n".join(parts), ARCHITECTURE_LINE_BUDGET)


def render_symbol_map(index: SymbolIndex) -> str:
    parts: list[str] = []
    file_count = 0

    for rel_path in sorted(index.by_file.keys()):
        entries = index.by_file[rel_path]
        if not entries:
            continue
        file_count += 1
        if file_count > 30:
            parts.append(f"\n[... {len(index.by_file) - 30} more files omitted]")
            break

        parts.append(f"### `{rel_path}`\n")
        parts.append("| Symbol | Kind | Signature | Used In |")
        parts.append("|--------|------|-----------|---------|")
        for entry in entries[:10]:
            sig = entry.signature
            if len(sig) > 80:
                sig = sig[:77] + "..."
            used = ", ".join(entry.used_in[:5])
            if len(entry.used_in) > 5:
                used += f" (+{len(entry.used_in) - 5})"
            parts.append(f"| {entry.name} | {entry.kind} | `{sig}` | {used} |")
        parts.append("")

    return _cap_lines("\n".join(parts), SYMBOL_MAP_LINE_BUDGET)


def render_context(clusters: ClusterResults, tests: TestInfo) -> str:
    parts: list[str] = []

    if clusters.clusters:
        parts.append("### Co-Import Clusters\n")
        parts.append(
            "Files clustered by shared importers "
            "(score = number of files that import both).\n"
        )
        for cl in clusters.clusters[:6]:
            files_str = ", ".join(cl.files[:8])
            if len(cl.files) > 8:
                files_str += f" (+{len(cl.files) - 8})"
            importers_str = ", ".join(cl.shared_importers[:4])
            parts.append(f"- **Cluster {cl.id}** (score {cl.score:.1f}): {files_str}")
            if importers_str:
                parts.append(f"  Shared importers: {importers_str}")
        parts.append("")

    if clusters.chains:
        parts.append("### Import Chains\n")
        for chain in clusters.chains[:8]:
            chain_str = " → ".join(chain.files)
            parts.append(f"- {chain_str}")
        parts.append("")

    if clusters.integrations:
        parts.append("### Integration Points\n")
        parts.append(
            "Files that bridge multiple clusters (editing these has broad impact).\n"
        )
        for f in clusters.integrations[:4]:
            parts.append(f"- {f}")
        parts.append("")

    parts.append("### Test Infrastructure\n")
    parts.append(f"- Test command: `{tests.test_command}`")
    if tests.test_dirs:
        parts.append(f"- Test directories: {', '.join(tests.test_dirs[:6])}")
    if tests.conftest_paths:
        parts.append(f"- Conftest files: {', '.join(tests.conftest_paths[:6])}")
    if getattr(tests, 'fixtures', None):
        parts.append(f"- Key fixtures: {', '.join(tests.fixtures[:10])}")
    parts.append("")

    return _cap_lines("\n".join(parts), CONTEXT_LINE_BUDGET)


def render_conventions(conv: Conventions) -> str:
    parts: list[str] = []

    if conv.docstring_style and conv.docstring_style != "unknown":
        parts.append(f"- Docstring style: **{conv.docstring_style}**")
    if conv.type_hint_prevalence > 0:
        parts.append(f"- Type hint coverage: {conv.type_hint_prevalence:.0%} of functions")
    if conv.import_style:
        parts.append(f"- Import style: {conv.import_style}")
    if conv.linter_configs:
        parts.append(f"- Linter/formatter configs: {', '.join(conv.linter_configs)}")
    for pattern in conv.detected_patterns:
        if pattern not in "\n".join(parts):
            parts.append(f"- {pattern}")

    return "\n".join(parts)
