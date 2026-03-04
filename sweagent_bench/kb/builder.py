"""Build a RepoKB deterministically from probe results."""
from __future__ import annotations

from sweagent_bench.kb.render import (
    render_architecture,
    render_context,
    render_conventions,
    render_symbol_map,
)
from sweagent_bench.kb.schema import RepoKB
from sweagent_bench.probes.schema import ProbeResults


def build_kb(repo: str, commit: str, probe_results: ProbeResults) -> RepoKB:
    """Build the RepoKB artifact from probe results. Fully deterministic."""
    architecture = render_architecture(probe_results.imports, probe_results.entry_points)
    symbol_map = render_symbol_map(probe_results.symbols)
    context = render_context(probe_results.clusters, probe_results.tests)
    conventions = render_conventions(probe_results.conventions)

    return RepoKB(
        repo=repo, commit=commit, version=0,
        architecture=architecture, symbol_map=symbol_map,
        context=context, conventions=conventions,
    )
