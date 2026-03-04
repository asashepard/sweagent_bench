"""Tree-sitter based static analysis probes for repository introspection.

Public API::

    from sweagent_bench.probes import run_all_probes

    results = run_all_probes(repo_dir)
"""
from __future__ import annotations

from pathlib import Path

from sweagent_bench.probes.clustering import build_clusters
from sweagent_bench.probes.conventions import detect_conventions
from sweagent_bench.probes.entrypoints import detect_entry_points
from sweagent_bench.probes.imports import build_import_graph
from sweagent_bench.probes.parser import parse_repo
from sweagent_bench.probes.schema import ProbeResults
from sweagent_bench.probes.symbols import build_symbol_index
from sweagent_bench.probes.tests import detect_tests


def run_all_probes(repo_dir: Path) -> ProbeResults:
    """Run all deterministic probes against a checked-out repository."""
    trees = parse_repo(repo_dir)

    source_map: dict[str, bytes] = {}
    for rel_path in trees:
        full = repo_dir / rel_path
        try:
            source_map[rel_path] = full.read_bytes()
        except OSError:
            source_map[rel_path] = b""

    import_graph = build_import_graph(trees, source_map)
    symbol_index = build_symbol_index(trees, source_map, import_graph)
    entry_points = detect_entry_points(trees, source_map)
    clusters = build_clusters(import_graph)
    test_info = detect_tests(repo_dir)
    conventions = detect_conventions(repo_dir, trees, source_map)

    return ProbeResults(
        repo_dir=str(repo_dir),
        imports=import_graph,
        symbols=symbol_index,
        entry_points=entry_points,
        clusters=clusters,
        tests=test_info,
        conventions=conventions,
    )


__all__ = ["run_all_probes", "ProbeResults"]
