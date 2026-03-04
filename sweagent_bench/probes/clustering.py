"""Co-import clustering probe."""
from __future__ import annotations

from collections import defaultdict

from sweagent_bench.probes.schema import (
    ClusterResults,
    FileCluster,
    ImportChain,
    ImportGraph,
)

MAX_CLUSTERS = 6
MAX_CHAINS = 8
MAX_INTEGRATIONS = 4


def _compute_co_import_scores(imported_by: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    importer_to_targets: dict[str, set[str]] = defaultdict(set)
    for target, importers in imported_by.items():
        for imp in importers:
            importer_to_targets[imp].add(target)

    scores: dict[tuple[str, str], int] = {}
    for _importer, targets in importer_to_targets.items():
        target_list = sorted(targets)
        for i in range(len(target_list)):
            for j in range(i + 1, len(target_list)):
                pair = (target_list[i], target_list[j])
                scores[pair] = scores.get(pair, 0) + 1

    return {pair: s for pair, s in scores.items() if s >= 2}


def _greedy_cluster(scores: dict[tuple[str, str], int]) -> list[FileCluster]:
    file_to_cluster: dict[str, int] = {}
    clusters: dict[int, set[str]] = {}
    cluster_scores: dict[int, list[int]] = {}
    next_id = 0

    for (a, b), score in sorted(scores.items(), key=lambda x: -x[1]):
        ca = file_to_cluster.get(a)
        cb = file_to_cluster.get(b)

        if ca is not None and cb is not None:
            if ca == cb:
                cluster_scores[ca].append(score)
            continue
        elif ca is not None:
            clusters[ca].add(b)
            file_to_cluster[b] = ca
            cluster_scores[ca].append(score)
        elif cb is not None:
            clusters[cb].add(a)
            file_to_cluster[a] = cb
            cluster_scores[cb].append(score)
        else:
            cid = next_id
            next_id += 1
            clusters[cid] = {a, b}
            file_to_cluster[a] = cid
            file_to_cluster[b] = cid
            cluster_scores[cid] = [score]

    result: list[FileCluster] = []
    for cid, members in sorted(clusters.items()):
        sc = cluster_scores.get(cid, [])
        avg = sum(sc) / len(sc) if sc else 0.0
        result.append(FileCluster(id=cid, files=sorted(members), shared_importers=[], score=avg))

    return result


def _find_shared_importers(cluster: FileCluster, imported_by: dict[str, list[str]]) -> list[str]:
    importer_counts: dict[str, int] = defaultdict(int)
    for f in cluster.files:
        for imp in imported_by.get(f, []):
            importer_counts[imp] += 1
    return sorted(imp for imp, cnt in importer_counts.items() if cnt >= 2)


def _find_import_chains(edges: dict[str, list[str]], max_length: int = 6) -> list[ImportChain]:
    best_chains: list[list[str]] = []

    def _dfs(file: str, chain: list[str], visited: set[str]) -> None:
        targets = edges.get(file, [])
        extended = False
        for t in targets:
            if t not in visited and len(chain) < max_length:
                visited.add(t)
                chain.append(t)
                _dfs(t, chain, visited)
                chain.pop()
                visited.discard(t)
                extended = True
        if not extended and len(chain) >= 3:
            best_chains.append(list(chain))

    for start in sorted(edges.keys()):
        _dfs(start, [start], {start})

    best_chains.sort(key=len, reverse=True)
    seen_starts: set[str] = set()
    unique: list[ImportChain] = []
    for chain in best_chains:
        if chain[0] not in seen_starts:
            seen_starts.add(chain[0])
            unique.append(ImportChain(files=chain, length=len(chain)))
        if len(unique) >= MAX_CHAINS:
            break

    return unique


def _find_integration_points(
    clusters: list[FileCluster], imported_by: dict[str, list[str]],
) -> list[str]:
    file_to_cluster: dict[str, int] = {}
    for cl in clusters:
        for f in cl.files:
            file_to_cluster[f] = cl.id

    bridge_scores: dict[str, int] = defaultdict(int)
    for f, importers in imported_by.items():
        f_cluster = file_to_cluster.get(f)
        for imp in importers:
            imp_cluster = file_to_cluster.get(imp)
            if imp_cluster is not None and f_cluster is not None and imp_cluster != f_cluster:
                bridge_scores[f] += 1

    bridges = sorted(bridge_scores.items(), key=lambda x: -x[1])
    return [f for f, _ in bridges[:MAX_INTEGRATIONS]]


def build_clusters(graph: ImportGraph) -> ClusterResults:
    scores = _compute_co_import_scores(graph.imported_by)
    raw_clusters = _greedy_cluster(scores)

    for cluster in raw_clusters:
        cluster.shared_importers = _find_shared_importers(cluster, graph.imported_by)

    raw_clusters.sort(key=lambda c: -c.score)
    clusters = raw_clusters[:MAX_CLUSTERS]

    chains = _find_import_chains(graph.edges)
    integrations = _find_integration_points(clusters, graph.imported_by)

    return ClusterResults(clusters=clusters, chains=chains, integrations=integrations)
