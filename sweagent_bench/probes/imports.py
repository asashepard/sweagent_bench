"""Import/dependency graph probe."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sweagent_bench.probes.schema import HubModule, ImportGraph

MAX_HUBS = 12
MAX_HUB_DETAILS = 3


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_imports_from_tree(tree: Any, source: bytes) -> list[str]:
    imports: list[str] = []

    def _walk(node: Any) -> None:
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(_node_text(child, source))
                elif child.type == "aliased_import":
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            imports.append(_node_text(sub, source))
                            break
        elif node.type == "import_from_statement":
            for child in node.children:
                if child.type in ("dotted_name", "relative_import"):
                    imports.append(_node_text(child, source))
                    break
        else:
            for child in node.children:
                _walk(child)

    _walk(tree.root_node)
    return imports


def _resolve_module_to_file(module: str, repo_files: set[str]) -> str | None:
    parts = module.lstrip(".").split(".")
    pkg_path = "/".join(parts) + "/__init__.py"
    if pkg_path in repo_files:
        return pkg_path
    mod_path = "/".join(parts) + ".py"
    if mod_path in repo_files:
        return mod_path
    if len(parts) > 1:
        parent_mod = "/".join(parts[:-1]) + ".py"
        if parent_mod in repo_files:
            return parent_mod
        parent_pkg = "/".join(parts[:-1]) + "/__init__.py"
        if parent_pkg in repo_files:
            return parent_pkg
    return None


def build_import_graph(trees: dict[str, Any], source_map: dict[str, bytes]) -> ImportGraph:
    repo_files = set(trees.keys())
    edges: dict[str, list[str]] = {}
    imported_by: dict[str, list[str]] = {f: [] for f in repo_files}

    for rel_path, tree in trees.items():
        source = source_map.get(rel_path, b"")
        raw_imports = _extract_imports_from_tree(tree, source)
        resolved: list[str] = []
        for mod in raw_imports:
            target = _resolve_module_to_file(mod, repo_files)
            if target and target != rel_path:
                resolved.append(target)
        seen: set[str] = set()
        deduped: list[str] = []
        for t in resolved:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        edges[rel_path] = deduped
        for t in deduped:
            imported_by.setdefault(t, []).append(rel_path)

    hub_candidates = sorted(
        imported_by.items(), key=lambda kv: len(kv[1]), reverse=True,
    )

    hubs: list[HubModule] = []
    for file, importers in hub_candidates[:MAX_HUBS]:
        if not importers:
            break
        exports: list[str] = []
        tree = trees.get(file)
        source = source_map.get(file, b"")
        if tree:
            for child in tree.root_node.children:
                if child.type in ("function_definition", "class_definition"):
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        exports.append(_node_text(name_node, source))
        hubs.append(HubModule(
            file=file,
            importers=importers[:MAX_HUB_DETAILS],
            exports=exports[:MAX_HUB_DETAILS],
            in_degree=len(importers),
        ))

    return ImportGraph(edges=edges, imported_by=imported_by, hubs=hubs)
