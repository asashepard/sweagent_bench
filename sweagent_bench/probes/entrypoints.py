"""Content-based entry point detection probe."""
from __future__ import annotations

from typing import Any

from sweagent_bench.probes.schema import EntryPoint, EntryPoints

MAX_ENTRY_POINTS = 10

_ROUTE_DECORATORS = frozenset({
    "route", "get", "post", "put", "delete", "patch",
    "api_view", "action",
    "app.route", "app.get", "app.post",
    "router.get", "router.post", "router.put", "router.delete",
    "bp.route", "blueprint.route",
})

_CLI_NAMES = frozenset({
    "ArgumentParser", "argparse",
    "click", "command", "group",
    "typer", "Typer", "app",
})

_APP_NAMES = frozenset({
    "application", "app", "wsgi", "asgi",
})


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _detect_if_main(tree: Any, source: bytes) -> bool:
    for node in tree.root_node.children:
        if node.type == "if_statement":
            cond = node.child_by_field_name("condition")
            if cond:
                text = _node_text(cond, source)
                if "__name__" in text and "__main__" in text:
                    return True
    return False


def _detect_route_decorators(tree: Any, source: bytes) -> list[str]:
    found: list[str] = []
    for node in tree.root_node.children:
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    text = _node_text(child, source).lstrip("@").strip()
                    for pat in _ROUTE_DECORATORS:
                        if pat in text.lower():
                            found.append(text)
                            break
    return found


def _detect_cli_patterns(tree: Any, source: bytes) -> list[str]:
    found: list[str] = []

    def _walk(node: Any) -> None:
        if node.type == "call":
            func_text = ""
            func_node = node.child_by_field_name("function")
            if func_node:
                func_text = _node_text(func_node, source)
            if any(name in func_text for name in _CLI_NAMES):
                found.append(func_text)
        if node.type == "import_from_statement":
            text = _node_text(node, source)
            if any(f"import {name}" in text or f"from {name}" in text
                   for name in ("argparse", "click", "typer")):
                found.append(text.strip())
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return found


def _detect_app_assignments(tree: Any, source: bytes) -> list[str]:
    found: list[str] = []
    for node in tree.root_node.children:
        if node.type == "expression_statement":
            for child in node.children:
                if child.type == "assignment":
                    left = child.child_by_field_name("left")
                    right = child.child_by_field_name("right")
                    if left and left.type == "identifier":
                        name = _node_text(left, source).lower()
                        if name in _APP_NAMES and right:
                            rhs = _node_text(right, source)
                            if any(fw in rhs for fw in (
                                "Flask", "FastAPI", "Django", "Starlette",
                                "Sanic", "Quart", "Falcon",
                                "get_wsgi_application", "get_asgi_application",
                            )):
                                found.append(f"{_node_text(left, source)} = {rhs}")
    return found


def detect_entry_points(
    trees: dict[str, Any],
    source_map: dict[str, bytes],
) -> EntryPoints:
    entries: list[EntryPoint] = []

    for rel_path, tree in sorted(trees.items()):
        source = source_map.get(rel_path, b"")

        if _detect_if_main(tree, source):
            classification = "tooling" if any(
                p in rel_path for p in ("script", "manage", "cli", "tool", "bin")
            ) else "runtime"
            entries.append(EntryPoint(
                file=rel_path, kind="if_main", classification=classification,
                confidence="high", detail='if __name__ == "__main__"',
            ))

        routes = _detect_route_decorators(tree, source)
        if routes:
            entries.append(EntryPoint(
                file=rel_path, kind="route", classification="runtime",
                confidence="high", detail=routes[0],
            ))

        cli = _detect_cli_patterns(tree, source)
        if cli:
            entries.append(EntryPoint(
                file=rel_path, kind="cli", classification="tooling",
                confidence="medium" if len(cli) == 1 else "high", detail=cli[0],
            ))

        apps = _detect_app_assignments(tree, source)
        if apps:
            entries.append(EntryPoint(
                file=rel_path, kind="asgi_wsgi", classification="runtime",
                confidence="high", detail=apps[0],
            ))

        if len(entries) >= MAX_ENTRY_POINTS:
            break

    seen_files: set[str] = set()
    deduped: list[EntryPoint] = []
    for ep in entries:
        if ep.file not in seen_files:
            seen_files.add(ep.file)
            deduped.append(ep)

    return EntryPoints(entries=deduped[:MAX_ENTRY_POINTS])
