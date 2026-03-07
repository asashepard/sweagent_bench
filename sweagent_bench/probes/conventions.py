"""Coding convention detection probe."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sweagent_bench.probes.schema import Conventions
from sweagent_bench.utils.ignore import should_ignore_dir, should_ignore_file

MAX_DIRECTORIES = 12

_LINTER_CONFIG_FILES = frozenset({
    ".flake8", ".pylintrc", "mypy.ini", ".mypy.ini",
    "ruff.toml", ".ruff.toml",
    ".isort.cfg", ".bandit",
    ".pre-commit-config.yaml",
})

_PYPROJECT_TOOL_SECTIONS = frozenset({
    "tool.ruff", "tool.mypy", "tool.pylint", "tool.isort",
    "tool.black", "tool.flake8", "tool.pytest", "tool.coverage",
})


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _detect_docstring_style(trees: dict[str, Any], source_map: dict[str, bytes]) -> str:
    google_count = 0
    numpy_count = 0
    sphinx_count = 0
    total = 0

    for rel_path, tree in list(trees.items())[:30]:
        source = source_map.get(rel_path, b"")
        for node in tree.root_node.children:
            if node.type in ("function_definition", "class_definition", "decorated_definition"):
                body = node.child_by_field_name("body")
                if body is None:
                    for child in node.children:
                        if child.type in ("function_definition", "class_definition"):
                            body = child.child_by_field_name("body")
                            break
                if body is None:
                    continue
                for child in body.children:
                    if child.type == "expression_statement":
                        for sub in child.children:
                            if sub.type == "string":
                                text = _node_text(sub, source)
                                total += 1
                                if "Args:" in text or "Returns:" in text:
                                    google_count += 1
                                elif "Parameters\n" in text or "----------" in text:
                                    numpy_count += 1
                                elif ":param " in text or ":type " in text:
                                    sphinx_count += 1
                        break

    if total == 0:
        return "unknown"
    counts = {"google": google_count, "numpy": numpy_count, "sphinx": sphinx_count}
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[best] == 0:
        return "unknown"
    return best


def _detect_type_hint_prevalence(trees: dict[str, Any], source_map: dict[str, bytes]) -> float:
    annotated = 0
    total = 0

    for rel_path, tree in list(trees.items())[:30]:
        source = source_map.get(rel_path, b"")
        for node in tree.root_node.children:
            func = None
            if node.type == "function_definition":
                func = node
            elif node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "function_definition":
                        func = child
                        break
            if func is None:
                continue
            total += 1
            ret = func.child_by_field_name("return_type")
            params = func.child_by_field_name("parameters")
            has_annotation = ret is not None
            if not has_annotation and params:
                for p in params.children:
                    if p.type in ("typed_parameter", "typed_default_parameter"):
                        has_annotation = True
                        break
            if has_annotation:
                annotated += 1

    return annotated / total if total > 0 else 0.0


def _detect_import_style(trees: dict[str, Any], source_map: dict[str, bytes]) -> str:
    future_first_count = 0
    samples = 0

    for rel_path, tree in list(trees.items())[:20]:
        source = source_map.get(rel_path, b"")
        imports = []
        for node in tree.root_node.children:
            if node.type in ("import_statement", "import_from_statement"):
                text = _node_text(node, source)
                imports.append(text)
        if not imports:
            continue
        samples += 1
        if imports[0].startswith("from __future__"):
            future_first_count += 1

    parts = []
    if samples > 0 and future_first_count > samples * 0.5:
        parts.append("__future__ imports first")
    if parts:
        return "; ".join(parts)
    return "standard"


def _detect_linter_configs(repo_dir: Path) -> list[str]:
    found: list[str] = []
    try:
        for item in sorted(repo_dir.iterdir()):
            if item.is_file() and item.name in _LINTER_CONFIG_FILES:
                found.append(item.name)
    except PermissionError:
        pass

    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            for section in _PYPROJECT_TOOL_SECTIONS:
                if f"[{section}]" in text:
                    found.append(f"pyproject.toml:[{section}]")
        except OSError:
            pass

    setup_cfg = repo_dir / "setup.cfg"
    if setup_cfg.exists():
        try:
            text = setup_cfg.read_text(encoding="utf-8", errors="ignore")
            for marker in ("[tool:pytest]", "[flake8]", "[isort]", "[mypy]"):
                if marker in text:
                    found.append(f"setup.cfg:{marker}")
        except OSError:
            pass

    return sorted(found)


def detect_conventions(
    repo_dir: Path,
    trees: dict[str, Any],
    source_map: dict[str, bytes],
) -> Conventions:
    docstring_style = _detect_docstring_style(trees, source_map)
    type_hint_prevalence = _detect_type_hint_prevalence(trees, source_map)
    import_style = _detect_import_style(trees, source_map)
    linter_configs = _detect_linter_configs(repo_dir)

    patterns: list[str] = []
    if docstring_style != "unknown":
        patterns.append(f"Docstring style: {docstring_style}")
    if type_hint_prevalence > 0.5:
        patterns.append(f"Type hints widely used ({type_hint_prevalence:.0%} of functions)")
    elif type_hint_prevalence > 0.1:
        patterns.append(f"Type hints partially used ({type_hint_prevalence:.0%} of functions)")
    if linter_configs:
        patterns.append(f"Linters/formatters: {', '.join(linter_configs)}")

    return Conventions(
        docstring_style=docstring_style,
        type_hint_prevalence=type_hint_prevalence,
        import_style=import_style,
        linter_configs=linter_configs,
        detected_patterns=patterns,
    )
