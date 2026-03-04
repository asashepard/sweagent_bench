"""Shared ignore patterns for directory traversal."""
from __future__ import annotations

IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".nox", ".eggs",
    ".venv", "venv", "env", ".env",
    "dist", "build", "_build", ".build", "htmlcov",
    "node_modules",
    ".coverage", ".cache",
})

IGNORE_FILES: frozenset[str] = frozenset({
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.so", "*.egg",
})

IGNORE_DIR_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
})


def should_ignore_dir(name: str) -> bool:
    if name in IGNORE_DIRS:
        return True
    for pattern in IGNORE_DIR_PATTERNS:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False


def should_ignore_file(name: str) -> bool:
    if name in IGNORE_FILES:
        return True
    for pattern in IGNORE_FILES:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False
