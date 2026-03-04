"""Test infrastructure discovery probe."""
from __future__ import annotations

import os
from pathlib import Path

from sweagent_bench.probes.schema import TestInfo
from sweagent_bench.utils.ignore import should_ignore_dir

_TEST_DIR_NAMES = {"tests", "test", "testing", "spec", "specs"}


def _find_test_dirs(repo_dir: Path) -> list[str]:
    found: list[str] = []
    for root, dirs, _files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        rel = Path(root).relative_to(repo_dir)
        depth = len(rel.parts)
        if depth > 3:
            dirs.clear()
            continue
        for d in dirs:
            if d.lower() in _TEST_DIR_NAMES:
                found.append(str(rel / d).replace("\\", "/"))
    return sorted(found)


def _find_conftest_files(repo_dir: Path) -> list[str]:
    found: list[str] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]
        rel = Path(root).relative_to(repo_dir)
        if len(rel.parts) > 4:
            dirs.clear()
            continue
        if "conftest.py" in files:
            found.append(str(rel / "conftest.py").replace("\\", "/"))
    return sorted(found)


def _detect_test_command(repo_dir: Path) -> str:
    if (repo_dir / "pytest.ini").exists():
        return "pytest"
    if (repo_dir / "pyproject.toml").exists():
        try:
            text = (repo_dir / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
            if "[tool.pytest" in text:
                return "pytest"
        except OSError:
            pass
    if (repo_dir / "setup.cfg").exists():
        try:
            text = (repo_dir / "setup.cfg").read_text(encoding="utf-8", errors="ignore")
            if "[tool:pytest]" in text:
                return "pytest"
        except OSError:
            pass
    if (repo_dir / "tox.ini").exists():
        return "tox"
    return "pytest"


def _detect_fixtures(repo_dir: Path, conftest_paths: list[str]) -> list[str]:
    fixtures: list[str] = []
    for rel in conftest_paths:
        full = repo_dir / rel
        if not full.exists():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("@pytest.fixture") or stripped.startswith("@fixture"):
                continue
            if stripped.startswith("def ") and "()" not in stripped:
                name = stripped.split("(")[0].replace("def ", "").strip()
                if name and not name.startswith("_"):
                    fixtures.append(name)
    return sorted(set(fixtures))[:20]


def detect_tests(repo_dir: Path) -> TestInfo:
    test_dirs = _find_test_dirs(repo_dir)
    conftest_paths = _find_conftest_files(repo_dir)
    test_command = _detect_test_command(repo_dir)
    fixtures = _detect_fixtures(repo_dir, conftest_paths)

    return TestInfo(
        test_command=test_command,
        test_dirs=test_dirs,
        conftest_paths=conftest_paths,
        fixtures=fixtures,
    )
