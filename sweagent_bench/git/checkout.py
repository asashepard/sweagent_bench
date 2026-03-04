"""Git worktree-based repo checkout utilities."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sweagent_bench.utils.paths import REPOS_CACHE_DIR, WORKTREES_DIR, repo_to_dirname


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def _get_head_commit(repo_path: Path) -> str | None:
    code, stdout, _ = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    if code == 0:
        return stdout.strip()
    return None


def _ensure_bare_mirror(repo: str) -> Path:
    mirror_path = REPOS_CACHE_DIR / f"{repo_to_dirname(repo)}.git"

    if mirror_path.exists():
        _run_git(["fetch", "--all"], cwd=mirror_path)
        return mirror_path

    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    code, _, stderr = _run_git(
        ["clone", "--mirror", url, str(mirror_path)]
    )
    if code != 0:
        raise RuntimeError(f"Failed to clone mirror for {repo}: {stderr}")

    return mirror_path


def _ensure_worktree(mirror_path: Path, repo: str, commit: str) -> Path:
    worktree_path = WORKTREES_DIR / repo_to_dirname(repo) / commit

    if worktree_path.exists():
        current_head = _get_head_commit(worktree_path)
        if current_head and current_head.startswith(commit[:7]):
            return worktree_path
        _run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=mirror_path)
        if worktree_path.exists():
            import shutil
            shutil.rmtree(worktree_path)

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    code, _, stderr = _run_git(
        ["worktree", "add", "--detach", str(worktree_path), commit],
        cwd=mirror_path,
    )
    if code != 0:
        raise RuntimeError(f"Failed to create worktree for {repo}@{commit}: {stderr}")

    return worktree_path


def _resolve_commit(mirror_path: Path, commit: str) -> str:
    """Resolve a symbolic commit ref (like HEAD) to a concrete SHA.

    If *commit* is already a 7+ hex-char SHA prefix, return it as-is.
    Otherwise run ``git rev-parse`` in the bare mirror to resolve it.
    """
    # Already a concrete SHA (at least 7 hex chars)?
    if len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit.lower()):
        return commit

    code, stdout, stderr = _run_git(["rev-parse", commit], cwd=mirror_path)
    if code != 0:
        raise RuntimeError(
            f"Cannot resolve commit '{commit}' in mirror {mirror_path}: {stderr}"
        )
    resolved = stdout.strip()
    if not resolved:
        raise RuntimeError(f"git rev-parse returned empty for '{commit}'")
    return resolved


def checkout_repo(repo: str, commit: str) -> Path:
    """Checkout a repo at *commit*, resolving symbolic refs like HEAD."""
    mirror_path = _ensure_bare_mirror(repo)
    resolved_commit = _resolve_commit(mirror_path, commit)
    worktree_path = _ensure_worktree(mirror_path, repo, resolved_commit)
    return worktree_path
