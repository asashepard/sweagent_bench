"""Path utilities and artifact location constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREDS_DIR = ARTIFACTS_DIR / "preds"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CONTEXTS_DIR = ARTIFACTS_DIR / "contexts"
REPOS_CACHE_DIR = ARTIFACTS_DIR / "repos_cache"
WORKTREES_DIR = ARTIFACTS_DIR / "worktrees"

RESULTS_DIR = PROJECT_ROOT / "results"


def repo_to_dirname(repo: str) -> str:
    return repo.replace("/", "__")


def get_context_path(repo: str, commit: str, instance_id: str | None = None) -> Path:
    base = CONTEXTS_DIR / repo_to_dirname(repo) / commit
    if instance_id:
        return base / instance_id / "context.md"
    return base / "context.md"


def get_worktree_path(repo: str, commit: str) -> Path:
    return WORKTREES_DIR / repo_to_dirname(repo) / commit
