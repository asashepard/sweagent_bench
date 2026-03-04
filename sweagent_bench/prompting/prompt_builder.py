"""Prompt construction for patch generation."""
from __future__ import annotations

from pathlib import Path

from sweagent_bench.utils.ignore import should_ignore_dir, should_ignore_file


def _should_ignore(name: str) -> bool:
    return should_ignore_dir(name) or should_ignore_file(name)


def _build_tree(repo_dir: Path, max_depth: int = 2) -> str:
    lines = []

    def _walk(current: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if not _should_ignore(e.name)]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(f"{repo_dir.name}/")
    _walk(repo_dir, "", 1)
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a coding assistant. Output ONLY a unified diff patch that fixes the issue.
Do not include any commentary, explanation, or markdown formatting outside the diff.
The patch must apply cleanly to the repository at the specified commit."""

USER_TEMPLATE = """\
## Repository
{repo} @ {commit}

## Issue
{problem_statement}

## Repository Structure (depth=2)
```
{tree}
```

## Output Requirements
- Output a unified diff patch only
- The patch must apply cleanly with `git apply`
- Make minimal, focused changes
- Do not include unrelated modifications"""

GUIDANCE_BLOCK_TEMPLATE = """

# REPO GUIDANCE (AUTO-TUNED)
{guidance_text}
# END REPO GUIDANCE"""

CONTEXT_BLOCK_TEMPLATE = GUIDANCE_BLOCK_TEMPLATE


def build_messages(
    problem_statement: str,
    repo: str,
    commit: str,
    repo_dir: Path,
    context_md: str | None = None,
    guidance_text: str | None = None,
) -> list[dict]:
    """Build OpenAI-format messages for patch generation."""
    text = guidance_text or context_md
    tree = _build_tree(repo_dir, max_depth=2)

    user_content = USER_TEMPLATE.format(
        repo=repo, commit=commit,
        problem_statement=problem_statement, tree=tree,
    )

    if text:
        user_content += GUIDANCE_BLOCK_TEMPLATE.format(guidance_text=text)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
