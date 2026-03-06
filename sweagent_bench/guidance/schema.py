"""RepoGuidance — the single tunable artifact per repository."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CHAR_BUDGET = 3000


@dataclass
class RepoGuidance:
    """A bounded, line-oriented guidance block for one repository."""

    repo: str
    commit: str
    lines: list[str] = field(default_factory=list)
    version: int = 0
    char_budget: int = DEFAULT_CHAR_BUDGET

    def render(self) -> str:
        return "\n".join(self.lines)

    def char_count(self) -> int:
        return len(self.render())

    def is_within_budget(self) -> bool:
        return self.char_count() <= self.char_budget

    def to_dict(self) -> dict:
        return {
            "repo": self.repo, "commit": self.commit,
            "lines": list(self.lines), "version": self.version,
            "char_budget": self.char_budget,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RepoGuidance:
        return cls(
            repo=d["repo"], commit=d["commit"],
            lines=list(d.get("lines", [])),
            version=int(d.get("version", 0)),
            char_budget=int(d.get("char_budget", DEFAULT_CHAR_BUDGET)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RepoGuidance:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def copy(self, *, version: int | None = None, lines: list[str] | None = None) -> RepoGuidance:
        return RepoGuidance(
            repo=self.repo, commit=self.commit,
            lines=list(lines) if lines is not None else list(self.lines),
            version=version if version is not None else self.version,
            char_budget=self.char_budget,
        )
