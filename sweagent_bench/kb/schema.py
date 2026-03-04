"""RepoKB — the structured knowledge base artifact for one repository."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepoKB:
    """Structured knowledge base for one repository snapshot."""

    repo: str
    commit: str
    version: int = 0

    architecture: str = ""
    symbol_map: str = ""
    context: str = ""
    conventions: str = ""

    def render(self) -> str:
        sections: list[str] = []
        if self.architecture:
            sections.append(f"## Architecture\n\n{self.architecture}")
        if self.symbol_map:
            sections.append(f"## Symbol Map\n\n{self.symbol_map}")
        if self.context:
            sections.append(f"## Context\n\n{self.context}")
        if self.conventions:
            sections.append(f"## Conventions\n\n{self.conventions}")
        return "\n\n".join(sections)

    def render_truncated(self, char_budget: int = 60_000) -> str:
        full = self.render()
        if len(full) <= char_budget:
            return full
        return full[:char_budget - 20] + "\n\n[KB truncated]"

    def to_dict(self) -> dict:
        return {
            "repo": self.repo, "commit": self.commit, "version": self.version,
            "architecture": self.architecture, "symbol_map": self.symbol_map,
            "context": self.context, "conventions": self.conventions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RepoKB:
        return cls(
            repo=d["repo"], commit=d["commit"],
            version=int(d.get("version", 0)),
            architecture=d.get("architecture", ""),
            symbol_map=d.get("symbol_map", ""),
            context=d.get("context", ""),
            conventions=d.get("conventions", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RepoKB:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
