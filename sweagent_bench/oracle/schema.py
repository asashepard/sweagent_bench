"""Dataclasses for the LLM-as-judge oracle evaluator loop."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Edit:
    """A structured edit to apply to AGENTS.md."""
    section: str
    action: str  # "add", "modify", "strengthen", "remove"
    content: str


@dataclass
class Probe:
    """A micro-test probe generated for AGENTS.md stress-testing."""
    id: str
    task: str
    expected_behaviors: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class BehaviorReview:
    """Diagnostic review of one expected behavior."""
    behavior: str
    assessment: str  # strong|partial|missing
    evidence: str = ""
    improvement: str = ""


@dataclass
class ProbeResult:
    """Diagnostic result of evaluating one probe."""
    probe_id: str
    task: str
    response: str
    behavior_reviews: list[BehaviorReview] = field(default_factory=list)
    proposed_edits: list[Edit] = field(default_factory=list)
    overall_notes: str = ""


@dataclass
class OracleConfig:
    """Configuration for the oracle evaluator loop."""
    repo: str
    commit: str
    model: str
    iterations: int = 5
    timeout_s: int = 120
    output_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OracleState:
    """Persistent state for the oracle evaluator loop."""
    repo: str
    current_version: int = 0
    completed_iterations: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OracleState:
        return cls(
            repo=d["repo"],
            current_version=d.get("current_version", 0),
            completed_iterations=d.get("completed_iterations", 0),
            history=list(d.get("history", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OracleState:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
