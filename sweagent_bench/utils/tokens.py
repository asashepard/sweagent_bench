"""Token estimation and accounting helpers."""
from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """Estimate token count with a lightweight, consistent heuristic."""
    if not text:
        return 0
    return int(math.ceil(len(text) / 4.0))


class TokenUsageTracker:
    """Accumulate reported and estimated token usage across LLM steps."""

    def __init__(self) -> None:
        self.reported_prompt_tokens = 0
        self.reported_completion_tokens = 0
        self.reported_total_tokens = 0
        self.estimated_prompt_tokens = 0
        self.estimated_completion_tokens = 0
        self.estimated_total_tokens = 0
        self._saw_reported = False

    def add_step(self, prompt_text: str, completion_text: str, usage: dict | None = None) -> dict:
        prompt_est = estimate_tokens(prompt_text)
        completion_est = estimate_tokens(completion_text)
        total_est = prompt_est + completion_est

        self.estimated_prompt_tokens += prompt_est
        self.estimated_completion_tokens += completion_est
        self.estimated_total_tokens += total_est

        prompt_rep = 0
        completion_rep = 0
        total_rep = 0

        if isinstance(usage, dict):
            prompt_rep = int(usage.get("prompt_tokens", 0) or 0)
            completion_rep = int(usage.get("completion_tokens", 0) or 0)
            total_rep = int(usage.get("total_tokens", 0) or 0)
            if total_rep <= 0 and (prompt_rep > 0 or completion_rep > 0):
                total_rep = prompt_rep + completion_rep

        if total_rep > 0 or prompt_rep > 0 or completion_rep > 0:
            self._saw_reported = True
            self.reported_prompt_tokens += prompt_rep
            self.reported_completion_tokens += completion_rep
            self.reported_total_tokens += total_rep
            source = "reported"
        else:
            source = "estimated"

        return {
            "source": source,
            "reported": {
                "prompt_tokens": prompt_rep,
                "completion_tokens": completion_rep,
                "total_tokens": total_rep,
            },
            "estimated": {
                "prompt_tokens": prompt_est,
                "completion_tokens": completion_est,
                "total_tokens": total_est,
            },
        }

    def export(self) -> dict:
        source = "reported" if self._saw_reported else "estimated"
        selected = {
            "prompt_tokens": self.reported_prompt_tokens if source == "reported" else self.estimated_prompt_tokens,
            "completion_tokens": self.reported_completion_tokens if source == "reported" else self.estimated_completion_tokens,
            "total_tokens": self.reported_total_tokens if source == "reported" else self.estimated_total_tokens,
        }
        return {
            "token_usage_source": source,
            "token_usage": selected,
            "reported_tokens": {
                "prompt_tokens": self.reported_prompt_tokens,
                "completion_tokens": self.reported_completion_tokens,
                "total_tokens": self.reported_total_tokens,
            },
            "estimated_tokens": {
                "prompt_tokens": self.estimated_prompt_tokens,
                "completion_tokens": self.estimated_completion_tokens,
                "total_tokens": self.estimated_total_tokens,
            },
        }
