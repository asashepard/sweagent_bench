"""Fallback single-shot patch generation.

Used when SWE-agent produces an empty patch. Calls the LLM directly
with the issue + optional guidance text and extracts a diff.

Adapted from context_policy/runner/single_shot.py — import paths
updated, max_tokens default lowered to 512 for Qwen 3.5 35B-A3B.
"""
from __future__ import annotations

from pathlib import Path

from sweagent_bench.git.checkout import checkout_repo
from sweagent_bench.llm.openai_compat import chat_completion
from sweagent_bench.prompting.prompt_builder import build_messages
from sweagent_bench.generation.patch_utils import MAX_PATCH_SIZE, extract_diff


def generate_patch(
    instance: dict,
    model: str,
    context_md: str | None = None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    timeout_s: int = 120,
) -> str:
    """Generate a patch for a SWE-bench instance using a single model call.

    Args:
        instance: Instance dict with instance_id, repo, base_commit, problem_statement.
        model: Model name to use for inference.
        context_md: Optional additional context to include in prompt.
        temperature: Sampling temperature (default 0.0).
        top_p: Top-p parameter (default 1.0).
        max_tokens: Maximum tokens to generate (default 512).
        timeout_s: Request timeout in seconds (default 120).

    Returns:
        Extracted unified diff patch string, or empty string if extraction failed.
    """
    repo = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]

    # Checkout repo at commit
    repo_dir = checkout_repo(repo, commit)

    # Build prompt messages
    messages = build_messages(
        problem_statement=problem_statement,
        repo=repo,
        commit=commit,
        repo_dir=repo_dir,
        context_md=context_md,
    )

    # Call model
    response = chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    # Extract diff
    diff = extract_diff(response)

    # Safety: reject oversized patches
    if len(diff) > MAX_PATCH_SIZE:
        return ""

    return diff
