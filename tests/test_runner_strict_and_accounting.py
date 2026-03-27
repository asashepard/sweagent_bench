"""Smoke tests for strict bash actionability, stall breaker, and token accounting."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import sweagent_bench.generation.sweagent_runner as runner


def _base_instance() -> dict:
    return {
        "instance_id": "iid-test",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix bug",
    }


def test_no_fenced_blocks_non_actionable_and_nudged() -> None:
    responses = [{"content": "I should inspect files first.", "usage": {}}]

    orig_chat = runner.chat_completion_with_metadata
    orig_exec = runner._execute_bash
    orig_git_diff = runner._extract_git_diff
    try:
        runner.chat_completion_with_metadata = lambda **kwargs: responses.pop(0)
        runner._execute_bash = lambda cmd, cwd: "ok"
        runner._extract_git_diff = lambda cwd: ""

        with TemporaryDirectory() as td:
            patch, info = runner._run_agent_loop(
                _base_instance(),
                "model",
                Path(td),
                max_steps=1,
                timeout_s=60,
                strict_bash_mode=True,
                patch_mode="git_diff",
            )

        assert patch == ""
        assert info["steps_non_actionable"] == 1
        assert info["steps_actionable"] == 0
        assert info["last_nudge"] == runner.ACTIONABLE_NUDGE
    finally:
        runner.chat_completion_with_metadata = orig_chat
        runner._execute_bash = orig_exec
        runner._extract_git_diff = orig_git_diff


def test_multiple_bash_blocks_first_used_rest_ignored() -> None:
    content = (
        "```bash\n"
        "echo first\n"
        "```\n"
        "```bash\n"
        "echo second\n"
        "```\n"
    )
    responses = [{"content": content, "usage": {}}]
    executed: list[str] = []

    orig_chat = runner.chat_completion_with_metadata
    orig_exec = runner._execute_bash
    orig_git_diff = runner._extract_git_diff
    try:
        runner.chat_completion_with_metadata = lambda **kwargs: responses.pop(0)

        def _fake_exec(cmd: str, cwd: Path) -> str:
            executed.append(cmd)
            return "ok"

        runner._execute_bash = _fake_exec
        runner._extract_git_diff = lambda cwd: ""

        with TemporaryDirectory() as td:
            _, info = runner._run_agent_loop(
                _base_instance(),
                "model",
                Path(td),
                max_steps=1,
                timeout_s=60,
                strict_bash_mode=True,
                patch_mode="git_diff",
            )

        assert executed == ["echo first"]
        assert info["extra_blocks_ignored"] == 1
    finally:
        runner.chat_completion_with_metadata = orig_chat
        runner._execute_bash = orig_exec
        runner._extract_git_diff = orig_git_diff


def test_repeated_command_three_times_triggers_stall_breaker() -> None:
    content = "```bash\necho loop\n```"
    responses = [
        {"content": content, "usage": {}},
        {"content": content, "usage": {}},
        {"content": content, "usage": {}},
    ]

    orig_chat = runner.chat_completion_with_metadata
    orig_exec = runner._execute_bash
    orig_git_diff = runner._extract_git_diff
    try:
        runner.chat_completion_with_metadata = lambda **kwargs: responses.pop(0)
        runner._execute_bash = lambda cmd, cwd: "ok"
        runner._extract_git_diff = lambda cwd: ""

        with TemporaryDirectory() as td:
            _, info = runner._run_agent_loop(
                _base_instance(),
                "model",
                Path(td),
                max_steps=3,
                timeout_s=60,
                strict_bash_mode=True,
                patch_mode="git_diff",
            )

        assert info["stall_detected"] is True
        assert info["stall_breaker_used"] is True
        assert info["stall_breaker_command"] == "git diff"
    finally:
        runner.chat_completion_with_metadata = orig_chat
        runner._execute_bash = orig_exec
        runner._extract_git_diff = orig_git_diff


def test_missing_usage_records_estimated_tokens() -> None:
    responses = [{"content": "```bash\ngit status\n```", "usage": {}}]

    orig_chat = runner.chat_completion_with_metadata
    orig_exec = runner._execute_bash
    orig_git_diff = runner._extract_git_diff
    try:
        runner.chat_completion_with_metadata = lambda **kwargs: responses.pop(0)
        runner._execute_bash = lambda cmd, cwd: "ok"
        runner._extract_git_diff = lambda cwd: ""

        with TemporaryDirectory() as td:
            _, info = runner._run_agent_loop(
                _base_instance(),
                "model",
                Path(td),
                max_steps=1,
                timeout_s=60,
                strict_bash_mode=True,
                patch_mode="git_diff",
            )

        assert info["token_usage_source"] == "estimated"
        assert int(info["estimated_tokens"]["total_tokens"]) > 0
        assert int(info["token_usage"]["total_tokens"]) > 0
    finally:
        runner.chat_completion_with_metadata = orig_chat
        runner._execute_bash = orig_exec
        runner._extract_git_diff = orig_git_diff


if __name__ == "__main__":
    test_no_fenced_blocks_non_actionable_and_nudged()
    test_multiple_bash_blocks_first_used_rest_ignored()
    test_repeated_command_three_times_triggers_stall_breaker()
    test_missing_usage_records_estimated_tokens()
    print("test_runner_strict_and_accounting: OK")
