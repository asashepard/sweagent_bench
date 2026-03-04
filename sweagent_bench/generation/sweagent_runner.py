"""Mini-swe-agent runner — inline iterative agent loop for patch generation.

Replaces SWE-agent subprocess with a lightweight agent loop that:
1. Checks out the repo via git worktree.
2. Presents the issue + repo structure to the LLM.
3. Loops up to max_steps, letting the model run bash commands.
4. Captures submitted diff patches from model output.
5. Falls back to single-shot generation if no patch produced.

Key design decisions:
- 1800s (30min) wall-clock timeout per instance.
- 50 max steps per agent run.
- ContextLengthError → immediate fail-fast, no blind retries.
- Bash commands executed in the repo worktree with 120s per-command cap.
- Stall detection: 3 consecutive identical commands → abort loop.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from sweagent_bench.generation.patch_utils import extract_diff, sanitize_patch_for_preds
from sweagent_bench.llm.openai_compat import ContextLengthError, chat_completion


# ── Constants ──────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 1800      # 30 minutes per instance
DEFAULT_MAX_STEPS = 50        # agent loop iterations
INPUT_SLACK_TOKENS = 1024     # never trim to exact boundary
DEFAULT_MAX_OUTPUT_TOKENS = 512   # single-shot fallback
AGENT_MAX_TOKENS = 4096           # per-turn budget in the agent loop
CMD_TIMEOUT_S = 120               # per-command timeout
STALL_THRESHOLD = 3               # consecutive identical commands → stall


# ── Agent prompts ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are an expert software engineer tasked with fixing a bug in a repository.
You have access to a bash shell in the repo's working directory.

## How to interact

**Run a command:** wrap it in a ```bash block:
```bash
grep -rn "some_pattern" src/
```

**Submit your fix:** when you are confident, output a unified diff in a ```diff block:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,3 +10,4 @@
 existing line
-old line
+new line
+added line
```

## Rules
- Explore the codebase first. Read relevant files before making changes.
- Make minimal, focused changes that fix only the reported issue.
- Your diff must apply cleanly with `git apply`.
- Do NOT run interactive commands (editors, pagers, less, vim).
- If a command produces too much output, use `head`, `tail`, or `grep` to filter.
- Do NOT modify test files unless the issue specifically asks for test changes.
"""

AGENT_USER_TEMPLATE = """\
## Repository
{repo} @ {commit}

## Issue
{problem_statement}

## Repository Structure (depth=2)
```
{tree}
```
{guidance_block}
Fix this issue. Start by exploring the relevant code, then submit a ```diff block with your patch."""


# ── Agent helpers ─────────────────────────────────────────────

def _build_agent_messages(
    instance: dict,
    repo_dir: Path,
    guidance_text: str | None = None,
) -> list[dict]:
    """Build the initial system + user messages for the agent loop."""
    from sweagent_bench.prompting.prompt_builder import _build_tree

    tree = _build_tree(repo_dir, max_depth=2)
    guidance_block = ""
    if guidance_text:
        guidance_block = (
            f"\n\n# REPO GUIDANCE (AUTO-TUNED)\n"
            f"{guidance_text}\n"
            f"# END REPO GUIDANCE"
        )

    user_content = AGENT_USER_TEMPLATE.format(
        repo=instance["repo"],
        commit=instance["base_commit"],
        problem_statement=instance.get("problem_statement", ""),
        tree=tree,
        guidance_block=guidance_block,
    )

    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Extract fenced code blocks as ``(language, content)`` tuples."""
    return re.findall(r"```(\w*)\s*\n(.*?)```", text, re.DOTALL)


def _execute_bash(cmd: str, cwd: Path) -> str:
    """Run a bash command in the worktree and return combined output."""
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT_S,
        )
        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += proc.stderr
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        # Truncate very long outputs to stay within context budget
        if len(output) > 10_000:
            output = output[:5000] + "\n\n... [truncated] ...\n\n" + output[-3000:]
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[command timed out after {CMD_TIMEOUT_S}s]"
    except Exception as e:
        return f"[error running command: {e}]"


def _extract_git_diff(cwd: Path) -> str:
    """Get uncommitted changes from the worktree via ``git diff``."""
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def _run_agent_loop(
    instance: dict,
    model: str,
    repo_dir: Path,
    *,
    guidance_text: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    api_base: str | None = None,
) -> tuple[str, dict]:
    """Run the iterative mini-swe-agent loop.

    Returns:
        ``(patch_string, info_dict)`` where *info_dict* carries
        ``steps_taken``, ``stall_detected``, ``stall_type``,
        ``stall_repeat_count``, ``status``, ``error``.
    """
    messages = _build_agent_messages(instance, repo_dir, guidance_text)

    info: dict = {
        "steps_taken": 0,
        "stall_detected": False,
        "stall_type": None,
        "stall_repeat_count": 0,
        "status": "ok",
        "error": None,
    }

    deadline = time.perf_counter() + timeout_s
    recent_cmds: list[str] = []

    for step in range(max_steps):
        if time.perf_counter() >= deadline:
            info["status"] = "timeout"
            info["error"] = f"Agent timed out after {timeout_s}s at step {step}"
            break

        info["steps_taken"] = step + 1

        # ── LLM call (env save/restore for api_base) ──
        prev_base = os.environ.get("OPENAI_BASE_URL")
        try:
            if api_base:
                os.environ["OPENAI_BASE_URL"] = api_base

            remaining = max(30, int(deadline - time.perf_counter()))
            response_text = chat_completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=AGENT_MAX_TOKENS,
                timeout_s=min(remaining, 300),
            )
        except ContextLengthError as e:
            info["status"] = "context_length_error"
            info["error"] = str(e)
            break
        except Exception as e:
            info["status"] = "error"
            info["error"] = f"LLM call failed at step {step}: {e}"
            break
        finally:
            if api_base:
                if prev_base is not None:
                    os.environ["OPENAI_BASE_URL"] = prev_base
                else:
                    os.environ.pop("OPENAI_BASE_URL", None)

        # ── Parse response for fenced blocks ──
        blocks = _parse_fenced_blocks(response_text)

        # Check for ```diff submission
        for lang, content in blocks:
            if lang.lower() == "diff" and content.strip():
                messages.append({"role": "assistant", "content": response_text})
                return content.strip(), info

        # Check for raw inline diff (model forgot to fence it)
        bash_langs = {"bash", "sh"}
        has_bash = any(lang.lower() in bash_langs for lang, _ in blocks)
        if not has_bash:
            raw = extract_diff(response_text)
            if raw and ("diff --git" in raw or "--- a/" in raw):
                messages.append({"role": "assistant", "content": response_text})
                return raw, info

        # ── Execute bash blocks ──
        bash_contents = [
            content for lang, content in blocks
            if lang.lower() in bash_langs and content.strip()
        ]

        if bash_contents:
            observations: list[str] = []
            stalled = False
            for cmd_text in bash_contents:
                cmd = cmd_text.strip()
                # Stall detection
                recent_cmds.append(cmd)
                if len(recent_cmds) > STALL_THRESHOLD:
                    recent_cmds = recent_cmds[-STALL_THRESHOLD:]
                if (
                    len(recent_cmds) >= STALL_THRESHOLD
                    and len(set(recent_cmds)) == 1
                ):
                    info["stall_detected"] = True
                    info["stall_type"] = "repeated_command"
                    info["stall_repeat_count"] = STALL_THRESHOLD
                    stalled = True
                    break

                output = _execute_bash(cmd, repo_dir)
                observations.append(f"$ {cmd}\n{output}")

            if stalled:
                break

            messages.append({"role": "assistant", "content": response_text})
            obs_text = "\n\n".join(observations)
            messages.append({
                "role": "user",
                "content": f"## Command Output\n{obs_text}",
            })
        else:
            # Model produced text without actionable blocks — nudge it
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    "Please run a ```bash command to explore the code, "
                    "or submit a ```diff block with your fix."
                ),
            })

    # ── Last resort: check git diff for any edits the agent made ──
    worktree_patch = _extract_git_diff(repo_dir)
    if worktree_patch:
        return worktree_patch, info

    return "", info


# ── Public API ────────────────────────────────────────────────

def generate_patch_with_sweagent(
    instance: dict,
    model: str,
    *,
    guidance_text: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_steps: int = DEFAULT_MAX_STEPS,
    traj_dir: Path | None = None,
    api_base: str | None = None,
) -> dict:
    """Run the mini-swe-agent on one instance and return result metadata.

    The function signature and return-dict shape are intentionally kept
    identical to the old SWE-agent subprocess runner so callers (the
    orchestrator) need zero changes.

    Args:
        instance: SWE-bench instance dict with instance_id, repo,
            base_commit, problem_statement.
        model: Model name/path.
        guidance_text: Optional AGENTS.md guidance to prepend.
        timeout_s: Wall-clock timeout in seconds (default 1800).
        max_steps: Maximum agent loop iterations (default 50).
        traj_dir: (unused, kept for API compat) trajectory directory.
        api_base: vLLM/OpenAI API base URL.

    Returns:
        Dict with keys: instance_id, patch, elapsed_s, token_usage,
        status, error, patch_source, fallback_single_shot_used, etc.
    """
    iid = instance["instance_id"]
    start = time.perf_counter()

    result = {
        "instance_id": iid,
        "patch": "",
        "elapsed_s": 0.0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "status": "ok",
        "error": None,
        "patch_source": "empty",
        "fallback_single_shot_used": False,
        "fallback_single_shot_patch_len": 0,
        "fallback_single_shot_raw_len": 0,
        "fallback_reason": None,
        "fallback_single_shot_truncated": False,
        "stall_detected": False,
        "stall_type": None,
        "stall_action": None,
        "stall_repeat_count": 0,
    }

    # ── Checkout repo ──
    try:
        from sweagent_bench.git.checkout import checkout_repo
        repo_dir = checkout_repo(instance["repo"], instance["base_commit"])
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Checkout failed: {exc}"
        result["elapsed_s"] = time.perf_counter() - start
        return result

    # ── Agent loop ──
    try:
        patch, loop_info = _run_agent_loop(
            instance,
            model,
            repo_dir,
            guidance_text=guidance_text,
            max_steps=max_steps,
            timeout_s=timeout_s,
            api_base=api_base,
        )
    except Exception as exc:
        patch = ""
        loop_info = {
            "status": "error", "error": str(exc),
            "stall_detected": False, "stall_type": None, "stall_repeat_count": 0,
        }

    # Propagate loop metadata
    result["status"] = loop_info.get("status", "ok")
    result["error"] = loop_info.get("error")
    result["stall_detected"] = loop_info.get("stall_detected", False)
    result["stall_type"] = loop_info.get("stall_type")
    result["stall_repeat_count"] = loop_info.get("stall_repeat_count", 0)

    if patch:
        result["patch"] = patch
        result["patch_source"] = "agent_loop"

    # ── Fallback to single-shot if agent produced no patch ──
    if not patch:
        try:
            patch = _fallback_single_shot(instance, model, guidance_text, api_base)
            if patch:
                result["patch"] = patch
                result["fallback_single_shot_used"] = True
                result["fallback_single_shot_patch_len"] = len(patch)
                result["fallback_reason"] = "agent_empty_patch"
                result["patch_source"] = "fallback_single_shot"
        except ContextLengthError:
            result["fallback_reason"] = "context_length_error_in_fallback"

    result["elapsed_s"] = time.perf_counter() - start
    return result


def _fallback_single_shot(
    instance: dict,
    model: str,
    guidance_text: str | None,
    api_base: str | None,
) -> str:
    """Fallback: generate patch via direct LLM call (no agent loop)."""
    from sweagent_bench.git.checkout import checkout_repo
    from sweagent_bench.llm.openai_compat import chat_completion
    from sweagent_bench.prompting.prompt_builder import build_messages

    repo_dir = checkout_repo(instance["repo"], instance["base_commit"])
    messages = build_messages(
        problem_statement=instance.get("problem_statement", ""),
        repo=instance["repo"],
        commit=instance["base_commit"],
        repo_dir=repo_dir,
        guidance_text=guidance_text,
    )

    # Temporarily override API base if provided, then restore.
    # This avoids permanently mutating the process environment.
    prev_base = os.environ.get("OPENAI_BASE_URL")
    try:
        if api_base:
            os.environ["OPENAI_BASE_URL"] = api_base

        raw = chat_completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            timeout_s=120,
        )
    finally:
        if api_base:
            if prev_base is not None:
                os.environ["OPENAI_BASE_URL"] = prev_base
            else:
                os.environ.pop("OPENAI_BASE_URL", None)

    from sweagent_bench.generation.patch_utils import extract_diff
    return extract_diff(raw)
