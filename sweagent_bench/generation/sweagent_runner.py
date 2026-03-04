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

Debug artifacts:
  Set SWEAGENT_BENCH_DEBUG=1 to write per-instance debug files under
  results/<experiment_id>/debug/<condition>/<instance_id>/:
    assistant_step_<n>.txt  — raw model response for each step
    bash_step_<n>.txt       — commands + stdout/stderr
    extraction_summary.txt  — extraction decisions
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sweagent_bench.generation.patch_utils import extract_diff, sanitize_patch_for_preds
from sweagent_bench.llm.openai_compat import ContextLengthError, chat_completion


# ── Constants ──────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 1800      # 30 minutes per instance
DEFAULT_MAX_STEPS = 50        # agent loop iterations
INPUT_SLACK_TOKENS = 1024     # never trim to exact boundary
DEFAULT_MAX_OUTPUT_TOKENS = 512   # single-shot fallback
AGENT_MAX_TOKENS = 2048           # per-turn budget in the agent loop
CMD_TIMEOUT_S = 120               # per-command timeout
STALL_THRESHOLD = 3               # consecutive identical commands → stall
PER_COMMAND_OBS_CAP = 4000        # per-command observation cap for message appends


def _rlog(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [runner] {msg}", flush=True)


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

def _is_debug() -> bool:
    """Return True when SWEAGENT_BENCH_DEBUG=1 is set."""
    return os.environ.get("SWEAGENT_BENCH_DEBUG", "") == "1"


def _debug_write(debug_dir: Path | None, filename: str, content: str) -> None:
    """Write a debug artifact file (no-op when *debug_dir* is None)."""
    if debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / filename).write_text(content, encoding="utf-8")
    except Exception:
        pass  # never let debug I/O crash the pipeline


def _strip_think_markers(text: str) -> str:
    """Remove literal ``<think>`` / ``</think>`` markers, keep content."""
    return text.replace("<think>", "").replace("</think>", "")


def _truncate_head_tail(text: str, max_len: int, head: int, tail: int) -> str:
    """Truncate text by keeping head+tail with an explicit middle marker."""
    if len(text) <= max_len:
        return text
    return text[:head] + "\n\n... [truncated] ...\n\n" + text[-tail:]


def _trim_messages(messages: list[dict], max_turns: int = 8) -> list[dict]:
    """Keep system + initial task statement + latest conversation turns.

    Policy:
    - Preserve first system message if present.
    - Preserve first user message after system (task statement).
    - Keep only latest ``max_turns * 2`` remaining user/assistant messages.
    """
    if not messages:
        return messages

    first_system = None
    rest_after_system: list[dict]
    if messages and messages[0].get("role") == "system":
        first_system = messages[0]
        rest_after_system = messages[1:]
    else:
        rest_after_system = messages[:]

    task_message = None
    remainder: list[dict]
    if rest_after_system and rest_after_system[0].get("role") == "user":
        task_message = rest_after_system[0]
        remainder = rest_after_system[1:]
    else:
        remainder = rest_after_system

    max_tail_messages = max(0, max_turns * 2)
    tail = remainder[-max_tail_messages:] if max_tail_messages else []

    trimmed: list[dict] = []
    if first_system is not None:
        trimmed.append(first_system)
    if task_message is not None:
        trimmed.append(task_message)
    trimmed.extend(tail)
    return trimmed


def _get_agent_max_tokens() -> int:
    """Get per-step completion budget from env, defaulting to 2048."""
    raw = os.environ.get("SWEAGENT_MAX_TOKENS", "").strip()
    if not raw:
        return AGENT_MAX_TOKENS
    try:
        val = int(raw)
        if val > 0:
            return val
    except ValueError:
        pass
    return AGENT_MAX_TOKENS


def _is_install_blocked_command(cmd: str) -> bool:
    """Return True if command should be blocked unless install override is enabled."""
    if os.environ.get("SWEAGENT_ALLOW_INSTALL", "") == "1":
        return False
    norm = cmd.strip().lower()
    if norm.startswith("pip install"):
        return True
    if norm.startswith("python setup.py"):
        return True
    return False


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
    """Extract fenced code blocks as ``(language, content)`` tuples.

    Handles ``diff``, ``patch``, ``bash``, ``sh``, and untagged fences.
    """
    return re.findall(r"```(\w*)\s*\n(.*?)```", text, re.DOTALL)


_DIFF_FENCE_LANGS = {"diff", "patch"}
_BASH_FENCE_LANGS = {"bash", "sh"}


def _extract_last_diff_block(text: str) -> str:
    """Extract the **last** diff block from model output.

    Checks in order:
    1. Fenced blocks tagged ``diff`` or ``patch`` — last one wins.
    2. Raw unfenced ``diff --git`` header — last occurrence wins.
    3. ``extract_diff()`` fallback for ``---`` headers.
    """
    blocks = _parse_fenced_blocks(text)

    # Collect all diff/patch fenced blocks
    diff_blocks = [
        content.strip()
        for lang, content in blocks
        if lang.lower() in _DIFF_FENCE_LANGS and content.strip()
    ]
    if diff_blocks:
        return diff_blocks[-1]  # last one wins

    # Raw unfenced: find last "diff --git" line
    lines = text.split("\n")
    last_start = None
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            last_start = i
    if last_start is not None:
        return "\n".join(lines[last_start:]).strip()

    # Final fallback via patch_utils.extract_diff
    return extract_diff(text)


def _execute_bash(cmd: str, cwd: Path) -> str:
    """Run a bash command in the worktree and return combined output."""
    if _is_install_blocked_command(cmd):
        return (
            "[blocked command] Installation commands are disabled in this loop "
            "(pip install / python setup.py). Proceed without installing packages. "
            "Set SWEAGENT_ALLOW_INSTALL=1 to allow installs."
        )

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
            output = _truncate_head_tail(output, max_len=10_000, head=5000, tail=3000)
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
    debug_dir: Path | None = None,
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
        "diff_block_found": False,
        "git_diff_non_empty": False,
    }

    deadline = time.perf_counter() + timeout_s
    recent_cmds: list[str] = []
    _rlog(
        f"Agent loop start iid={instance.get('instance_id')} repo={instance.get('repo')} "
        f"max_steps={max_steps} timeout_s={timeout_s}"
    )

    for step in range(max_steps):
        step_start = time.perf_counter()
        if time.perf_counter() >= deadline:
            info["status"] = "timeout"
            info["error"] = f"Agent timed out after {timeout_s}s at step {step}"
            _rlog(f"Agent loop timeout at step={step} iid={instance.get('instance_id')}")
            break

        info["steps_taken"] = step + 1
        _rlog(f"Step {step+1}/{max_steps} start iid={instance.get('instance_id')}")

        # ── LLM call (env save/restore for api_base) ──
        prev_base = os.environ.get("OPENAI_BASE_URL")
        try:
            if api_base:
                os.environ["OPENAI_BASE_URL"] = api_base

            remaining = max(30, int(deadline - time.perf_counter()))
            t_llm = time.perf_counter()
            messages = _trim_messages(messages, max_turns=8)
            max_tokens = _get_agent_max_tokens()
            response_text = chat_completion(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_s=min(remaining, 300),
            )
            _rlog(
                f"Step {step+1}: LLM response received in {time.perf_counter() - t_llm:.2f}s "
                f"chars={len(response_text)} max_tokens={max_tokens}"
            )
        except ContextLengthError as e:
            info["status"] = "context_length_error"
            info["error"] = str(e)
            _rlog(f"Step {step+1}: context length error: {e}")
            break
        except Exception as e:
            info["status"] = "error"
            info["error"] = f"LLM call failed at step {step}: {e}"
            _rlog(f"Step {step+1}: LLM call error: {e}")
            break
        finally:
            if api_base:
                if prev_base is not None:
                    os.environ["OPENAI_BASE_URL"] = prev_base
                else:
                    os.environ.pop("OPENAI_BASE_URL", None)

        # ── Debug: write raw assistant response ──
        _debug_write(debug_dir, f"assistant_step_{step}.txt", response_text)

        # ── Pre-process: strip <think>/</ think> markers, keep content ──
        cleaned = _strip_think_markers(response_text)

        # ── Parse response for fenced blocks ──
        blocks = _parse_fenced_blocks(cleaned)
        _rlog(f"Step {step+1}: parsed fenced blocks count={len(blocks)}")

        # Check for diff/patch submission (last block wins)
        diff_candidate = _extract_last_diff_block(cleaned)
        has_bash = any(lang.lower() in _BASH_FENCE_LANGS for lang, _ in blocks)

        if diff_candidate and ("diff --git" in diff_candidate or "--- " in diff_candidate):
            info["diff_block_found"] = True
            messages.append({"role": "assistant", "content": response_text})
            _debug_write(debug_dir, f"diff_extracted_step_{step}.txt", diff_candidate)
            _rlog(
                f"Step {step+1}: extracted diff candidate len={len(diff_candidate)} "
                f"returning patch"
            )
            return diff_candidate, info

        # ── Execute bash blocks ──
        bash_contents = [
            content for lang, content in blocks
            if lang.lower() in _BASH_FENCE_LANGS and content.strip()
        ]

        if bash_contents:
            _rlog(f"Step {step+1}: executing {len(bash_contents)} bash command block(s)")
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
                    _rlog(
                        f"Step {step+1}: stall detected repeated command x{STALL_THRESHOLD}: {cmd}"
                    )
                    stalled = True
                    break

                _rlog(f"Step {step+1}: running command: {cmd}")
                t_cmd = time.perf_counter()
                output = _execute_bash(cmd, repo_dir)
                output_for_observation = _truncate_head_tail(
                    output,
                    max_len=PER_COMMAND_OBS_CAP,
                    head=2000,
                    tail=2000,
                )
                _rlog(
                    f"Step {step+1}: command done in {time.perf_counter() - t_cmd:.2f}s "
                    f"output_chars={len(output)} obs_chars={len(output_for_observation)}"
                )
                observations.append(f"$ {cmd}\n{output_for_observation}")

            # Debug: write bash commands + outputs
            _debug_write(
                debug_dir,
                f"bash_step_{step}.txt",
                "\n\n".join(observations),
            )

            if stalled:
                break

            messages.append({"role": "assistant", "content": response_text})
            obs_text = "\n\n".join(observations)
            messages.append({
                "role": "user",
                "content": f"## Command Output\n{obs_text}",
            })
            _rlog(
                f"Step {step+1}: command observations appended chars={len(obs_text)} "
                f"step_elapsed={time.perf_counter() - step_start:.2f}s"
            )
        else:
            # No bash and no diff — check for raw unfenced diff one more time
            if not has_bash:
                raw = extract_diff(cleaned)
                if raw and ("diff --git" in raw or "--- a/" in raw):
                    info["diff_block_found"] = True
                    messages.append({"role": "assistant", "content": response_text})
                    _debug_write(debug_dir, f"diff_extracted_step_{step}.txt", raw)
                    _rlog(
                        f"Step {step+1}: extracted raw diff fallback len={len(raw)} returning patch"
                    )
                    return raw, info

            # Model produced text without actionable blocks — nudge it
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Next response MUST contain either exactly one bash block OR exactly one diff/patch block. No other text.",
            })
            _rlog(
                f"Step {step+1}: no actionable block found; nudged model "
                f"step_elapsed={time.perf_counter() - step_start:.2f}s"
            )

    # ── Last resort: check git diff for any edits the agent made ──
    worktree_patch = _extract_git_diff(repo_dir)
    if worktree_patch:
        info["git_diff_non_empty"] = True
        _debug_write(debug_dir, "git_diff_fallback.txt", worktree_patch)
        _rlog(f"Agent loop fallback git diff found len={len(worktree_patch)}")
        return worktree_patch, info

    _rlog(
        f"Agent loop ended without patch iid={instance.get('instance_id')} "
        f"status={info.get('status')} steps={info.get('steps_taken')}"
    )
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
        traj_dir: Directory for debug artifacts (reused for compat).
        api_base: vLLM/OpenAI API base URL.

    Returns:
        Dict with keys: instance_id, patch, elapsed_s, token_usage,
        status, error, patch_source, fallback_single_shot_used, etc.
    """
    iid = instance["instance_id"]
    start = time.perf_counter()
    _rlog(
        f"Instance start iid={iid} repo={instance.get('repo')} commit={instance.get('base_commit')} "
        f"model={model} timeout_s={timeout_s} max_steps={max_steps}"
    )

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

    # ── Debug directory (opt-in via SWEAGENT_BENCH_DEBUG=1) ──
    debug_dir: Path | None = None
    if _is_debug() and traj_dir is not None:
        # traj_dir is typically: artifacts/preds/<exp>/<condition>/trajectories
        # we go up two levels to get the experiment root, then use debug/ subdir
        debug_dir = traj_dir.parent.parent.parent / "debug" / traj_dir.parent.name / iid
        debug_dir.mkdir(parents=True, exist_ok=True)
        _rlog(f"Instance {iid}: debug enabled dir={debug_dir}")

    # ── Checkout repo ──
    try:
        from sweagent_bench.git.checkout import checkout_repo
        t_checkout = time.perf_counter()
        repo_dir = checkout_repo(instance["repo"], instance["base_commit"])
        _rlog(f"Instance {iid}: checkout complete in {time.perf_counter() - t_checkout:.2f}s path={repo_dir}")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Checkout failed: {exc}"
        result["elapsed_s"] = time.perf_counter() - start
        _rlog(f"Instance {iid}: checkout failed: {exc}")
        return result

    # ── Agent loop ──
    try:
        t_loop = time.perf_counter()
        patch, loop_info = _run_agent_loop(
            instance,
            model,
            repo_dir,
            guidance_text=guidance_text,
            max_steps=max_steps,
            timeout_s=timeout_s,
            api_base=api_base,
            debug_dir=debug_dir,
        )
        _rlog(
            f"Instance {iid}: agent loop complete in {time.perf_counter() - t_loop:.2f}s "
            f"status={loop_info.get('status')} patch_len={len(patch)}"
        )
    except Exception as exc:
        patch = ""
        loop_info = {
            "status": "error", "error": str(exc),
            "stall_detected": False, "stall_type": None, "stall_repeat_count": 0,
        }
        _rlog(f"Instance {iid}: agent loop exception: {exc}")

    # Propagate loop metadata
    result["status"] = loop_info.get("status", "ok")
    result["error"] = loop_info.get("error")
    result["stall_detected"] = loop_info.get("stall_detected", False)
    result["stall_type"] = loop_info.get("stall_type")
    result["stall_repeat_count"] = loop_info.get("stall_repeat_count", 0)

    if patch:
        result["patch"] = patch
        result["patch_source"] = "agent_loop"
        _rlog(f"Instance {iid}: patch from agent loop len={len(patch)}")

    # ── Fallback to single-shot if agent produced no patch ──
    if not patch:
        try:
            _rlog(f"Instance {iid}: entering single-shot fallback")
            t_fallback = time.perf_counter()
            patch = _fallback_single_shot(instance, model, guidance_text, api_base)
            if patch:
                result["patch"] = patch
                result["fallback_single_shot_used"] = True
                result["fallback_single_shot_patch_len"] = len(patch)
                result["fallback_reason"] = "agent_empty_patch"
                result["patch_source"] = "fallback_single_shot"
                _rlog(
                    f"Instance {iid}: fallback produced patch len={len(patch)} "
                    f"in {time.perf_counter() - t_fallback:.2f}s"
                )
            else:
                _rlog(
                    f"Instance {iid}: fallback returned empty patch in "
                    f"{time.perf_counter() - t_fallback:.2f}s"
                )
        except ContextLengthError:
            result["fallback_reason"] = "context_length_error_in_fallback"
            _rlog(f"Instance {iid}: fallback context length error")

    result["elapsed_s"] = time.perf_counter() - start
    _rlog(
        f"Instance done iid={iid} elapsed={result['elapsed_s']:.2f}s "
        f"status={result['status']} patch_source={result['patch_source']} "
        f"patch_len={len(result['patch'])}"
    )

    # ── Debug: extraction summary ──
    _debug_write(debug_dir, "extraction_summary.txt", (
        f"instance_id: {iid}\n"
        f"steps_taken: {loop_info.get('steps_taken', 0)}\n"
        f"diff_block_found: {loop_info.get('diff_block_found', False)}\n"
        f"git_diff_non_empty: {loop_info.get('git_diff_non_empty', False)}\n"
        f"patch_source: {result['patch_source']}\n"
        f"patch_len_before_sanitation: {len(result['patch'])}\n"
        f"fallback_single_shot_used: {result['fallback_single_shot_used']}\n"
        f"status: {result['status']}\n"
        f"error: {result['error']}\n"
        f"stall_detected: {result['stall_detected']}\n"
    ))

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
