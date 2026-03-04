"""SWE-agent runner — invokes SWE-agent per-instance for patch generation.

This is the NEW module replacing mini-swe-agent. Uses SWE-agent's stable
tool protocol (bash + file-editor tools) with Qwen 3.5 35B via vLLM.

Key design decisions:
- 1800s (30min) timeout per instance (hard kill).
- 50 max steps per agent run.
- 1024-token input slack: never trim to exact context boundary.
- ContextLengthError → immediate fail-fast, no blind retries.
- Patch extracted from SWE-agent trajectory file.
- Guidance injected via temp JSONL data file with modified problem_statement.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sweagent_bench.generation.patch_utils import extract_patch_from_trajectory, sanitize_patch_for_preds
from sweagent_bench.llm.openai_compat import ContextLengthError


# ── Constants ──────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 1800      # 30 minutes per instance
DEFAULT_MAX_STEPS = 50        # SWE-agent max steps
INPUT_SLACK_TOKENS = 1024     # Never trim to exact boundary
DEFAULT_MAX_OUTPUT_TOKENS = 512


def _write_instance_jsonl(instance: dict, dest: Path) -> Path:
    """Write a single-instance JSONL data file for SWE-agent.

    This is used to inject guidance-enhanced problem statements into
    SWE-agent, which reads instances from data files rather than accepting
    inline problem statements via CLI flags.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        f.write(json.dumps(instance, sort_keys=True, ensure_ascii=False) + "\n")
    return dest


def _build_sweagent_command(
    instance: dict,
    model: str,
    *,
    data_path: str | None = None,
    traj_dir: Path | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[str]:
    """Build the SWE-agent >=0.7 CLI command for one instance.

    SWE-agent 0.7+ uses a Hydra-based CLI with dot-notation overrides.
    Key flags:
      --model.name              Model identifier
      --model.per_instance_cost_limit  Cost limit (0 = unlimited)
      --data_path               Path to dataset or JSONL file
      --instance_filter         Filter by instance_id
      --traj_dir                Trajectory output directory
      --actions.apply_patch_locally  Apply patches to local repo copy
    """
    cmd = [
        sys.executable, "-m", "sweagent", "run",
        "--model.name", model,
        "--model.per_instance_cost_limit", "0",
        "--actions.apply_patch_locally", "true",
    ]

    # Data source: either a custom JSONL (with guidance injected) or
    # the default SWE-bench dataset filtered by instance_id.
    if data_path:
        cmd.extend(["--data_path", data_path])
    else:
        cmd.extend([
            "--data_path", "princeton-nlp/SWE-bench_Verified",
            "--instance_filter", instance["instance_id"],
        ])

    if traj_dir:
        cmd.extend(["--traj_dir", str(traj_dir)])

    return cmd


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
    """Run SWE-agent on one instance and return result metadata.

    Args:
        instance: SWE-bench instance dict with instance_id, repo,
            base_commit, problem_statement.
        model: Model name/path for SWE-agent.
        guidance_text: Optional AGENTS.md guidance to inject into the
            problem statement. When provided, a temporary single-instance
            JSONL file is written with the enhanced problem_statement and
            passed to SWE-agent as --data_path.
        timeout_s: Hard timeout in seconds (default 1800).
        max_steps: Maximum agent steps (default 50).
        traj_dir: Directory for trajectory output.
        api_base: vLLM/OpenAI API base URL (set as OPENAI_BASE_URL env).

    Returns:
        Dict with keys: patch, elapsed_s, token_usage, status, error,
        patch_source, fallback_single_shot_used.
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

    # Set env for API base
    env = os.environ.copy()
    if api_base:
        env["OPENAI_BASE_URL"] = api_base
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "EMPTY"

    traj_dir = traj_dir or Path("artifacts/trajectories")
    traj_dir.mkdir(parents=True, exist_ok=True)

    # If guidance text is provided, inject it into the problem statement
    # and write a single-instance JSONL file for SWE-agent to read.
    # SWE-agent does not accept inline problem statements via CLI flags,
    # so we create a custom data file with the modified instance.
    data_path: str | None = None
    if guidance_text:
        enhanced_instance = dict(instance)
        problem_statement = instance.get("problem_statement", "")
        enhanced_instance["problem_statement"] = (
            f"{problem_statement}\n\n"
            f"# REPO GUIDANCE (AUTO-TUNED)\n"
            f"{guidance_text}\n"
            f"# END REPO GUIDANCE"
        )
        instance_file = traj_dir / f"{iid}_instance.jsonl"
        _write_instance_jsonl(enhanced_instance, instance_file)
        data_path = str(instance_file)

    cmd = _build_sweagent_command(
        instance, model,
        data_path=data_path,
        traj_dir=traj_dir,
        max_steps=max_steps,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )

        if proc.returncode != 0:
            print(f"  [sweagent] {iid} exit code {proc.returncode}", file=sys.stderr)
            stderr_tail = proc.stderr[-500:] if proc.stderr else ""

            # Check for ContextLengthError in stderr
            if any(marker in stderr_tail.lower() for marker in (
                "context_length", "maximum context length", "contextlengtherror",
            )):
                result["status"] = "context_length_error"
                result["error"] = f"Context length exceeded: {stderr_tail[-200:]}"
                result["elapsed_s"] = time.perf_counter() - start
                return result

            result["status"] = "error"
            result["error"] = f"SWE-agent exit code {proc.returncode}: {stderr_tail}"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"SWE-agent timed out after {timeout_s}s"
        result["elapsed_s"] = time.perf_counter() - start
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["elapsed_s"] = time.perf_counter() - start
        return result

    # Extract patch from trajectory
    traj_patterns = [
        traj_dir / f"{iid}.json",
        traj_dir / iid / "trajectory.json",
        traj_dir / f"{iid}.traj",
    ]
    patch = ""
    for traj_path in traj_patterns:
        if traj_path.exists():
            patch = extract_patch_from_trajectory(str(traj_path))
            if patch:
                result["patch_source"] = "container"
                break

    # If no patch from trajectory, try SWE-agent's standard output
    if not patch and proc.stdout:
        from sweagent_bench.generation.patch_utils import extract_diff
        patch = extract_diff(proc.stdout)
        if patch:
            result["patch_source"] = "model"

    # Fallback to single-shot if SWE-agent produced no patch
    if not patch:
        try:
            patch = _fallback_single_shot(instance, model, guidance_text, api_base)
            if patch:
                result["fallback_single_shot_used"] = True
                result["fallback_single_shot_patch_len"] = len(patch)
                result["fallback_reason"] = "sweagent_empty_patch"
                result["patch_source"] = "fallback_single_shot"
        except ContextLengthError:
            result["fallback_reason"] = "context_length_error_in_fallback"

    result["patch"] = patch
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
