"""Utility functions for extracting patches from model/agent output."""
from __future__ import annotations

import json
import re
from collections import Counter

MAX_PATCH_SIZE = 200_000


def extract_diff(text: str) -> str:
    fence_pattern = r"```(?:diff)?\s*\n(.*?)```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            if "---" in match or "diff --git" in match:
                return match.strip()

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            return "\n".join(lines[i:]).strip()

    for i, line in enumerate(lines):
        if line.startswith("--- "):
            return "\n".join(lines[i:]).strip()

    return ""


def _strip_to_first_fenced_diff_block(text: str) -> str:
    if "```" not in text:
        return text
    fence_pattern = r"```[^\n]*\n(.*?)```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for block in matches:
        for line in block.splitlines():
            if line.startswith("diff --git ") or line.startswith("--- a/"):
                return block.strip()
    return text.replace("```", "")


def _slice_from_first_diff_start(text: str) -> str:
    lines = text.splitlines()
    start_index = None
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- a/"):
            start_index = idx
            break
    if start_index is None:
        return ""
    return "\n".join(lines[start_index:]).strip()


def _is_noop_diff(patch: str) -> bool:
    if not patch or not patch.strip():
        return True
    minus_lines: list[str] = []
    plus_lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("-"):
            minus_lines.append(line[1:])
        elif line.startswith("+"):
            plus_lines.append(line[1:])
    if not minus_lines and not plus_lines:
        return True
    return Counter(minus_lines) == Counter(plus_lines)


def sanitize_patch_for_preds(patch: str) -> tuple[str, bool]:
    candidate = _strip_to_first_fenced_diff_block(patch or "")
    sanitized = _slice_from_first_diff_start(candidate)
    is_noop = _is_noop_diff(sanitized)
    if is_noop:
        return "", True
    return sanitized, False


def extract_patch_from_trajectory(traj_path: str) -> str:
    """Extract the final patch from a SWE-agent trajectory JSON file.

    SWE-agent writes trajectory files that contain the agent's actions
    and the final patch. This function reads the trajectory and extracts
    the patch.

    Args:
        traj_path: Path to the trajectory JSON file.

    Returns:
        Extracted unified diff patch string, or empty string.
    """
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""

    # SWE-agent trajectory format: look for "patch" or "model_patch" key
    if isinstance(traj, dict):
        # Direct patch key
        for key in ("patch", "model_patch", "diff"):
            if key in traj and isinstance(traj[key], str) and traj[key].strip():
                return traj[key].strip()

        # Check info dict
        info = traj.get("info", {})
        if isinstance(info, dict):
            for key in ("patch", "model_patch", "submission"):
                if key in info and isinstance(info[key], str) and info[key].strip():
                    return info[key].strip()

        # Scan trajectory actions for the last edit/patch output
        trajectory = traj.get("trajectory", [])
        if isinstance(trajectory, list):
            for step in reversed(trajectory):
                if isinstance(step, dict):
                    obs = step.get("observation", "")
                    if isinstance(obs, str) and "diff --git" in obs:
                        return extract_diff(obs)

    return ""
