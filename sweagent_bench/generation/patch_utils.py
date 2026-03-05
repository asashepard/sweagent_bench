"""Utility functions for extracting patches from model/agent output."""
from __future__ import annotations

import json
import re
from collections import Counter

MAX_PATCH_SIZE = 200_000
_BOGUS_INDEX_RE = re.compile(r"^index [0-9a-f]{7,}\.\.[0-9a-f]{7,}(\s+\d+)?$")


def normalize_patch_text(patch: str) -> str:
    """Normalize patch text for stable application by the evaluator.

    - Convert CRLF/CR to LF
    - Ensure non-empty patch ends with exactly one trailing newline
    """
    if not patch:
        return ""
    normalized = patch.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return ""
    return normalized.rstrip("\n") + "\n"


def extract_unified_diff(patch_text: str) -> str:
    """Extract only the unified-diff portion from model output.

    Rules:
    - Start at first ``diff --git`` or ``--- a/`` line.
    - Keep contiguous diff blocks, including multi-file patches.
    - After at least one hunk starts, stop at the first clearly non-diff line.
    - Remove hallucinated ``index <hash>..<hash> [mode]`` lines.
    - Return text ending with exactly one trailing newline.
    """
    if not patch_text:
        return ""

    text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    start_index = None
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- a/"):
            start_index = idx
            break
    if start_index is None:
        return ""

    collected: list[str] = []
    saw_hunk = False
    in_hunk = False

    for line in lines[start_index:]:
        if _BOGUS_INDEX_RE.match(line):
            continue

        if line.startswith("diff --git "):
            collected.append(line)
            in_hunk = False
            continue

        if line.startswith("--- a/") or line.startswith("+++ b/"):
            collected.append(line)
            in_hunk = False
            continue

        if re.match(r"^@@\s+.*\s+@@", line):
            collected.append(line)
            saw_hunk = True
            in_hunk = True
            continue

        if line.startswith((
            "old mode ",
            "new mode ",
            "deleted file mode ",
            "new file mode ",
            "similarity index ",
            "rename from ",
            "rename to ",
            "copy from ",
            "copy to ",
            "Binary files ",
            "GIT binary patch",
        )):
            collected.append(line)
            in_hunk = False
            continue

        if in_hunk:
            if line.startswith((" ", "+", "-", "\\")):
                collected.append(line)
                continue
            break

        if saw_hunk and not line.startswith(("diff --git ", "--- a/")):
            break

        if line == "":
            if saw_hunk:
                break
            continue

        # Non-diff material before any hunk: keep extraction bounded.
        break

    if not collected:
        return ""

    out = "\n".join(collected).strip("\n")
    if not out:
        return ""
    return out + "\n"


def _validate_diff_git_sections(patch: str) -> tuple[bool, str | None]:
    section_starts = [m.start() for m in re.finditer(r"(?m)^diff --git\s+", patch)]
    if not section_starts:
        return True, None

    section_starts.append(len(patch))
    for idx in range(len(section_starts) - 1):
        start = section_starts[idx]
        end = section_starts[idx + 1]
        section = patch[start:end]
        header = section.splitlines()[0] if section.splitlines() else f"section {idx + 1}"

        has_old = bool(re.search(r"(?m)^---\s+a/", section))
        has_new = bool(re.search(r"(?m)^\+\+\+\s+b/", section))
        has_hunk = bool(re.search(r"(?m)^@@\s+.*\s+@@", section))

        if not has_old:
            return False, f"missing old file header in section {idx + 1}: {header}"
        if not has_new:
            return False, f"missing new file header in section {idx + 1}: {header}"
        if not has_hunk:
            return False, f"missing hunk header in section {idx + 1}: {header}"

    return True, None


def validate_diff_format(patch: str) -> tuple[bool, str | None]:
    """Validate that patch looks like a minimally valid unified diff.

    Requirements for non-empty patch:
    - contains file headers (--- a/ and +++ b/)
    - contains at least one hunk header (@@ ... @@)
    """
    if not patch or not patch.strip():
        return True, None

    if re.search(r"(?m)^diff --git\s+", patch):
        return _validate_diff_git_sections(patch)

    has_file_headers = bool(
        re.search(r"(?m)^---\s+a/", patch) and re.search(r"(?m)^\+\+\+\s+b/", patch)
    )
    has_hunk = bool(re.search(r"(?m)^@@\s+.*\s+@@", patch))

    if not has_file_headers or not has_hunk:
        return False, "invalid diff format"

    return True, None


def normalize_and_validate_patch(patch: str) -> tuple[str, str | None]:
    """Normalize patch and return an error if diff format is invalid."""
    normalized = normalize_patch_text(patch)
    extracted = extract_unified_diff(normalized)

    if normalized and not extracted:
        return "", "invalid diff format: no unified diff found"

    normalized = normalize_patch_text(extracted)
    ok, err = validate_diff_format(normalized)
    if not ok:
        return "", err
    return normalized, None


def extract_diff(text: str) -> str:
    """Extract a unified diff from *text*.

    Accepts fenced blocks tagged ``diff``, ``patch``, or untagged.
    When multiple candidate blocks exist the **last** one wins (the model
    typically refines its answer across turns).
    """
    # 1. Fenced blocks: ```diff, ```patch, or plain ```
    fence_pattern = r"```(?:diff|patch)?\s*\n(.*?)```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    # Walk backwards so the *last* qualifying block wins.
    if matches:
        for match in reversed(matches):
            if "---" in match or "diff --git" in match:
                return extract_unified_diff(match)

    # 2. Raw unfenced diff starting with "diff --git"
    lines = text.split("\n")
    last_raw_start = None
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            last_raw_start = i
    if last_raw_start is not None:
        return extract_unified_diff("\n".join(lines[last_raw_start:]))

    # 3. Fallback: raw "--- " header
    for i, line in enumerate(lines):
        if line.startswith("--- "):
            return extract_unified_diff("\n".join(lines[i:]))

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
    sanitized = extract_unified_diff(candidate)
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
