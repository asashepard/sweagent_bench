"""Quick smoke tests for patch extraction logic."""
from sweagent_bench.generation.sweagent_runner import (
    _extract_last_diff_block,
    _strip_think_markers,
    _parse_fenced_blocks,
)
from sweagent_bench.generation.patch_utils import extract_diff


def test_diff_fence():
    text = (
        "Some prose\n"
        "```diff\n"
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
        "More prose"
    )
    d = _extract_last_diff_block(text)
    assert "diff --git" in d, f"FAIL: {d!r}"
    print("PASS: diff fence")


def test_patch_fence():
    text = (
        "```patch\n"
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    d = _extract_last_diff_block(text)
    assert "diff --git" in d, f"FAIL: {d!r}"
    print("PASS: patch fence")


def test_last_block_wins():
    text = (
        "```diff\n"
        "--- a/old.py\n"
        "+++ b/old.py\n"
        "-wrong\n"
        "+wrong_fix\n"
        "```\n"
        "\n"
        "Actually let me fix:\n"
        "\n"
        "```diff\n"
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+correct_fix\n"
        "```\n"
    )
    d = _extract_last_diff_block(text)
    assert "correct_fix" in d, f"FAIL: {d!r}"
    print("PASS: last diff block wins")


def test_think_tags():
    text = (
        "<think>Let me think...\n"
        "```diff\n"
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
        "```\n"
        "</think>"
    )
    cleaned = _strip_think_markers(text)
    d = _extract_last_diff_block(cleaned)
    assert "diff --git" in d, f"FAIL: {d!r}"
    print("PASS: think tags stripped, content kept")


def test_raw_unfenced():
    text = (
        "Here is the fix:\n"
        "\n"
        "diff --git a/bar.py b/bar.py\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    d = _extract_last_diff_block(text)
    assert "diff --git" in d, f"FAIL: {d!r}"
    print("PASS: raw unfenced diff")


def test_extract_diff_patch_fence():
    text = (
        "```patch\n"
        "diff --git a/z.py b/z.py\n"
        "--- a/z.py\n"
        "+++ b/z.py\n"
        "@@ -1 +1 @@\n"
        "-p\n"
        "+q\n"
        "```"
    )
    d = extract_diff(text)
    assert "diff --git" in d, f"FAIL: {d!r}"
    print("PASS: extract_diff handles patch fence")


def test_extract_diff_last_wins():
    text = (
        "```diff\n"
        "--- a/old.py\n"
        "+++ b/old.py\n"
        "-first\n"
        "+first_fix\n"
        "```\n"
        "```diff\n"
        "--- a/new.py\n"
        "+++ b/new.py\n"
        "-second\n"
        "+second_fix\n"
        "```\n"
    )
    d = extract_diff(text)
    assert "second_fix" in d, f"FAIL: {d!r}"
    print("PASS: extract_diff last match wins")


def test_mixed_bash_and_diff():
    """Model outputs bash exploration then a diff — diff should be captured."""
    text = (
        "Let me check:\n"
        "```bash\n"
        "grep -rn 'def foo' src/\n"
        "```\n"
        "\n"
        "I see the issue. Here's the fix:\n"
        "```diff\n"
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -5,3 +5,3 @@\n"
        "-    return None\n"
        "+    return 42\n"
        "```\n"
    )
    d = _extract_last_diff_block(text)
    assert "return 42" in d, f"FAIL: {d!r}"
    print("PASS: mixed bash + diff — diff captured")


if __name__ == "__main__":
    test_diff_fence()
    test_patch_fence()
    test_last_block_wins()
    test_think_tags()
    test_raw_unfenced()
    test_extract_diff_patch_fence()
    test_extract_diff_last_wins()
    test_mixed_bash_and_diff()
    print("\nAll 8 extraction tests PASSED")
