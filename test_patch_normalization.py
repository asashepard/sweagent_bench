"""Smoke tests for patch normalization/validation helpers."""
from sweagent_bench.generation.patch_utils import (
    normalize_and_validate_patch,
    normalize_patch_text,
    validate_diff_format,
)


def test_missing_trailing_newline_is_fixed() -> None:
    raw = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
    normalized = normalize_patch_text(raw)
    assert normalized.endswith("\n")
    assert not normalized.endswith("\n\n")
    ok, err = validate_diff_format(normalized)
    assert ok and err is None


def test_crlf_is_normalized_to_lf() -> None:
    raw = "diff --git a/a.py b/a.py\r\n--- a/a.py\r\n+++ b/a.py\r\n@@ -1 +1 @@\r\n-old\r\n+new\r\n"
    normalized = normalize_patch_text(raw)
    assert "\r" not in normalized
    assert normalized.endswith("\n")


def test_empty_patch_remains_empty() -> None:
    normalized, err = normalize_and_validate_patch("")
    assert normalized == ""
    assert err is None


def test_random_text_fails_validation() -> None:
    normalized, err = normalize_and_validate_patch("hello world\nthis is not a diff")
    assert normalized == ""
    assert err == "invalid diff format"


if __name__ == "__main__":
    test_missing_trailing_newline_is_fixed()
    test_crlf_is_normalized_to_lf()
    test_empty_patch_remains_empty()
    test_random_text_fails_validation()
    print("test_patch_normalization: OK")
