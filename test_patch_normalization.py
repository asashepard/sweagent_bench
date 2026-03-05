"""Smoke tests for patch normalization/validation helpers."""
from sweagent_bench.generation.patch_utils import (
    extract_unified_diff,
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
    assert err == "invalid diff format: no unified diff found"


def test_extraction_strips_trailing_commentary() -> None:
    raw = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "I need to look at the actual Django code...\n"
    )
    extracted = extract_unified_diff(raw)
    assert "Django code" not in extracted
    ok, err = validate_diff_format(extracted)
    assert ok and err is None


def test_bogus_index_line_removed() -> None:
    raw = (
        "diff --git a/a.py b/a.py\n"
        "index 1234567..abcdef0 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    extracted = extract_unified_diff(raw)
    assert "index 1234567..abcdef0 100644" not in extracted
    ok, err = validate_diff_format(extracted)
    assert ok and err is None


def test_multisection_validation_reports_section_error() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "+added_without_hunk\n"
    )
    normalized, err = normalize_and_validate_patch(patch)
    assert normalized == ""
    assert err is not None and "missing hunk header in section 2" in err


if __name__ == "__main__":
    test_missing_trailing_newline_is_fixed()
    test_crlf_is_normalized_to_lf()
    test_empty_patch_remains_empty()
    test_random_text_fails_validation()
    test_extraction_strips_trailing_commentary()
    test_bogus_index_line_removed()
    test_multisection_validation_reports_section_error()
    print("test_patch_normalization: OK")
