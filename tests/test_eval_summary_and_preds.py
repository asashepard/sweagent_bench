"""Smoke tests for preds record writing and harness summary parsing."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sweagent_bench.evaluation.summarize import load_results_details
from sweagent_bench.orchestrator import _build_preds_record


def test_preds_record_includes_patch_and_model_patch() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    rec = _build_preds_record("iid-1", "Qwen/Qwen3.5-35B-A3B", patch)
    assert rec["patch"] == patch
    assert rec["model_patch"] == patch


def test_load_results_details_reads_instances_schema_from_project_root() -> None:
    run_id = "exp123__oracle_tuned"
    with TemporaryDirectory() as td:
        project_root = Path(td)
        results_dir = project_root / "results" / run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "submitted_instances": 10,
            "completed_instances": 9,
            "resolved_instances": 4,
            "unresolved_instances": 3,
            "error_instances": 2,
            "empty_patch_instances": 1,
        }
        report_path = project_root / f"Qwen__Qwen3.5-35B-A3B.{run_id}.json"
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

        details = load_results_details(results_dir)

        assert details["total"] == 10
        assert details["completed"] == 9
        assert details["resolved"] == 4
        assert details["unresolved"] == 3
        assert details["errors"] == 2
        assert details["empty_patches"] == 1
        assert details["source"] == "harness_report_json"
        assert details["report_path"] == str(report_path)


if __name__ == "__main__":
    test_preds_record_includes_patch_and_model_patch()
    test_load_results_details_reads_instances_schema_from_project_root()
    print("test_eval_summary_and_preds: OK")
