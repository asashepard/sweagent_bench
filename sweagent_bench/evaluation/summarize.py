"""Load and summarize SWE-bench evaluation results."""
from __future__ import annotations

import json
from pathlib import Path


def _candidate_eval_dirs(results_dir: Path) -> list[Path]:
    dirs = [results_dir]
    run_id = results_dir.name
    project_root = results_dir.parent.parent
    dirs.append(project_root / "logs" / "run_evaluation" / run_id)
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_results(results_dir: Path) -> tuple[int, int]:
    for eval_dir in _candidate_eval_dirs(results_dir):
        results_json = eval_dir / "results.json"
        if results_json.exists():
            try:
                data = json.loads(results_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    resolved_field = data.get("resolved", [])
                    applied_field = data.get("applied", [])
                    resolved_count = (
                        resolved_field if isinstance(resolved_field, int) else len(resolved_field)
                    )
                    if isinstance(applied_field, int):
                        total = applied_field
                    elif applied_field:
                        total = len(applied_field)
                    else:
                        all_ids = set()
                        for key in ["resolved", "applied", "failed", "error", "unresolved"]:
                            value = data.get(key, [])
                            if isinstance(value, list):
                                all_ids.update(value)
                        total = len(all_ids) if all_ids else resolved_count
                    return int(resolved_count), int(total)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        instance_results = eval_dir / "instance_results.jsonl"
        if instance_results.exists():
            resolved = 0
            total = 0
            for line in instance_results.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    total += 1
                    if record.get("resolved") or record.get("passed"):
                        resolved += 1
                except json.JSONDecodeError:
                    continue
            return resolved, total

    return 0, 0


def compute_rate(resolved: int, total: int) -> float:
    return resolved / total if total > 0 else 0.0


def load_instance_records(results_dir: Path) -> list[dict]:
    for eval_dir in _candidate_eval_dirs(results_dir):
        instance_results = eval_dir / "instance_results.jsonl"
        if not instance_results.exists():
            continue
        records: list[dict] = []
        for raw_line in instance_results.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
    return []


def classify_failure(record: dict) -> str:
    if record.get("resolved") or record.get("passed"):
        return "resolved"
    text_parts = []
    for key in ["error", "error_message", "failure_reason", "report", "status"]:
        value = record.get(key)
        if value:
            text_parts.append(str(value).lower())
    text = " ".join(text_parts)
    if "timeout" in text:
        return "timeout"
    if "apply" in text or "patch" in text:
        return "patch_apply_failure"
    if "importerror" in text or "module" in text or "environment" in text:
        return "environment_failure"
    if "test" in text or "assert" in text or "fail" in text:
        return "test_failure"
    if "error" in text or "exception" in text:
        return "runtime_error"
    return "unresolved_unknown"


def summarize_failure_taxonomy(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        category = classify_failure(record)
        counts[category] = counts.get(category, 0) + 1
    return counts
