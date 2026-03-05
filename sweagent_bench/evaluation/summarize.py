"""Load and summarize SWE-bench evaluation results."""
from __future__ import annotations

import json
from pathlib import Path


def _candidate_eval_dirs(results_dir: Path) -> list[Path]:
    dirs = [results_dir]
    run_id = results_dir.name
    project_root = results_dir.parent.parent
    dirs.append(project_root)
    dirs.append(project_root / "logs" / "run_evaluation" / run_id)
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _extract_count(data: dict, key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return 0


def _find_harness_report_json(eval_dir: Path, run_id: str) -> Path | None:
    """Find harness report file like <model>.<run_id>.json in eval output dir."""
    try:
        candidates = sorted(eval_dir.glob(f"*{run_id}.json"))
        if candidates:
            return candidates[0]
        # fallback: any JSON file containing run_id in name
        fuzzy = sorted(p for p in eval_dir.glob("*.json") if run_id in p.name)
        if fuzzy:
            return fuzzy[0]
    except OSError:
        return None
    return None


def load_results_details(results_dir: Path) -> dict:
    """Load detailed evaluation counts from harness artifacts.

    Returns dict with keys: resolved, total, unresolved, errors, completed,
    report_path (optional), source.
    """
    run_id = results_dir.name

    for eval_dir in _candidate_eval_dirs(results_dir):
        # 1) Preferred: harness report json (<model>.<run_id>.json)
        report_json = _find_harness_report_json(eval_dir, run_id)
        if report_json and report_json.exists():
            try:
                data = json.loads(report_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    total = _extract_count(data, "submitted_instances")
                    if total <= 0:
                        total = _extract_count(data, "submitted_ids")

                    completed = _extract_count(data, "completed_instances")
                    resolved = _extract_count(data, "resolved_instances")
                    unresolved = _extract_count(data, "unresolved_instances")
                    errors = _extract_count(data, "error_instances")
                    empty_patches = _extract_count(data, "empty_patch_instances")

                    if total <= 0:
                        total = completed if completed > 0 else (resolved + unresolved + errors)
                    if completed <= 0:
                        completed = resolved + unresolved + errors

                    return {
                        "resolved": int(resolved),
                        "total": int(total),
                        "unresolved": int(unresolved),
                        "errors": int(errors),
                        "completed": int(completed),
                        "empty_patches": int(empty_patches),
                        "report_path": str(report_json),
                        "source": "harness_report_json",
                    }
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        # 2) Legacy results.json
        results_json = eval_dir / "results.json"
        if results_json.exists():
            try:
                data = json.loads(results_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    resolved_field = data.get("resolved", [])
                    applied_field = data.get("applied", [])
                    unresolved_field = data.get("unresolved", [])
                    error_field = data.get("error", [])

                    resolved_count = (
                        resolved_field if isinstance(resolved_field, int) else len(resolved_field)
                    )
                    unresolved_count = (
                        unresolved_field if isinstance(unresolved_field, int) else len(unresolved_field)
                    )
                    error_count = error_field if isinstance(error_field, int) else len(error_field)

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

                    completed = total if total > 0 else (resolved_count + unresolved_count + error_count)
                    return {
                        "resolved": int(resolved_count),
                        "total": int(total),
                        "unresolved": int(unresolved_count),
                        "errors": int(error_count),
                        "completed": int(completed),
                        "empty_patches": 0,
                        "report_path": str(results_json),
                        "source": "results_json",
                    }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # 3) instance_results fallback
        instance_results = eval_dir / "instance_results.jsonl"
        if instance_results.exists():
            resolved = 0
            total = 0
            errors = 0
            for line in instance_results.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    total += 1
                    if record.get("resolved") or record.get("passed"):
                        resolved += 1
                    elif record.get("error") or record.get("error_message"):
                        errors += 1
                except json.JSONDecodeError:
                    continue
            unresolved = max(total - resolved - errors, 0)
            return {
                "resolved": int(resolved),
                "total": int(total),
                "unresolved": int(unresolved),
                "errors": int(errors),
                "completed": int(total),
                "empty_patches": 0,
                "report_path": str(instance_results),
                "source": "instance_results_jsonl",
            }

    return {
        "resolved": 0,
        "total": 0,
        "unresolved": 0,
        "errors": 0,
        "completed": 0,
        "empty_patches": 0,
        "report_path": None,
        "source": "none",
    }


def load_results(results_dir: Path) -> tuple[int, int]:
    details = load_results_details(results_dir)
    return int(details.get("resolved", 0)), int(details.get("total", 0))


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
