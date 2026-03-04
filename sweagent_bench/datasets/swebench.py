"""SWE-bench dataset loading utilities."""
from __future__ import annotations

import json
from pathlib import Path


def read_instance_ids(path: str | Path) -> list[str]:
    path = Path(path)
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def load_instances(
    dataset_name: str,
    split: str,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
    tasks_file: str | Path | None = None,
) -> list[dict]:
    if tasks_file is not None:
        instances = load_instances_from_tasks_file(tasks_file)
        if instance_ids is not None:
            id_set = set(instance_ids)
            instances = [row for row in instances if row["instance_id"] in id_set]
        if limit is not None:
            instances = instances[:limit]
        return instances

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    if instance_ids is not None:
        id_set = set(instance_ids)
        ds = ds.filter(lambda x: x["instance_id"] in id_set)

    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))

    instances = []
    for row in ds:
        inst = {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        }
        if "version" in row:
            inst["version"] = row["version"]
        if "environment_setup_commit" in row:
            inst["environment_setup_commit"] = row["environment_setup_commit"]
        instances.append(inst)

    return instances


def _normalize_instance_row(row: dict) -> dict:
    instance_id = row.get("instance_id") or row.get("id")
    repo = row.get("repo") or row.get("repository")
    base_commit = row.get("base_commit") or row.get("commit") or row.get("base_sha")
    problem_statement = (
        row.get("problem_statement")
        or row.get("issue")
        or row.get("problem")
        or row.get("prompt")
        or ""
    )

    if not instance_id or not repo or not base_commit:
        raise ValueError(
            "Task row missing required fields: instance_id/id, repo/repository, base_commit/commit/base_sha"
        )

    normalized = {
        "instance_id": str(instance_id),
        "repo": str(repo),
        "base_commit": str(base_commit),
        "problem_statement": str(problem_statement),
    }

    if row.get("version") is not None:
        normalized["version"] = row["version"]
    if row.get("environment_setup_commit") is not None:
        normalized["environment_setup_commit"] = row["environment_setup_commit"]
    return normalized


def load_instances_from_tasks_file(path: str | Path) -> list[dict]:
    tasks_path = Path(path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    if tasks_path.suffix.lower() == ".jsonl":
        rows = []
        for raw_line in tasks_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    else:
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "tasks" in payload:
            rows = payload["tasks"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unsupported tasks payload. Expected list or object with 'tasks'.")

    instances = [_normalize_instance_row(row) for row in rows]
    return instances
