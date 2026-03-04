#!/usr/bin/env python3
"""Ensure SWE-bench Docker images exist for a set of instances.

Deterministic flow:
1) Resolve expected image name per instance via SWE-bench helpers.
2) Pull missing images (or repull when --force is set).
3) Verify each required image with `docker image inspect`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sweagent_bench.datasets.swebench import load_instances, read_instance_ids


def _list_local_images() -> list[str]:
    """Return locally available Docker images as repo:tag strings."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def _docker_image_exists(image: str) -> bool:
    """Check whether a specific Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode == 0
    except Exception:
        return False


def _fallback_image_from_local(instance_id: str, local_images: list[str]) -> str | None:
    """Fallback image resolution by matching short instance ID in local images."""
    short_id = instance_id.split("__")[-1]
    for img in local_images:
        if short_id in img and "sweb.eval" in img:
            return img
    return None


def _normalize_image_tag(image: str | None) -> str | None:
    if not image:
        return None
    image = image.strip()
    if not image:
        return None
    if ":" not in image.rsplit("/", 1)[-1]:
        return f"{image}:latest"
    return image


def _derive_image_from_instance_id(instance_id: str) -> str | None:
    """Derive SWE-bench image from instance_id using observed naming convention."""
    if "__" not in instance_id:
        return None
    repo, short = instance_id.split("__", 1)
    if not repo or not short:
        return None
    return f"swebench/sweb.eval.x86_64.{repo}_1776_{short}:latest"


def _resolve_instance_image(instance: dict, local_images: list[str]) -> tuple[str | None, str]:
    """Resolve expected image name for a SWE-bench instance.

    Priority:
    1) swebench.harness.docker_utils.get_instance_docker_image
    2) swebench.harness.test_spec.make_test_spec(...).instance_image_key
    3) derive from instance_id naming convention
    4) fallback search in local images by short instance id
    """
    instance_id = instance["instance_id"]

    try:
        from swebench.harness.docker_utils import get_instance_docker_image

        image = _normalize_image_tag(get_instance_docker_image(instance))
        if image:
            return image, "helper:get_instance_docker_image"
    except Exception:
        pass

    try:
        from swebench.harness.test_spec import make_test_spec

        spec = make_test_spec(instance)
        image = _normalize_image_tag(getattr(spec, "instance_image_key", None))
        if image:
            return image, "helper:make_test_spec.instance_image_key"
    except Exception:
        pass

    derived = _derive_image_from_instance_id(instance_id)
    if derived:
        return derived, "derived:instance_id"

    fallback = _fallback_image_from_local(instance_id, local_images)
    if fallback:
        return fallback, "fallback:local_image_scan"

    return None, "unresolved"


def _resolve_images(instances: list[dict]) -> tuple[dict[str, str | None], dict[str, str]]:
    """Resolve image names and methods for all instances."""
    local_images = _list_local_images()
    image_mapping: dict[str, str | None] = {}
    method_mapping: dict[str, str] = {}
    for inst in instances:
        iid = inst["instance_id"]
        image, method = _resolve_instance_image(inst, local_images)
        image_mapping[iid] = image
        method_mapping[iid] = method
    return image_mapping, method_mapping


def _pull_image(image: str) -> bool:
    """Pull image from registry."""
    print(f"  Pulling: {image}")
    try:
        result = subprocess.run(["docker", "pull", image])
        return result.returncode == 0
    except Exception as exc:
        print(f"  ERROR pull {image}: {exc}")
        return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ensure SWE-bench Docker images exist.")
    parser.add_argument(
        "--instance_ids_file",
        required=True,
        help="Path to file with instance IDs (one per line).",
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Unused for now; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repull images even if they already exist locally.",
    )
    args = parser.parse_args()

    instance_ids = read_instance_ids(args.instance_ids_file)
    if not instance_ids:
        print("No instance IDs found.")
        return 1
    print(f"Loaded {len(instance_ids)} instance IDs from {args.instance_ids_file}")

    print("Loading instance data from dataset...")
    instances = load_instances(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=instance_ids,
    )
    if not instances:
        print("No matching instances found in dataset.")
        return 1

    print("Resolving expected image names...")
    resolved, resolve_methods = _resolve_images(instances)
    print("Resolution methods:")
    for inst in instances:
        iid = inst["instance_id"]
        image = resolved[iid] or "UNRESOLVED"
        method = resolve_methods.get(iid, "unknown")
        print(f"  - {iid}: {method} -> {image}")

    to_pull: list[tuple[str, str]] = []
    already_present: list[tuple[str, str]] = []
    for inst in instances:
        iid = inst["instance_id"]
        image = resolved[iid]
        if image is None:
            continue
        exists = _docker_image_exists(image)
        if args.force or not exists:
            to_pull.append((iid, image))
        else:
            already_present.append((iid, image))

    if already_present:
        print(f"Already present: {len(already_present)} images")
        for iid, image in already_present:
            print(f"  OK {iid} -> {image}")

    if to_pull:
        print(f"Pulling {len(to_pull)} image(s)...")
        pull_failures: list[tuple[str, str]] = []
        for iid, image in to_pull:
            ok = _pull_image(image)
            if not ok:
                pull_failures.append((iid, image))

        if pull_failures:
            print("\nWARNING: Pull failed for some images:")
            for iid, image in pull_failures:
                print(f"  FAIL {iid} -> {image}")
    else:
        print("All required images are already present. Use --force to repull.")

    print("\nVerifying required images...")
    ok = 0
    fail = 0
    missing_details: list[tuple[str, str]] = []
    for inst in instances:
        iid = inst["instance_id"]
        image = resolved[iid]
        if image and _docker_image_exists(image):
            print(f"  OK {iid} -> {image}")
            ok += 1
        else:
            missing_image = image or "UNRESOLVED"
            print(f"  FAIL {iid} -> {missing_image}")
            missing_details.append((iid, missing_image))
            fail += 1

    print(f"\nResult: {ok} present, {fail} missing")
    if fail:
        print("Missing required images:")
        for iid, image in missing_details:
            print(f"  - {iid} -> {image}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
