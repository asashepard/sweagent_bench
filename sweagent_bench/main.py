"""CLI entry point for sweagent-bench experiments.

Usage:
    python -m sweagent_bench.main \
        --model Qwen/Qwen3.5-35B \
        --experiment-id exp_qwen35_v1 \
        --conditions no_context static_kb oracle_tuned \
        --instance-ids-file ids/verified_mini_ids.txt \
        --repos-config configs/repos_12.json

Environment variables:
    OPENAI_BASE_URL  — vLLM endpoint (default: http://localhost:8000/v1)
    OPENAI_API_KEY   — set to 'EMPTY' for vLLM (default: EMPTY)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sweagent_bench.orchestrator import ExperimentConfig, run_experiment
from sweagent_bench.preflight import run_preflight
from sweagent_bench.utils.paths import PROJECT_ROOT


def _load_repos_config(path: str) -> list[dict]:
    """Load repos configuration from JSON file."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "repos" in data:
        return data["repos"]
    raise ValueError(f"Cannot parse repos config from {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweagent-bench",
        description="Run SWE-bench experiments with SWE-agent + context guidance",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name/path (e.g. Qwen/Qwen3.5-35B)",
    )
    parser.add_argument(
        "--experiment-id", required=True,
        help="Unique experiment identifier (used for result dirs)",
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["no_context", "static_kb", "oracle_tuned"],
        choices=["no_context", "static_kb", "oracle_tuned"],
        help="Conditions to evaluate (default: all three)",
    )
    parser.add_argument(
        "--instance-ids-file",
        help="Path to text file with instance IDs (one per line)",
    )
    parser.add_argument(
        "--repos-config", default="configs/repos_12.json",
        help="Path to repos JSON config (default: configs/repos_12.json)",
    )
    parser.add_argument(
        "--oracle-model",
        help="Model for oracle loop (default: same as --model)",
    )
    parser.add_argument(
        "--oracle-iterations", type=int, default=5,
        help="Oracle tuning iterations (default: 5)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-instance timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--step-limit", type=int, default=50,
        help="Max SWE-agent steps per instance (default: 50)",
    )
    parser.add_argument(
        "--api-base",
        help="vLLM API base URL (default: $OPENAI_BASE_URL or http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--context-window", type=int, default=32768,
        help="Model context window size (default: 32768)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip inference, produce empty patches (for testing)",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight checks",
    )
    parser.add_argument(
        "--max-workers-eval", type=int, default=4,
        help="Parallel workers for SWE-bench eval harness (default: 4)",
    )

    args = parser.parse_args(argv)

    # Resolve API base
    api_base = args.api_base or os.environ.get(
        "OPENAI_BASE_URL", "http://localhost:8000/v1"
    )

    # Ensure OPENAI_API_KEY is set (vLLM doesn't need a real key)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    # Pre-flight checks
    if not args.skip_preflight and not args.dry_run:
        print("[main] Running pre-flight checks...")
        if not run_preflight(api_base=api_base):
            print("[main] Pre-flight checks FAILED. Use --skip-preflight to bypass.", file=sys.stderr)
            return 1
        print("[main] Pre-flight checks passed.\n")

    # Load repos config
    repos = _load_repos_config(args.repos_config)

    # Build experiment config
    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        model=args.model,
        repos=repos,
        oracle_model=args.oracle_model,
        conditions=args.conditions,
        oracle_iterations=args.oracle_iterations,
        timeout_s=args.timeout,
        step_limit=args.step_limit,
        api_base=api_base,
        context_window=args.context_window,
        eval_instance_ids_file=args.instance_ids_file,
        max_workers_eval=args.max_workers_eval,
    )

    print(f"[main] Experiment: {config.experiment_id}")
    print(f"[main] Model: {config.model}")
    print(f"[main] Conditions: {config.conditions}")
    print(f"[main] API base: {api_base}")
    print(f"[main] Timeout: {config.timeout_s}s, Steps: {config.step_limit}")
    print(f"[main] Repos: {len(repos)}")
    if args.instance_ids_file:
        print(f"[main] Instance IDs: {args.instance_ids_file}")
    print()

    result_dir = run_experiment(config, dry_run=args.dry_run)
    print(f"\n[main] Results: {result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
