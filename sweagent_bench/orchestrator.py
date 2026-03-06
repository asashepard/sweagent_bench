"""Per-repo guidance tuning orchestrator.

Supports three experimental conditions:
1. ``no_context``: agent sees only the issue + tree (baseline).
2. ``static_kb``: agent sees tree-sitter KB rendered as AGENTS.md (no LLM tuning).
3. ``oracle_tuned``: agent sees AGENTS.md refined by the LLM-as-judge oracle loop.

Adapted from context_policy/loop/orchestrator.py for SWE-agent + Qwen 3.5 35B:
- Replaces generate_patch_with_mini_swebench_result with
  generate_patch_with_sweagent from generation.sweagent_runner.
- Default timeout_s raised to 1800 (30 min).
- Default step_limit raised to 50.
- Added api_base, context_window fields to ExperimentConfig.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sweagent_bench.datasets.swebench import load_instances, read_instance_ids
from sweagent_bench.guidance.schema import RepoGuidance
from sweagent_bench.oracle.loop import run_oracle_loop
from sweagent_bench.oracle.schema import OracleConfig
from sweagent_bench.generation.patch_utils import normalize_and_validate_patch, sanitize_patch_for_preds
from sweagent_bench.generation.sweagent_runner import generate_patch_with_sweagent
from sweagent_bench.utils.jsonl import read_jsonl
from sweagent_bench.utils.paths import PREDS_DIR, PROJECT_ROOT, RESULTS_DIR
from sweagent_bench.utils.subproc import run as subproc_run


# Valid experimental conditions
VALID_CONDITIONS = ("no_context", "static_kb", "oracle_tuned")


def _elog(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [experiment] {msg}", flush=True)


# ── data classes ───────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_id: str
    model: str
    repos: list[dict]  # each: {"repo": str, "commit": str}
    oracle_model: str | None = None

    # Conditions to evaluate
    conditions: list[str] = field(default_factory=lambda: list(VALID_CONDITIONS))

    # Oracle tuning hyperparams
    oracle_iterations: int = 5
    oracle_probe_timeout_s: int = 600
    oracle_probe_max_steps: int = 25

    # Runner settings — adapted for SWE-agent + Qwen 3.5 35B
    timeout_s: int = 1800       # 30 minutes (was 600)
    step_limit: int = 50        # SWE-agent max steps (was 30)
    max_workers_gen: int = 1    # generation workers per condition (default keeps prior behavior)
    api_base: str | None = None         # vLLM API base URL
    context_window: int = 32768         # Qwen 3.5 35B context window

    # Eval settings
    eval_dataset: str = "princeton-nlp/SWE-bench_Verified"
    eval_split: str = "test"
    eval_instance_ids_file: str | None = None
    max_workers_eval: int = 4

    def __post_init__(self) -> None:
        for c in self.conditions:
            if c not in VALID_CONDITIONS:
                raise ValueError(
                    f"Unknown condition '{c}'. Valid: {VALID_CONDITIONS}"
                )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentState:
    """Persistent experiment state for resume support."""

    experiment_id: str
    created_at: str = ""
    tuning_completed: list[str] = field(default_factory=list)
    eval_completed: list[str] = field(default_factory=list)  # "<repo>__<condition>"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentState:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)


# ── run experiment ─────────────────────────────────────────────


def run_experiment(config: ExperimentConfig, *, dry_run: bool = False) -> Path:
    """Run the full tuning + evaluation experiment.

    Phase 1: For each repo, build KB and run oracle loop (if conditions require).
    Phase 2: Evaluate on SWE-bench Verified under selected conditions.

    Args:
        config: Experiment configuration.
        dry_run: If True, skip inference (produce empty patches).

    Returns:
        Path to the experiment results directory.
    """
    t_run_start = time.perf_counter()
    _elog(
        f"Starting run id={config.experiment_id} model={config.model} "
        f"conditions={config.conditions} dry_run={dry_run}"
    )

    exp_root = RESULTS_DIR / config.experiment_id
    exp_root.mkdir(parents=True, exist_ok=True)
    state_path = exp_root / "experiment_state.json"

    if state_path.exists():
        state = ExperimentState.load(state_path)
        _elog(
            "Loaded existing state "
            f"(tuning_completed={len(state.tuning_completed)}, "
            f"eval_completed={len(state.eval_completed)})"
        )
    else:
        state = ExperimentState(
            experiment_id=config.experiment_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        state.save(state_path)
        _elog("Initialized new experiment state")

    # Save config
    config_path = exp_root / "experiment_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    _elog(f"Saved experiment config to {config_path}")

    # Optional repo restriction: if eval instance IDs are provided, tune/prep only
    # repos that appear in those eval instances.
    eval_instances_prefiltered: list[dict] | None = None
    restricted_repos: set[str] | None = None
    if config.eval_instance_ids_file:
        _elog(f"Prefiltering eval instances from ids file: {config.eval_instance_ids_file}")
        prefilter_ids = read_instance_ids(config.eval_instance_ids_file)
        _elog(f"Loaded {len(prefilter_ids)} prefilter ids")
        eval_instances_prefiltered = load_instances(
            dataset_name=config.eval_dataset,
            split=config.eval_split,
            instance_ids=prefilter_ids,
        )
        _elog(f"Prefiltered eval set size: {len(eval_instances_prefiltered)}")
        restricted_repos = {inst["repo"] for inst in eval_instances_prefiltered}

        repo_order = [r["repo"] for r in config.repos]
        restricted_in_config = [r for r in repo_order if r in restricted_repos]
        skipped_count = len(repo_order) - len(restricted_in_config)
        print(
            "[experiment] eval_instance_ids provided: restricting tuning/prep "
            f"to {len(restricted_in_config)}/{len(repo_order)} repos "
            f"(skipped {skipped_count}): {', '.join(restricted_in_config)}"
        , flush=True)

    # ── Phase 1: Build KB + Oracle tuning ──────────────────────
    needs_kb = "static_kb" in config.conditions or "oracle_tuned" in config.conditions
    needs_oracle = "oracle_tuned" in config.conditions
    _elog(f"Phase 1 setup: needs_kb={needs_kb}, needs_oracle={needs_oracle}")

    guidance_map: dict[str, dict[str, RepoGuidance]] = {}

    for repo_info in config.repos:
        repo = repo_info["repo"]
        commit = repo_info["commit"]
        _elog(f"Repo loop entry: {repo}@{commit}")

        if restricted_repos is not None and repo not in restricted_repos:
            _elog(f"Skipping {repo}: not in restricted repo set")
            continue

        guidance_map.setdefault(repo, {})

        if not needs_kb:
            if repo in state.tuning_completed:
                print(f"[experiment] Skipping KB for {repo} (no KB conditions)")
            _elog(f"Skipping KB/oracle for {repo}: no KB-required conditions")
            continue

        if repo in state.tuning_completed:
            g_dir = exp_root / "guidance" / repo.replace("/", "__")
            static_path = g_dir / "versions" / "v0.json"
            best_path = g_dir / "best_guidance.json"

            if static_path.exists():
                guidance_map[repo]["static_kb"] = RepoGuidance.load(static_path)
            if best_path.exists():
                guidance_map[repo]["oracle_tuned"] = RepoGuidance.load(best_path)
            print(f"[experiment] Skipping {repo} (already done)")
            _elog(f"Loaded cached guidance for {repo} from {g_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"[experiment] Processing {repo}")
        print(f"{'='*60}")
        _elog(f"Processing repo {repo} started")

        oracle_iters = config.oracle_iterations if needs_oracle else 0
        out_dir = str(exp_root / "guidance" / repo.replace("/", "__"))
        _elog(f"Repo {repo}: oracle_iters={oracle_iters}, guidance_out={out_dir}")

        if dry_run:
            t_repo = time.perf_counter()
            g = RepoGuidance(repo=repo, commit=commit, lines=["- (dry run)"], version=0)
            g_out = Path(out_dir)
            g_out.mkdir(parents=True, exist_ok=True)
            g.save(g_out / "best_guidance.json")
            (g_out / "versions").mkdir(parents=True, exist_ok=True)
            g.save(g_out / "versions" / "v0.json")
            guidance_map[repo]["static_kb"] = g
            guidance_map[repo]["oracle_tuned"] = g
            _elog(f"Repo {repo}: dry-run guidance materialized in {time.perf_counter() - t_repo:.2f}s")
        else:
            t_repo = time.perf_counter()
            oc = OracleConfig(
                repo=repo,
                commit=commit,
                model=config.oracle_model or config.model,
                iterations=oracle_iters,
                timeout_s=config.timeout_s,
                probe_timeout_s=config.oracle_probe_timeout_s,
                probe_max_steps=config.oracle_probe_max_steps,
                api_base=config.api_base,
                output_dir=out_dir,
            )
            kb, best = run_oracle_loop(oc)
            kb_sections = sum(
                1 for section in (kb.architecture, kb.symbol_map, kb.context, kb.conventions)
                if section
            )
            _elog(
                f"Repo {repo}: oracle loop complete in {time.perf_counter() - t_repo:.2f}s "
                f"(kb_sections={kb_sections}, kb_chars={len(kb.render())}, best_version={best.version})"
            )

            v0_path = Path(out_dir) / "versions" / "v0.json"
            if v0_path.exists():
                guidance_map[repo]["static_kb"] = RepoGuidance.load(v0_path)
            else:
                guidance_map[repo]["static_kb"] = best

            guidance_map[repo]["oracle_tuned"] = best

        state.tuning_completed.append(repo)
        state.save(state_path)
        _elog(f"Repo {repo}: state persisted (tuning_completed={len(state.tuning_completed)})")

    # ── Phase 2: Verified evaluation ───────────────────────────
    print(f"\n{'='*60}")
    print(f"[experiment] Phase 2: SWE-bench Verified evaluation")
    print(f"{'='*60}")
    _elog("Phase 2 started")

    instance_ids = None
    if eval_instances_prefiltered is not None:
        eval_instances = eval_instances_prefiltered
    else:
        if config.eval_instance_ids_file:
            instance_ids = read_instance_ids(config.eval_instance_ids_file)
            _elog(f"Loaded {len(instance_ids)} eval instance IDs")

        eval_instances = load_instances(
            dataset_name=config.eval_dataset,
            split=config.eval_split,
            instance_ids=instance_ids,
        )
    _elog(f"Loaded eval instances: {len(eval_instances)}")
    print(f"  Loaded {len(eval_instances)} eval instances")

    # Group instances by repo
    instances_by_repo: dict[str, list[dict]] = {}
    for inst in eval_instances:
        instances_by_repo.setdefault(inst["repo"], []).append(inst)
    _elog(f"Grouped eval instances into {len(instances_by_repo)} repos")
    expected_instance_ids = {inst["instance_id"] for inst in eval_instances}
    expected_total = len(expected_instance_ids)

    eval_results: dict[str, dict] = {}

    for condition in config.conditions:
        t_condition = time.perf_counter()
        _elog(f"Condition start: {condition}")
        cond_preds_path = PREDS_DIR / config.experiment_id / condition / "preds.jsonl"
        cond_preds_path.parent.mkdir(parents=True, exist_ok=True)
        cond_metrics_path = exp_root / "metrics" / f"{condition}_instances.jsonl"
        cond_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        _elog(f"Condition {condition}: preds_path={cond_preds_path}")
        _elog(f"Condition {condition}: metrics_path={cond_metrics_path}")

        # Resume support
        completed_ids: set[str] = set()
        completed_metrics: dict[str, dict] = {}
        if cond_preds_path.exists():
            completed_ids = {r["instance_id"] for r in read_jsonl(cond_preds_path)}
        if cond_metrics_path.exists():
            for rec in read_jsonl(cond_metrics_path):
                completed_metrics[rec["instance_id"]] = rec
        _elog(
            f"Condition {condition}: resume loaded "
            f"completed_ids={len(completed_ids)} completed_metrics={len(completed_metrics)}"
        )

        total_instances = sum(len(v) for v in instances_by_repo.values())
        done_count = len(completed_ids)

        repo_keys_to_mark: list[str] = []
        generation_jobs: list[tuple[str, dict, str | None]] = []

        for repo, instances in instances_by_repo.items():
            key = f"{repo}__{condition}"
            if key in state.eval_completed:
                _elog(f"Condition {condition}: skipping {repo} (already completed key={key})")
                continue
            repo_keys_to_mark.append(key)

            _elog(f"Condition {condition}: repo {repo} has {len(instances)} instances")

            guidance_text = None
            if condition in ("static_kb", "oracle_tuned"):
                repo_guidance = guidance_map.get(repo, {}).get(condition)
                if repo_guidance is not None:
                    guidance_text = repo_guidance.render()
                    _elog(
                        f"Condition {condition}: guidance loaded for {repo} "
                        f"(chars={len(guidance_text)})"
                    )
                else:
                    _elog(f"Condition {condition}: no guidance found for {repo}")

            pending_instances = [
                instance for instance in instances
                if instance["instance_id"] not in completed_ids
            ]
            for i, instance in enumerate(pending_instances):
                _elog(
                    f"Condition {condition}: queue iid={instance['instance_id']} "
                    f"({i+1}/{len(pending_instances)} in repo {repo})"
                )
                generation_jobs.append((repo, instance, guidance_text))

        def _run_one_instance(
            repo: str,
            instance: dict,
            guidance_text: str | None,
        ) -> tuple[str, str, str, dict, float]:
            iid_local = instance["instance_id"]
            t_instance = time.perf_counter()
            try:
                if dry_run:
                    patch_local = ""
                    run_meta_local = {
                        "elapsed_s": 0.0,
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "status": "dry_run",
                        "error": None,
                    }
                else:
                    run_meta_local = generate_patch_with_sweagent(
                        instance=instance,
                        model=config.model,
                        guidance_text=guidance_text,
                        timeout_s=config.timeout_s,
                        max_steps=config.step_limit,
                        traj_dir=PREDS_DIR / config.experiment_id / condition / "trajectories",
                        api_base=config.api_base,
                    )
                    patch_local = run_meta_local.get("patch", "")
                    patch_local, _patch_is_noop = sanitize_patch_for_preds(patch_local)

                    normalized_patch, patch_format_error = normalize_and_validate_patch(patch_local)
                    if patch_format_error:
                        patch_local = ""
                        run_meta_local["status"] = "error"
                        run_meta_local["error"] = patch_format_error
                    else:
                        patch_local = normalized_patch
            except Exception as exc:
                patch_local = ""
                run_meta_local = {
                    "elapsed_s": 0.0,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "status": "error",
                    "error": str(exc),
                }

            wall_s = time.perf_counter() - t_instance
            return repo, iid_local, patch_local, run_meta_local, wall_s

        if config.max_workers_gen <= 1 or len(generation_jobs) <= 1:
            completed_runs = [
                _run_one_instance(repo, instance, guidance_text)
                for repo, instance, guidance_text in generation_jobs
            ]
        else:
            max_workers = min(config.max_workers_gen, len(generation_jobs))
            _elog(
                f"Condition {condition}: running generation with "
                f"max_workers_gen={max_workers} jobs={len(generation_jobs)}"
            )
            completed_runs: list[tuple[str, str, str, dict, float]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {
                    executor.submit(_run_one_instance, repo, instance, guidance_text): (repo, instance)
                    for repo, instance, guidance_text in generation_jobs
                }
                for future in as_completed(future_to_job):
                    completed_runs.append(future.result())

        for repo, iid, patch, run_meta, wall_s in completed_runs:
            if run_meta.get("status") == "error" and run_meta.get("error"):
                _elog(f"Condition {condition}: iid={iid} exception/error: {run_meta.get('error')}")
            else:
                _elog(
                    f"Condition {condition}: iid={iid} runner status={run_meta.get('status')} "
                    f"patch_source={run_meta.get('patch_source')} elapsed={run_meta.get('elapsed_s', 0.0):.2f}s"
                )
                _elog(f"Condition {condition}: iid={iid} raw patch length={len(run_meta.get('patch', '') or '')}")
                _elog(
                    f"Condition {condition}: iid={iid} sanitized patch length={len(patch)} "
                    f"no_op={not bool(patch and patch.strip())}"
                )

                record = {
                    "instance_id": iid,
                    "model_name_or_path": config.model,
                    "patch": patch,
                    "model_patch": patch,
                }
                with cond_preds_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                _elog(f"Condition {condition}: iid={iid} appended preds record")

                usage = run_meta.get("token_usage", {}) if isinstance(run_meta, dict) else {}
                metrics_record = {
                    "instance_id": iid,
                    "repo": repo,
                    "condition": condition,
                    "wall_s": float(wall_s),
                    "elapsed_s": float(run_meta.get("elapsed_s", 0.0) or 0.0),
                    "patch_non_empty": bool(patch and patch.strip()),
                    "patch_len_chars": int(len(patch or "")),
                    "status": run_meta.get("status", "ok"),
                    "error": run_meta.get("error"),
                    "patch_source": run_meta.get("patch_source", "empty"),
                    "steps_taken": int(run_meta.get("steps_taken", 0) or 0),
                    "used_max_steps": bool(int(run_meta.get("steps_taken", 0) or 0) >= int(config.step_limit)),
                    "diff_block_found": bool(run_meta.get("diff_block_found", False)),
                    "git_diff_non_empty": bool(run_meta.get("git_diff_non_empty", False)),
                    "fallback_single_shot_used": bool(run_meta.get("fallback_single_shot_used", False)),
                    "fallback_single_shot_patch_len": int(run_meta.get("fallback_single_shot_patch_len", 0) or 0),
                    "fallback_single_shot_raw_len": int(run_meta.get("fallback_single_shot_raw_len", 0) or 0),
                    "fallback_reason": run_meta.get("fallback_reason"),
                    "fallback_single_shot_truncated": bool(run_meta.get("fallback_single_shot_truncated", False)),
                    "stall_detected": bool(run_meta.get("stall_detected", False)),
                    "stall_type": run_meta.get("stall_type"),
                    "stall_action": run_meta.get("stall_action"),
                    "stall_repeat_count": int(run_meta.get("stall_repeat_count", 0) or 0),
                    "repeated_command_stall_count": int(run_meta.get("repeated_command_stall_count", 0) or 0),
                    "debug_enabled": bool(run_meta.get("debug_enabled", False)),
                    "debug_dir": run_meta.get("debug_dir"),
                    "assistant_step_artifacts": int(run_meta.get("assistant_step_artifacts", 0) or 0),
                    "bash_step_artifacts": int(run_meta.get("bash_step_artifacts", 0) or 0),
                    "no_bash_block_count": int(run_meta.get("no_bash_block_count", 0) or 0),
                    "empty_bash_block_count": int(run_meta.get("empty_bash_block_count", 0) or 0),
                    "non_actionable_reason_counts": run_meta.get(
                        "non_actionable_reason_counts",
                        {"no_bash_block": 0, "empty_bash_block": 0},
                    ),
                    "token_usage_source": str(run_meta.get("token_usage_source", "estimated") or "estimated"),
                    "reported_tokens": run_meta.get(
                        "reported_tokens",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    ),
                    "estimated_tokens": run_meta.get(
                        "estimated_tokens",
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    ),
                    "token_usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    },
                }
                with cond_metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_record, sort_keys=True, ensure_ascii=False) + "\n")
                _elog(
                    f"Condition {condition}: iid={iid} appended metrics "
                    f"status={metrics_record['status']} patch_non_empty={metrics_record['patch_non_empty']}"
                )

                done_count += 1
                completed_ids.add(iid)
                status = "OK" if patch else "EMPTY"
                print(f"  [{condition}] [{done_count}/{total_instances}] {iid} -> {status}")
                _elog(
                    f"Condition {condition}: completed iid={iid} overall_status={status} "
                    f"wall={wall_s:.2f}s"
                )

        for key in repo_keys_to_mark:
            if key not in state.eval_completed:
                state.eval_completed.append(key)
                state.save(state_path)
                _elog(f"Condition {condition}: repo state key persisted ({key})")

        # Run SWE-bench evaluation harness
        run_id = f"{config.experiment_id}__{condition}"
        eval_cmd = [
            "bash", "scripts/run_swebench_eval.sh",
            config.eval_dataset,
            str(cond_preds_path),
            run_id,
            str(config.max_workers_eval),
        ]
        logs_dir = exp_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        _elog(f"Condition {condition}: running swebench harness run_id={run_id}")

        rc = subproc_run(
            eval_cmd,
            cwd=PROJECT_ROOT,
            stdout_path=logs_dir / f"eval_{condition}.stdout.log",
            stderr_path=logs_dir / f"eval_{condition}.stderr.log",
            timeout_s=3600,
        )
        _elog(f"Condition {condition}: swebench harness exit_code={rc}")
        if rc != 0:
            print(f"  WARNING: eval harness failed for {condition}")

        generation_metrics = _collect_condition_generation_stats(
            cond_metrics_path,
            cond_preds_path,
            expected_instance_ids,
        )
        _elog(
            f"Condition {condition}: generation summary "
            f"patch_non_empty={generation_metrics['patch_non_empty']} "
            f"attempted={generation_metrics['attempted']} "
            f"errors={generation_metrics['error_count']}"
        )

        resolved = 0
        load_error = None
        try:
            from sweagent_bench.evaluation.summarize import compute_rate, load_results

            resolved, _harness_total = load_results(RESULTS_DIR / run_id)
            rate = compute_rate(resolved, expected_total)
            _elog(
                f"Condition {condition}: evaluation results loaded resolved={resolved} "
                f"total={expected_total} rate={rate:.4f}"
            )
        except Exception as exc:
            print(f"  WARNING: could not load results for {condition}: {exc}")
            load_error = str(exc)
            rate = 0.0
            _elog(f"Condition {condition}: load_results failed: {exc}")

        eval_results[condition] = {
            "run_id": run_id,
            "resolved": resolved,
            "total": expected_total,
            "attempted": expected_total,
            "rate": rate,
            "preds_path": str(cond_preds_path),
            "instance_metrics_path": str(cond_metrics_path),
            "generation_metrics": generation_metrics,
        }
        if load_error is not None:
            eval_results[condition]["error"] = load_error

        _elog(f"Condition complete: {condition} in {time.perf_counter() - t_condition:.2f}s")

    # ── Summary ────────────────────────────────────────────────
    summary = {
        "experiment_id": config.experiment_id,
        "model": config.model,
        "repos": [r["repo"] for r in config.repos],
        "conditions": config.conditions,
        "oracle_iterations": config.oracle_iterations,
        "eval_results": eval_results,
    }

    # Compute deltas between conditions
    nc = eval_results.get("no_context", {})
    sk = eval_results.get("static_kb", {})
    ot = eval_results.get("oracle_tuned", {})
    deltas: dict[str, dict] = {}
    if "rate" in nc and "rate" in sk:
        deltas["static_kb_vs_no_context"] = {
            "absolute": sk["rate"] - nc["rate"],
            "no_context_rate": nc["rate"],
            "static_kb_rate": sk["rate"],
        }
    if "rate" in nc and "rate" in ot:
        deltas["oracle_tuned_vs_no_context"] = {
            "absolute": ot["rate"] - nc["rate"],
            "no_context_rate": nc["rate"],
            "oracle_tuned_rate": ot["rate"],
        }
    if "rate" in sk and "rate" in ot:
        deltas["oracle_tuned_vs_static_kb"] = {
            "absolute": ot["rate"] - sk["rate"],
            "static_kb_rate": sk["rate"],
            "oracle_tuned_rate": ot["rate"],
        }
    if deltas:
        summary["deltas"] = deltas

    summary_path = exp_root / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[experiment] Summary written to {summary_path}")
    print(json.dumps(summary, indent=2))
    _elog(
        f"Run complete in {time.perf_counter() - t_run_start:.2f}s; "
        f"summary_path={summary_path}"
    )

    return exp_root


# ── stats helper ───────────────────────────────────────────────


def _collect_condition_generation_stats(
    metrics_path: Path,
    preds_path: Path,
    target_ids: set[str],
) -> dict:
    """Aggregate generation metrics for one condition."""
    metrics_by_id: dict[str, dict] = {}
    if metrics_path.exists():
        for rec in read_jsonl(metrics_path):
            iid = rec.get("instance_id")
            if iid in target_ids:
                metrics_by_id[iid] = rec

    preds_by_id: dict[str, dict] = {}
    if preds_path.exists():
        for rec in read_jsonl(preds_path):
            iid = rec.get("instance_id")
            if iid in target_ids:
                preds_by_id[iid] = rec

    attempted = len(target_ids)
    patch_non_empty = 0
    empty_patch_count = 0
    missing_image_count = 0
    error_count = 0
    fallback_single_shot_used_count = 0
    fallback_single_shot_success_count = 0
    stalled_repeat_failure_count = 0
    patch_source_counts: dict[str, int] = {
        "container": 0,
        "model": 0,
        "fallback_single_shot": 0,
        "empty": 0,
    }
    elapsed_s = 0.0
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    token_usage_by_source = {
        "reported": 0,
        "estimated": 0,
    }
    patch_len_total = 0
    patch_len_non_empty_total = 0
    patch_len_max = 0
    steps_total = 0
    steps_for_patch_total = 0
    steps_for_patch_count = 0
    used_max_steps_count = 0
    diff_block_found_count = 0
    git_diff_non_empty_count = 0
    no_bash_block_total = 0
    empty_bash_block_total = 0
    wall_s_total = 0.0
    wall_to_elapsed_ratio_sum = 0.0
    wall_to_elapsed_ratio_count = 0
    token_per_elapsed_instance_count = 0
    tokens_per_second_sum = 0.0

    for iid in target_ids:
        metric = metrics_by_id.get(iid)
        pred = preds_by_id.get(iid, {})
        patch_text = pred.get("model_patch", "") if isinstance(pred, dict) else ""

        if metric is not None:
            patch_non_empty_i = bool(metric.get("patch_non_empty"))
            metric_elapsed_s = float(metric.get("elapsed_s", 0.0) or 0.0)
            elapsed_s += metric_elapsed_s
            usage = metric.get("token_usage", {}) if isinstance(metric, dict) else {}
            token_total_i = int(usage.get("total_tokens", 0) or 0)
            token_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            token_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            token_usage["total_tokens"] += token_total_i

            token_source = str(metric.get("token_usage_source", "") or "")
            if token_source in token_usage_by_source:
                token_usage_by_source[token_source] += 1

            patch_len_i = int(metric.get("patch_len_chars", 0) or 0)
            patch_len_total += patch_len_i
            patch_len_max = max(patch_len_max, patch_len_i)

            steps_i = int(metric.get("steps_taken", 0) or 0)
            steps_total += steps_i
            if patch_non_empty_i:
                steps_for_patch_total += steps_i
                steps_for_patch_count += 1

            if bool(metric.get("used_max_steps", False)):
                used_max_steps_count += 1
            if bool(metric.get("diff_block_found", False)):
                diff_block_found_count += 1
            if bool(metric.get("git_diff_non_empty", False)):
                git_diff_non_empty_count += 1

            no_bash_block_total += int(metric.get("no_bash_block_count", 0) or 0)
            empty_bash_block_total += int(metric.get("empty_bash_block_count", 0) or 0)

            wall_s_i = float(metric.get("wall_s", 0.0) or 0.0)
            wall_s_total += wall_s_i
            if wall_s_i > 0.0 and metric_elapsed_s > 0.0:
                wall_to_elapsed_ratio_sum += wall_s_i / metric_elapsed_s
                wall_to_elapsed_ratio_count += 1

            if metric_elapsed_s > 0.0:
                tokens_per_second_sum += token_total_i / metric_elapsed_s
                token_per_elapsed_instance_count += 1

            status = str(metric.get("status", "") or "")
            if status == "missing_image":
                missing_image_count += 1
            elif status == "error":
                error_count += 1
            elif not patch_non_empty_i:
                empty_patch_count += 1

            if bool(metric.get("fallback_single_shot_used", False)):
                fallback_single_shot_used_count += 1
            if int(metric.get("fallback_single_shot_patch_len", 0) or 0) > 0:
                fallback_single_shot_success_count += 1
            if str(metric.get("stall_type", "") or "") == "repeat_failed_action":
                stalled_repeat_failure_count += 1

            patch_source = str(metric.get("patch_source", "") or "")
            if patch_source in patch_source_counts:
                patch_source_counts[patch_source] += 1
            elif patch_source:
                patch_source_counts[patch_source] = patch_source_counts.get(patch_source, 0) + 1
        else:
            patch_non_empty_i = bool(patch_text and str(patch_text).strip())
            if not patch_non_empty_i:
                empty_patch_count += 1
            patch_source_counts["model" if patch_non_empty_i else "empty"] += 1
            patch_len_i = len(str(patch_text or ""))
            patch_len_total += patch_len_i
            patch_len_max = max(patch_len_max, patch_len_i)

        if patch_non_empty_i:
            patch_non_empty += 1
            if metric is not None:
                patch_len_non_empty_total += int(metric.get("patch_len_chars", 0) or 0)
            else:
                patch_len_non_empty_total += len(str(patch_text or ""))

    return {
        "attempted": attempted,
        "instances_processed": attempted,
        "patch_non_empty": patch_non_empty,
        "patch_non_empty_rate": (patch_non_empty / attempted) if attempted else 0.0,
        "avg_patch_len_chars_all": (patch_len_total / attempted) if attempted else 0.0,
        "avg_patch_len_chars_non_empty": (
            patch_len_non_empty_total / patch_non_empty
        ) if patch_non_empty else 0.0,
        "max_patch_len_chars": patch_len_max,
        "empty_patch_count": empty_patch_count,
        "missing_image_count": missing_image_count,
        "error_count": error_count,
        "fallback_single_shot_used_count": fallback_single_shot_used_count,
        "fallback_single_shot_success_count": fallback_single_shot_success_count,
        "stalled_repeat_failure_count": stalled_repeat_failure_count,
        "steps_total": steps_total,
        "avg_steps_per_instance": (steps_total / attempted) if attempted else 0.0,
        "avg_steps_to_non_empty_patch": (
            steps_for_patch_total / steps_for_patch_count
        ) if steps_for_patch_count else 0.0,
        "used_max_steps_count": used_max_steps_count,
        "used_max_steps_rate": (used_max_steps_count / attempted) if attempted else 0.0,
        "diff_block_found_count": diff_block_found_count,
        "git_diff_non_empty_count": git_diff_non_empty_count,
        "no_bash_block_total": no_bash_block_total,
        "empty_bash_block_total": empty_bash_block_total,
        "patch_source_counts": patch_source_counts,
        "elapsed_s": elapsed_s,
        "mean_elapsed_s": (elapsed_s / attempted) if attempted else 0.0,
        "wall_s_total": wall_s_total,
        "mean_wall_s": (wall_s_total / attempted) if attempted else 0.0,
        "mean_wall_to_elapsed_ratio": (
            wall_to_elapsed_ratio_sum / wall_to_elapsed_ratio_count
        ) if wall_to_elapsed_ratio_count else 0.0,
        "token_usage": token_usage,
        "token_usage_by_source": token_usage_by_source,
        "tokens_per_second": (
            tokens_per_second_sum / token_per_elapsed_instance_count
        ) if token_per_elapsed_instance_count else 0.0,
        "tokens_per_non_empty_patch": (
            token_usage["total_tokens"] / patch_non_empty
        ) if patch_non_empty else 0.0,
    }
