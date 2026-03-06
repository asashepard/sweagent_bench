# sweagent-bench

SWE-agent + Qwen 3.5 35B + vLLM + SWE-bench harness experiment pipeline.

Three experimental conditions:
- **no_context**: agent sees only the issue + file tree
- **static_kb**: agent sees issue + tree + deterministic AGENTS.md from tree-sitter KB
- **oracle_tuned**: agent sees issue + tree + LLM-as-judge refined AGENTS.md

## Quick Start

```bash
pip install -e .

# Set the vLLM endpoint (required)
export OPENAI_BASE_URL=http://gpmoo-a1:8000/v1

# Run with defaults (50 verified instances, all 3 conditions)
bash scripts/run_experiment.sh

# Or override via env vars / CLI flags:
IDS_FILE=ids/verified_smoke_4_ids.txt \
  bash scripts/run_experiment.sh --conditions no_context static_kb oracle_tuned
```

## Serve vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <checkpoint> \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000
```

## SLURM Batch Runs

Use the provided SLURM scripts with env-var overrides for clean, repeatable runs.

```bash
# Submit vLLM server job
sbatch slurm/serve_vllm.sh

# Submit experiment job (all defaults)
sbatch slurm/run_experiment.sh

# Submit experiment with overrides
MODEL=Qwen/Qwen3.5-35B \
OPENAI_BASE_URL=http://gpmoo-a1:8000/v1 \
EXPERIMENT_ID=exp_oracle_runner_$(date +%Y%m%d_%H%M%S) \
CONDITIONS="no_context static_kb oracle_tuned" \
IDS_FILE=ids/verified_smoke_4_ids.txt \
ORACLE_ITERS=3 \
ORACLE_PROBE_TIMEOUT=180 \
TIMEOUT=1800 \
STEP_LIMIT=50 \
sbatch slurm/run_experiment.sh
```

`slurm/run_experiment.sh` forwards those values to `scripts/run_experiment.sh`,
which passes `--oracle-probe-timeout` to the CLI.

## Output

```
results/<exp_id>/
  experiment_config.json
  experiment_state.json
  experiment_summary.json
  guidance/<repo>/...
  metrics/
  logs/
```

`metrics/<condition>_instances.jsonl` records per-instance runner accounting, including:
- `steps_taken`, `elapsed_s`, `wall_s`, `patch_len_chars`
- `patch_source`, `fallback_single_shot_used`, `fallback_reason`
- `stall_type`, `stall_repeat_count`, `no_bash_block_count`, `empty_bash_block_count`
- `token_usage_source` (`reported` or `estimated`)
- `token_usage`, `reported_tokens`, `estimated_tokens`

`experiment_summary.json` includes condition-level aggregates under each condition's
`generation_metrics`, including patch yield/rates, patch-size averages, step efficiency,
fallback usage, token totals/rates, and patch-source distribution.
