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
export OPENAI_BASE_URL=http://gpmoo-a1:8001/v1

# Run with defaults (50 verified instances, all 3 conditions)
bash scripts/run_experiment.sh

# Or override via env vars / CLI flags:
IDS_FILE=ids/verified_smoke_4_ids.txt \
  bash scripts/run_experiment.sh --conditions no_context static_kb oracle_tuned
```

## Serve vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-35B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8001
```

## Serve SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B \
  --context-length 16384 \
  --mem-fraction-static 0.90 \
  --host 0.0.0.0 --port 8001
```

## SLURM Batch Runs

Use the provided SLURM scripts with env-var overrides for clean, repeatable runs.

```bash
# Submit vLLM server job
sbatch slurm/serve_vllm.sh

# Or SGLang server job
sbatch slurm/serve_sglang.sh

# Submit experiment job (all defaults)
sbatch slurm/run_experiment.sh

# Submit experiment with overrides
MODEL=Qwen/Qwen3.5-35B \
OPENAI_BASE_URL=http://gpmoo-a1:8001/v1 \
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

## Resume / Run a Single Condition

The orchestrator saves state (`experiment_state.json`) after each repo's tuning and
each condition's evaluation. To resume or run only one condition against an existing
experiment (e.g. reuse already-tuned repos and existing preds):

```bash
# Resume with just oracle_tuned on an existing experiment
bash scripts/run_experiment.sh \
  --experiment-id exp_existing_20260301_120000 \
  --conditions oracle_tuned

# Or equivalently via env vars
EXPERIMENT_ID=exp_existing_20260301_120000 \
CONDITIONS="oracle_tuned" \
  bash scripts/run_experiment.sh
```

The orchestrator will:
1. Load the saved `experiment_state.json` (skip repos whose tuning is already done).
2. Load existing preds from `artifacts/preds/<exp_id>/<condition>/preds.jsonl`.
3. Only generate patches for instances not yet completed.
4. Run eval only for the requested condition(s).

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
