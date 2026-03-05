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
- `steps_total`, `steps_actionable`, `steps_non_actionable`
- `elapsed_s_total`, `elapsed_s_llm`, `elapsed_s_tools`, `elapsed_s_eval`
- `chars_prompt`, `chars_completion`
- `token_usage_source` (`reported` or `estimated`)
- `token_usage`, `reported_tokens`, `estimated_tokens`

`experiment_summary.json` includes condition-level aggregates for the same step/time/token fields
under each condition's `generation_metrics`.
