#!/usr/bin/env bash
# Run sweagent-bench experiment via SLURM.
#
# Submit with: sbatch slurm/run_experiment.sh
#
#SBATCH --job-name=sweagent-bench
#SBATCH --partition=gpmoo-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
export MODEL="${MODEL:-Qwen/Qwen3.5-35B}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-exp_slurm_$(date +%Y%m%d_%H%M%S)}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://gpmoo-a1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CONDITIONS="${CONDITIONS:-no_context static_kb oracle_tuned}"
export IDS_FILE="${IDS_FILE:-ids/verified_mini_ids.txt}"
export TIMEOUT="${TIMEOUT:-1800}"
export STEP_LIMIT="${STEP_LIMIT:-50}"
export MAX_WORKERS_GEN="${MAX_WORKERS_GEN:-1}"
export ORACLE_ITERS="${ORACLE_ITERS:-5}"
export ORACLE_PROBE_TIMEOUT="${ORACLE_PROBE_TIMEOUT:-600}"
export ORACLE_PROBE_MAX_STEPS="${ORACLE_PROBE_MAX_STEPS:-25}"
export MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-8}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

mkdir -p logs

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Experiment: $EXPERIMENT_ID"
echo "Model: $MODEL"
echo "API: $OPENAI_BASE_URL"
echo "Conditions: $CONDITIONS"
echo "IDs file: $IDS_FILE"
echo "Timeout/steps: ${TIMEOUT}s / $STEP_LIMIT"
echo "Gen workers: $MAX_WORKERS_GEN"
echo "Oracle iters: $ORACLE_ITERS"
echo "Oracle probe timeout/steps: ${ORACLE_PROBE_TIMEOUT}s / $ORACLE_PROBE_MAX_STEPS"
echo "Eval workers: $MAX_WORKERS_EVAL"
echo "Skip preflight: $SKIP_PREFLIGHT"

# Run experiment
bash scripts/run_experiment.sh

echo "SLURM job complete."
