#!/usr/bin/env bash
#SBATCH --job-name=sweagent-bench
#SBATCH --partition=gpmoo-a
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/27as66/sweagent_bench/logs/experiment_%j.out
#SBATCH --error=/shared/27as66/sweagent_bench/logs/experiment_%j.err
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3.5-35B}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-exp_slurm_$(date +%Y%m%d_%H%M%S)}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://gpmoo-a1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CONDITIONS="${CONDITIONS:-no_context static_kb oracle_tuned}"
export IDS_FILE="${IDS_FILE:-ids/verified_full_ids.txt}"
export TIMEOUT="${TIMEOUT:-1800}"
export STEP_LIMIT="${STEP_LIMIT:-100}"
export MAX_WORKERS_GEN="${MAX_WORKERS_GEN:-4}"
export ORACLE_ITERS="${ORACLE_ITERS:-5}"
export ORACLE_PROBE_TIMEOUT="${ORACLE_PROBE_TIMEOUT:-600}"
export MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-8}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"

PROJECT_ROOT="/shared/27as66/sweagent_bench"
cd "$PROJECT_ROOT"

if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

mkdir -p "$PROJECT_ROOT/logs"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Experiment: $EXPERIMENT_ID"
echo "Model: $MODEL"
echo "API: $OPENAI_BASE_URL"
echo "Conditions: $CONDITIONS"
echo "IDs file: $IDS_FILE"
echo "Timeout/steps: ${TIMEOUT}s / $STEP_LIMIT"
echo "Gen workers: $MAX_WORKERS_GEN"
echo "Eval workers: $MAX_WORKERS_EVAL"

bash scripts/run_experiment.sh

echo "SLURM job complete."
