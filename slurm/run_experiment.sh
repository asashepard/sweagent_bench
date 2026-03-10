#!/usr/bin/env bash
# Run sweagent-bench experiment via SLURM.
#
# Submit with: sbatch slurm/run_experiment.sh
#
#SBATCH --job-name=sweagent-bench
#SBATCH --partition=gpmoo-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

# Activate conda env (non-interactive SLURM jobs don't source .bashrc)
CONDA_ENV="${CONDA_ENV:-sweagent}"
if [ -f /shared/bin/anaconda3/etc/profile.d/conda.sh ]; then
    source /shared/bin/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

# ── Configuration ─────────────────────────────────────────────
export MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-exp_slurm_$(date +%Y%m%d_%H%M%S)}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://gpmoo-a1:8001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CONDITIONS="${CONDITIONS:-no_context static_kb oracle_tuned}"
export IDS_FILE="${IDS_FILE:-ids/verified_mini_ids.txt}"
export TIMEOUT="${TIMEOUT:-1800}"
export STEP_LIMIT="${STEP_LIMIT:-50}"
export MAX_WORKERS_GEN="${MAX_WORKERS_GEN:-1}"
export ORACLE_ITERS="${ORACLE_ITERS:-5}"
export ORACLE_PROBE_TIMEOUT="${ORACLE_PROBE_TIMEOUT:-600}"
export MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-4}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"\n\nmkdir -p logs

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
echo "Oracle probe mode/timeout: single_shot / ${ORACLE_PROBE_TIMEOUT}s"
echo "Eval workers: $MAX_WORKERS_EVAL"
echo "Skip preflight: $SKIP_PREFLIGHT"

# Run experiment
bash scripts/run_experiment.sh

echo "SLURM job complete."
