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

# Run experiment
bash scripts/run_experiment.sh

echo "SLURM job complete."
