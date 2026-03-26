#!/usr/bin/env bash
#SBATCH -J swebench-resume-oracle
#SBATCH -p gpmoo-a
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 4-00:00
#SBATCH -o /shared/27as66/sweagent_bench/logs/resume_oracle_%j.out
#SBATCH -e /shared/27as66/sweagent_bench/logs/resume_oracle_%j.err

set -euo pipefail

source /shared/bin/anaconda3/etc/profile.d/conda.sh
conda activate sweagent-bench-py311

cd /shared/27as66/sweagent_bench

export MODEL="Qwen/Qwen3.5-35B-A3B"
export EXPERIMENT_ID="verified_full500_20260307_170937"
export OPENAI_BASE_URL="http://gpmoo-a1:8000/v1"
export OPENAI_API_KEY=EMPTY
export CONDITIONS="oracle_tuned"
export IDS_FILE="/shared/27as66/sweagent_bench/ids/verified_full_ids.txt"

export ORACLE_ITERS=5
export SKIP_PREFLIGHT=1

curl -sS "$OPENAI_BASE_URL/models"
echo

bash scripts/run_experiment.sh
