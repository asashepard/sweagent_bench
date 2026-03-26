#!/usr/bin/env bash
#SBATCH --job-name=reeval-oracle-200steps
#SBATCH --partition=gpmoo-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/27as66/sweagent_bench/logs/reeval_oracle_200steps_%j.out
#SBATCH --error=/shared/27as66/sweagent_bench/logs/reeval_oracle_200steps_%j.err

set -euo pipefail

export PATH="/shared/27as66/.conda/envs/sweagent/bin:$PATH"

PREDS="/shared/27as66/sweagent_bench/artifacts/preds/verified_full500_200steps_20260315_230142/oracle_tuned/preds.jsonl"
RUN_ID="verified_full500_200steps_20260315_230142__oracle_tuned"

echo "Node: $(hostname)"
echo "Run ID: $RUN_ID"
echo "Preds: $PREDS"
echo "Start: $(date)"

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path "$PREDS" \
    --max_workers 8 \
    --run_id "$RUN_ID"

echo "Done: $(date)"
