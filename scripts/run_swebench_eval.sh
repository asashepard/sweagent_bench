#!/usr/bin/env bash
# Run SWE-bench evaluation harness with logging.
# Usage: run_swebench_eval.sh DATASET_NAME PREDS_PATH RUN_ID [MAX_WORKERS]

set -euo pipefail

DATASET_NAME="${1:?Error: DATASET_NAME required}"
PREDS_PATH="${2:?Error: PREDS_PATH required}"
RUN_ID="${3:?Error: RUN_ID required}"
MAX_WORKERS="${4:-1}"

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/results/$RUN_ID"
mkdir -p "$RESULTS_DIR"

# Build command
CMD="python -m swebench.harness.run_evaluation \
    --dataset_name $DATASET_NAME \
    --predictions_path $PREDS_PATH \
    --max_workers $MAX_WORKERS \
    --run_id $RUN_ID"

# Write command to file
echo "$CMD" > "$RESULTS_DIR/cmd.txt"
echo "Command saved to: $RESULTS_DIR/cmd.txt"

# Run with tee to capture stdout/stderr
echo "Running SWE-bench evaluation..."
echo "Results will be saved to: $RESULTS_DIR"

# Execute and capture outputs
set +e  # Don't exit on error, we want to capture the exit code
python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$PREDS_PATH" \
    --max_workers "$MAX_WORKERS" \
    --run_id "$RUN_ID" \
    --report_dir "$RESULTS_DIR" \
    > >(tee "$RESULTS_DIR/stdout.log") \
    2> >(tee "$RESULTS_DIR/stderr.log" >&2)
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "WARNING: evaluation failed (exit $EXIT_CODE). Dumping --help for debugging:"
    python -m swebench.harness.run_evaluation --help 2>&1 | head -40 || true
fi

echo ""
echo "Evaluation complete. Exit code: $EXIT_CODE"
echo "Logs: $RESULTS_DIR/stdout.log, $RESULTS_DIR/stderr.log"

exit $EXIT_CODE
