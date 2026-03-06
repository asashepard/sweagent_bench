#!/usr/bin/env bash
# Run a sweagent-bench experiment end-to-end.
#
# Usage:
#   ./run_experiment.sh [--dry-run] [--conditions COND...] [--ids IDS_FILE]
#
# Required env vars:
#   OPENAI_BASE_URL  — vLLM endpoint (e.g. http://gpmoo-a1:8000/v1)
#
# Optional env vars:
#   MODEL            — model name (default: Qwen/Qwen3.5-35B)
#   EXPERIMENT_ID    — experiment identifier (default: auto-generated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3.5-35B}"
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
CONDITIONS="${CONDITIONS:-no_context static_kb oracle_tuned}"
IDS_FILE="${IDS_FILE:-ids/verified_mini_ids.txt}"
TIMEOUT="${TIMEOUT:-1800}"
STEP_LIMIT="${STEP_LIMIT:-50}"
MAX_WORKERS_GEN="${MAX_WORKERS_GEN:-1}"
ORACLE_ITERS="${ORACLE_ITERS:-5}"
ORACLE_PROBE_TIMEOUT="${ORACLE_PROBE_TIMEOUT:-600}"
ORACLE_PROBE_MAX_STEPS="${ORACLE_PROBE_MAX_STEPS:-16}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-8}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"

# Parse CLI overrides
DRY_RUN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --conditions)
            shift
            CONDITIONS=""
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                CONDITIONS="$CONDITIONS $1"
                shift
            done
            CONDITIONS="${CONDITIONS# }"  # strip leading space
            ;;
        --ids) IDS_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Verify environment ────────────────────────────────────────
echo "================================================"
echo "sweagent-bench experiment"
echo "================================================"
echo "  Model:         $MODEL"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Conditions:    $CONDITIONS"
echo "  Instance IDs:  $IDS_FILE"
echo "  Timeout:       ${TIMEOUT}s"
echo "  Step limit:    $STEP_LIMIT"
echo "  Gen workers:   $MAX_WORKERS_GEN"
echo "  Oracle probe timeout: ${ORACLE_PROBE_TIMEOUT}s"
echo "  Oracle probe max steps: $ORACLE_PROBE_MAX_STEPS"
echo "  Eval workers:  $MAX_WORKERS_EVAL"
echo "  Skip preflight: $SKIP_PREFLIGHT"
echo "  API base:      ${OPENAI_BASE_URL:-http://localhost:8000/v1}"
echo "================================================"

# Activate virtualenv if present
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

# Set dummy API key for vLLM
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

SKIP_PREFLIGHT_FLAG=""
if [[ "$SKIP_PREFLIGHT" == "1" ]]; then
    SKIP_PREFLIGHT_FLAG="--skip-preflight"
fi

# ── Run ───────────────────────────────────────────────────────
python -m sweagent_bench.main \
    --model "$MODEL" \
    --experiment-id "$EXPERIMENT_ID" \
    --conditions $CONDITIONS \
    --instance-ids-file "$IDS_FILE" \
    --repos-config configs/repos_12.json \
    --oracle-iterations "$ORACLE_ITERS" \
    --oracle-probe-timeout "$ORACLE_PROBE_TIMEOUT" \
    --oracle-probe-max-steps "$ORACLE_PROBE_MAX_STEPS" \
    --max-workers-gen "$MAX_WORKERS_GEN" \
    --max-workers-eval "$MAX_WORKERS_EVAL" \
    --timeout "$TIMEOUT" \
    --step-limit "$STEP_LIMIT" \
    $SKIP_PREFLIGHT_FLAG \
    $DRY_RUN

echo ""
echo "Experiment complete: $EXPERIMENT_ID"
echo "Results: results/$EXPERIMENT_ID/"
