#!/usr/bin/env bash
# Serve a model via SGLang with an OpenAI-compatible API.
#
# Usage: sbatch slurm/serve_sglang.sh
#
#SBATCH --job-name=sglang-serve
#SBATCH --partition=PARTITION        # <-- set your SLURM partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=7-00:00
#SBATCH --output=logs/sglang_%j.out
#SBATCH --error=logs/sglang_%j.err

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-8000}"
CONTEXT_LEN="${CONTEXT_LEN:-16384}"
MEM_FRAC="${MEM_FRAC:-0.90}"

echo "Starting SGLang server at $(date)"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Context length: $CONTEXT_LEN"
echo "  Node: $(hostname)"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size 1 \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRAC"
