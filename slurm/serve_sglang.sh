#!/usr/bin/env bash
# Serve Qwen 3.5 35B via SGLang on gpmoo-a1.
#
# SLURM job — submit with: sbatch slurm/serve_sglang.sh
#
#SBATCH --job-name=sglang-qwen35
#SBATCH --partition=gpmoo-a
#SBATCH --nodelist=gpmoo-a1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sglang_%j.out
#SBATCH --error=logs/sglang_%j.err

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TP:-1}"

echo "Starting SGLang server..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Max model len: $MAX_MODEL_LEN"
echo "  GPU util: $GPU_UTIL"
echo "  Tensor parallel: $TP"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$GPU_UTIL" \
    --tp-size "$TP" \
    --trust-remote-code \
    --disable-log-requests
