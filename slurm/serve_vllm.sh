#!/usr/bin/env bash
# Serve Qwen 3.5 35B via vLLM on gpmoo-a1.
#
# SLURM job — submit with: sbatch slurm/serve_vllm.sh
#
#SBATCH --job-name=vllm-qwen35
#SBATCH --partition=gpmoo-a
#SBATCH --nodelist=gpmoo-a1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TP:-1}"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Max model len: $MAX_MODEL_LEN"
echo "  GPU util: $GPU_UTIL"
echo "  Tensor parallel: $TP"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --disable-log-requests
