#!/usr/bin/env bash
# Debug script — run vLLM interactively to see all output.
#
#SBATCH --job-name=debug-vllm
#SBATCH --partition=gpmoo-a
#SBATCH --nodelist=gpmoo-a1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

export PYTHONUNBUFFERED=1

echo "=== Debug info ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Which python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "nvidia-smi:"
nvidia-smi 2>&1 || echo "nvidia-smi failed"
echo ""

echo "=== Testing vllm import ==="
python -c "import vllm; print('vllm version:', vllm.__version__)" 2>&1
echo "vllm import exit code: $?"
echo ""

echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    2>&1

echo "=== vLLM exited with code: $? ==="
