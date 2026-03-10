#!/usr/bin/env bash
# Debug script — run SGLang interactively to see all output.
#
#SBATCH --job-name=debug-sglang
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
echo ""

echo "=== Testing sglang import ==="
python -c "import sglang; print('sglang version:', sglang.__version__)" 2>&1
echo "sglang import exit code: $?"
echo ""

echo "=== Starting SGLang server ==="
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8001 \
    --context-length 16384 \
    --mem-fraction-static 0.90 \
    --tp-size 1 \
    --trust-remote-code \
    2>&1

echo "=== SGLang exited with code: $? ==="
