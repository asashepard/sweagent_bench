#!/usr/bin/env bash
# Debug script — try both vLLM and SGLang to see which works.
#
#SBATCH --job-name=debug-serve
#SBATCH --partition=gpmoo-a
#SBATCH --nodelist=gpmoo-a1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

export PYTHONUNBUFFERED=1

# Activate conda env (non-interactive SLURM jobs don't source .bashrc)
CONDA_ENV="${CONDA_ENV:-sweagent}"
if [ -f /shared/bin/anaconda3/etc/profile.d/conda.sh ]; then
    source /shared/bin/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

echo "=== Debug info ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Which python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "nvidia-smi (short):"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
echo ""

echo "=== Torch CUDA check ==="
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
" 2>&1
echo "torch check exit code: $?"
echo ""

echo "=== Testing vllm import ==="
python -c "import vllm; print('vllm version:', vllm.__version__)" 2>&1
VLLM_OK=$?
echo "vllm import exit code: $VLLM_OK"
echo ""

if [ $VLLM_OK -eq 0 ]; then
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
else
    echo "=== vLLM not available, trying SGLang ==="
    python -c "import sglang; print('sglang version:', sglang.__version__)" 2>&1
    echo "sglang import exit code: $?"

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
fi
