#!/usr/bin/env bash
#SBATCH -J sglang-qwen35b
#SBATCH -p gpmoo-a
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 4-00:00
#SBATCH -o /shared/27as66/sweagent_bench/logs/sglang-%j.out
#SBATCH -e /shared/27as66/sweagent_bench/logs/sglang-%j.err

set -euo pipefail
unset PYTHONPATH
export PYTHONPATH="/shared/27as66/.conda/envs/sweagent-bench-py311/lib/python3.11/site-packages"

# Instead of conda activate (which hangs), manually set the env
export PATH="/shared/27as66/.conda/envs/sweagent/bin:$PATH"
export CONDA_DEFAULT_ENV=sweagent
export CONDA_PREFIX=/shared/27as66/.conda/envs/sweagent

cd /shared/27as66/sweagent_bench

export HF_HOME=/shared/27as66/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="Qwen/Qwen3.5-35B-A3B"
PORT="${PORT:-8000}"
CONTEXT_LEN=16384

echo "MODEL_ID=$MODEL_ID"
echo "PORT=$PORT"
echo "CONTEXT_LEN=$CONTEXT_LEN"
echo "HOSTNAME=$(hostname)"
echo "PYTHON=$(which python)"
python --version

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size 1 \
  --context-length "$CONTEXT_LEN" \
  --mem-fraction-static 0.90
