#!/usr/bin/env bash
#SBATCH -J sglang-qwen35b
#SBATCH -p gpmoo-a
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=96G
#SBATCH -t 7-00:00
#SBATCH -o /shared/27as66/sweagent_bench/logs/sglang-%j.out
#SBATCH -e /shared/27as66/sweagent_bench/logs/sglang-%j.err

export HOME=/shared/27as66
export PATH="/shared/27as66/.conda/envs/sweagent-bench-py311/bin:$PATH"
export CONDA_PREFIX="/shared/27as66/.conda/envs/sweagent-bench-py311"
export HF_HOME=/shared/27as66/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
unset PYTHONPATH

cd /shared/27as66/sweagent_bench

MODEL_ID="Qwen/Qwen3.5-35B-A3B"
PORT=8000
CONTEXT_LEN=16384

echo "starting at $(date)"
echo "HOSTNAME=$(hostname)"
echo "MODEL_ID=$MODEL_ID PORT=$PORT"
echo "Python: $(which python)"

python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size 1 \
  --context-length "$CONTEXT_LEN" \
  --mem-fraction-static 0.90
