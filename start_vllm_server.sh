#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/vllm_%j.log

# Ensure all caching successfully routes memory to scratch
SCRATCH_DIR="/scratch/$USER"
export HF_HOME="$SCRATCH_DIR/huggingface"
export XDG_CACHE_HOME="$SCRATCH_DIR/xdg"

# Safely activate conda inside SLURM batch jobs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$SCRATCH_DIR/phase2-conda-env"

echo "Starting vLLM server..."

# Spin up the vLLM OpenAI-Compatible API Server on the A100.
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dtype auto \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --port 8000
