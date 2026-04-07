#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/svijsy46/vllm_%j.log

# Ensure all caching successfully routes memory to scratch
SCRATCH_DIR="/scratch/$USER"
export HF_HOME="$SCRATCH_DIR/huggingface"
export XDG_CACHE_HOME="$SCRATCH_DIR/xdg"

# Activate the scratch-hosted virtual environment
source "$SCRATCH_DIR/phase2-venv/bin/activate"

echo "Starting vLLM server..."

# Spin up the vLLM OpenAI-Compatible API Server on the A100.
# You can change the model array or quantize if running the massive 32B models.
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dtype auto \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --port 8000
