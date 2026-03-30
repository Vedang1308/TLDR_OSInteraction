#!/bin/bash
# ==============================================================================
# Phase 1 - Agent Safety Baseline (AgentHARM)
# Target: Intel Gaudi HPU / SLURM Environment (SOL Supercomputer)
# ==============================================================================

echo "=== INITIALIZING AGENTHARM SAFETY BENCHMARK ON SOL ==="

# 1. Ensure Python 3.10+ Environment is active
# (inspect_ai requires Py3.10+, which SOL modules provide)
if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "[FATAL ERROR] You must run this in a Python 3.10+ Conda/Module environment!"
  echo "Try running: module load python/3.10 (or activate your conda env)"
  exit 1
fi

# 2. Install the UK Safety Institute Framework natively
echo "[1/3] Installing UK AISI framework dependencies..."
python3 -m pip install "inspect-ai>=0.3" "inspect-evals" "datasets" "vllm" --quiet

# Choose your model via command line argument (e.g. Qwen/Qwen3-VL-2B-Instruct)
MODEL_ID=$1

if [ -z "$MODEL_ID" ]; then
    echo "[FATAL ERROR] You must provide a model to evaluate!"
    echo "Usage: ./run_agentharm.sh <model_id>"
    echo "Example: ./run_agentharm.sh Qwen/Qwen3-VL-2B-Instruct"
    exit 1
fi

# Dynamically route the port based on string hashing or just default to 8000 safely
PORT=8000

echo "[2/3] Booting Local vLLM Inference Server ($MODEL_ID) on port $PORT..."
echo "      (This translates Qwen's hardware inferences into an OpenAI-compatible API stream)"

# We start the vLLM OpenAI-compatible server in the background
# NOTE: Set tensor-parallel-size if running the 8B model across multiple GPUs/HPUs
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 4096 \
    > vllm_server.log 2>&1 &
    
VLLM_PID=$!

# Wait for server to physically bind and load model weights into HPU memory
echo "      Waiting for model weights to load into HPU memory (Check vllm_server.log for details)..."
while ! curl -s http://localhost:$PORT/v1/models >/dev/null; do
    sleep 5
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[FATAL ERROR] vLLM Server crashed during loading. See vllm_server.log"
        exit 1
    fi
done
echo "      vLLM API Server is online!"

# 3. Execute the safety evaluation against your offline inference node
echo "[3/3] Triggering full AgentHARM execution loop..."

# Format model name exactly like OmniACT pipeline: "Qwen_Qwen3-VL-2B-Instruct"
SAFE_MODEL_NAME="${MODEL_ID//\//_}"
RESULTS_DIR="results/$SAFE_MODEL_NAME/agentharm"

# Ensure clean directory exists
mkdir -p "$RESULTS_DIR"

# Map the generic OpenAI provider syntax natively to our local vLLM pipeline
export INSPECT_EVAL_MODEL="openai/$MODEL_ID"
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="sk-mock-key"

# We pass the eval command using inspect-evals repository defaults for agentharm
inspect eval inspect_evals/agentharm \
    --model $INSPECT_EVAL_MODEL \
    --log-dir "$RESULTS_DIR"

# 4. Clean up HPC resources
echo "=== Benchmarking Complete ==="
echo "Shutting down vLLM Inference Server (PID $VLLM_PID)..."
kill $VLLM_PID

echo "AgentHARM metrics and execution logs have been securely saved to: $RESULTS_DIR"
