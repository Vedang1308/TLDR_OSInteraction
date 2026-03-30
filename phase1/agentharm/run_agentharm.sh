#!/bin/bash
# ==============================================================================
# Phase 1 - Agent Safety Baseline (AgentHARM)
# Target: Intel Gaudi HPU / SLURM Environment (SOL Supercomputer)
# ==============================================================================

# Choose your model via command line argument (e.g. Qwen/Qwen3-VL-2B-Instruct)
MODEL_ID=$1

if [ -z "$MODEL_ID" ]; then
    echo "[FATAL ERROR] You must provide a model to evaluate!"
    echo "Usage: ./run_agentharm.sh <model_id>"
    echo "Example: ./run_agentharm.sh Qwen/Qwen3-VL-2B-Instruct"
    exit 1
fi

PORT=8000

echo "=== INITIALIZING AGENTHARM SAFETY BENCHMARK ON SOL ==="
echo "[1/2] Connecting to active vLLM Inference Server on port $PORT..."

# Format model name exactly like OmniACT pipeline: "Qwen_Qwen3-VL-2B-Instruct"
SAFE_MODEL_NAME="${MODEL_ID//\//_}"
RESULTS_DIR="results/$SAFE_MODEL_NAME/agentharm"

# Ensure clean directory exists
mkdir -p "$RESULTS_DIR"

# Map the generic OpenAI provider syntax natively to our local vLLM pipeline
export INSPECT_EVAL_MODEL="openai/$MODEL_ID"
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="sk-mock-key"

echo "[2/2] Triggering full AgentHARM execution loop..."

# We pass the eval command using inspect-evals repository defaults for agentharm
inspect eval inspect_evals/agentharm \
    --model $INSPECT_EVAL_MODEL \
    --log-dir "$RESULTS_DIR"

echo "=== Benchmarking Complete ==="
echo "AgentHARM metrics and execution logs have been securely saved to: $RESULTS_DIR"
