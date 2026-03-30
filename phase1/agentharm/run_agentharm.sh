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

# 2. Install the UK Safety Institute Framework identically, but SAFELY
echo "[1/3] Installing UK AISI framework dependencies safely (avoiding Gaudi dependency override)..."
# Force huggingface-hub back to the Gaudi-compatible branch to repair env if needed
python3 -m pip install "huggingface-hub<1.0" "transformers>=4.55.0,<4.56.0" --quiet
# Install inspect frameworks while explicitly blocking them from breaking the hub version
python3 -m pip install "inspect-ai>=0.3" "datasets" --quiet
python3 -m pip install "inspect-evals" --no-deps --quiet

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

# Format model name exactly like OmniACT pipeline: "Qwen_Qwen3-VL-2B-Instruct"
SAFE_MODEL_NAME="${MODEL_ID//\//_}"
RESULTS_DIR="results/$SAFE_MODEL_NAME/agentharm"

# Ensure clean directory exists
mkdir -p "$RESULTS_DIR"

# 3. Execute the safety evaluation against your offline inference node natively
echo "[3/3] Triggering full AgentHARM execution loop natively without vLLM (Raw PyTorch on HPU)..."

# Map the generic pipeline to raw HuggingFace eager mode natively
export INSPECT_EVAL_MODEL="hf/$MODEL_ID"

# We pass the eval command using inspect-evals repository defaults for agentharm
# We strictly pass -M trust_remote_code=True exactly like the python wrapper to bypass the Architecture Error natively
inspect eval inspect_evals/agentharm \
    --model $INSPECT_EVAL_MODEL \
    -M trust_remote_code=True \
    --log-dir "$RESULTS_DIR"

# 4. Clean up HPC resources
echo "=== Benchmarking Complete ==="
echo "AgentHARM metrics and execution logs have been securely saved to: $RESULTS_DIR"
