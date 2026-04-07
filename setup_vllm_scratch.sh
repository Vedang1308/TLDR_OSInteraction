#!/bin/bash
# Script to set up a clean python environment FOR vLLM strictly on the scratch disk
# This will prevent your home directory from running out of space/inodes

echo "Configuring environment paths to use /scratch/svijsy46..."

# 1. Define Scratch Targets
SCRATCH_DIR="/scratch/$USER"
ENV_DIR="$SCRATCH_DIR/phase2-venv"

export HF_HOME="$SCRATCH_DIR/huggingface"
export XDG_CACHE_HOME="$SCRATCH_DIR/xdg"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip_cache"

mkdir -p $HF_HOME $XDG_CACHE_HOME $PIP_CACHE_DIR

# 2. Create the Python Virtual Environment
echo "Creating Python virtual environment at $ENV_DIR..."
python3 -m venv $ENV_DIR

# 3. Activate the environment
source $ENV_DIR/bin/activate

# 4. Install Core Dependencies
echo "Installing vLLM, Transformers, and Torch..."
pip install --upgrade pip
pip install vllm transformers accelerate torch torchvision
pip install bitsandbytes # Useful for tuning vllm quantization on massive models

echo "======================================"
echo "Environment successfully created!"
echo "To activate it in the future, run:"
echo "source $ENV_DIR/bin/activate"
echo "======================================"
