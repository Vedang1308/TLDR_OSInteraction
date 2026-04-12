#!/bin/bash
# Script to set up a clean CONDA environment FOR vLLM strictly on the scratch disk

echo "Configuring environment paths to use /scratch/$USER..."

# 1. Define Scratch Targets
SCRATCH_DIR="/scratch/$USER"
ENV_DIR="$SCRATCH_DIR/phase2-conda-env"

export HF_HOME="$SCRATCH_DIR/huggingface"
export XDG_CACHE_HOME="$SCRATCH_DIR/xdg"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip_cache"

mkdir -p $HF_HOME $XDG_CACHE_HOME $PIP_CACHE_DIR

# 2. Initialize Conda (Mamba) properly for scripts
source $(conda info --base)/etc/profile.d/conda.sh

# 3. Create the Conda Environment with Python 3.10 (Fixes vllm compilation issues)
echo "Creating Conda environment at $ENV_DIR with Python 3.10..."
conda create --prefix $ENV_DIR python=3.10 -y

# 4. Activate the newly created environment
conda activate $ENV_DIR

# 5. Install dependencies securely enforcing the conda-specific pip
echo "Installing vLLM, Transformers, and Torch..."
python -m pip install --upgrade pip
python -m pip install vllm transformers accelerate torch torchvision bitsandbytes

echo "======================================"
echo "Conda environment successfully created!"
echo "To activate it interactively in the future, run:"
echo "conda activate $ENV_DIR"
echo "======================================"
