#!/bin/bash
# setup_benchmarks.sh
# Script to set up the environment and clone benchmarks on SOL supercomputer

# Enable strict mode
set -e

echo "================================================="
echo "Setting up unified environment for OSWorld, BLIND-ACT, and OmniACT"
echo "================================================="

# Environment variables for Cache
export HF_HOME="/scratch/$USER/huggingface_cache"
export XDG_CACHE_HOME="/scratch/$USER/xdg_cache"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
export TMPDIR="/scratch/$USER/tmp"

mkdir -p "$HF_HOME"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$TMPDIR"

echo "Cache and Temp directories set to /scratch/$USER"

# Create a common conda environment in scratch
if [ ! -d "/scratch/$USER/benchmarks_env" ]; then
    echo "Creating Conda environment in /scratch/$USER/benchmarks_env with Python 3.10..."
    conda create -p "/scratch/$USER/benchmarks_env" python=3.10 -y
fi

# We must use 'source activate' or standard activation for bash script
eval "$(conda shell.bash hook)"
conda activate "/scratch/$USER/benchmarks_env"

# Install general dependencies
echo "Installing general dependencies..."
pip install transformers accelerate datasets "datasets[vision]" pillow optimum requests

# Hardware-specific torch initialization
if command -v hl-smi &> /dev/null; then
    echo "-------------------------------------------------"
    echo "Intel Gaudi HPU Detected via hl-smi."
    echo "Installing Habana PyTorch packages..."
    echo "-------------------------------------------------"
    # Use intel habana PyTorch versions (assumes working from Habana deep learning containers, or install via pip)
    pip install optimum[habana]
elif command -v nvidia-smi &> /dev/null; then
    echo "-------------------------------------------------"
    echo "Nvidia GPU Detected via nvidia-smi."
    echo "Installing standard PyTorch with CUDA 12.1..."
    echo "-------------------------------------------------"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU/HPU detected. Proceeding with CPU-only PyTorch."
    pip install torch torchvision torchaudio
fi

echo "================================================="
echo "Cloning Repository Codebases"
echo "================================================="

# 1. OSWorld
if [ ! -d "OSWorld" ]; then
    echo "Cloning OSWorld..."
    git clone https://github.com/xlang-ai/OSWorld.git
    cd OSWorld
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd ..
else
    echo "OSWorld already cloned."
fi

# 2. BLIND-ACT
if [ ! -d "cua-blind-goal-directedness" ]; then
    echo "Cloning BLIND-ACT..."
    git clone https://github.com/microsoft/cua-blind-goal-directedness.git
    cd cua-blind-goal-directedness
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd ..
else
    echo "BLIND-ACT already cloned."
fi

# 3. OmniACT (Available via HuggingFace Datasets)
echo "OmniACT does not require a repository clone; it will be loaded from HuggingFace Datasets (Writer/omniact)."

echo "================================================="
echo "Setup complete! To begin, activate the environment:"
echo "conda activate /scratch/$USER/benchmarks_env"
echo "================================================="
