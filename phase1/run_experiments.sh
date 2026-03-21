#!/bin/bash
# run_experiments.sh
# Script to run all the Qwen3-VL models on the 3 benchmarks sequentially

set -e

export HF_HOME="/scratch/$USER/huggingface_cache"
export XDG_CACHE_HOME="/scratch/$USER/xdg_cache"
export TMPDIR="/scratch/$USER/tmp"

# Activate environment dynamically depending on the hardware cluster
eval "$(conda shell.bash hook)"
if command -v hl-smi &> /dev/null; then
    conda activate vllm_gaudi
    echo "Intel Gaudi HPU Detected natively! Activating vllm_gaudi environment..."
elif command -v nvidia-smi &> /dev/null; then
    # Nvidia A100 environments require a standard CUDA pytorch installation rather than Intel Habana drivers
    # conda activate your_cuda_env 
    echo "Nvidia CUDA Node Detected natively! Assuming your standard GPU Conda environment is actively loaded!"
fi

MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
)

BENCHMARKS=(
    "osworld"
    "omniact"
)

echo "Starting evaluation run for Qwen3-VL suite."

for MODEL in "${MODELS[@]}"; do
    echo "================================================================="
    echo " MODEL IN EVALUATION: $MODEL "
    echo "================================================================="
    
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        echo ">>> LAUNCHING $BENCHMARK <<<"
        
        # We pass execution to the unified evaluate wrapper, which detects hardware
        # and runs the evaluation logic for the model.
        python eval_wrapper.py --benchmark "$BENCHMARK" --model "$MODEL"
        
        echo ">>> FINISHED $BENCHMARK <<<"
        echo "-----------------------------------------------------------------"
    done
done

echo "All evaluations completed successfully!"
