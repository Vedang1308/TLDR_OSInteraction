#!/bin/bash
# run_experiments.sh
# Script to run all the Qwen3-VL models on the 3 benchmarks sequentially

set -e

export HF_HOME="/scratch/$USER/huggingface_cache"
export XDG_CACHE_HOME="/scratch/$USER/xdg_cache"
export TMPDIR="/scratch/$USER/tmp"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate vllm_gaudi

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
