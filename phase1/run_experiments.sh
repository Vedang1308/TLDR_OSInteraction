#!/bin/bash
# run_experiments.sh
# Script to run all the Qwen3-VL models on the 3 benchmarks sequentially

set -e

export HF_HOME="/scratch/$USER/huggingface_cache"
export XDG_CACHE_HOME="/scratch/$USER/xdg_cache"
export TMPDIR="/scratch/$USER/tmp"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate "/scratch/$USER/benchmarks_env"

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
        
        if [ "$BENCHMARK" == "osworld" ]; then
            echo "Running OSWorld using Gaudi-optimized runner..."
            # Execute the new runner which imports OSWorld logic natively from phase1/
            python gaudi_osworld_runner.py \
                          --config_path OSWorld/evaluation_examples/test.json \
                          --model "$MODEL" --max_steps 15 --provider "docker"
        else
            # We pass execution to the unified evaluate wrapper, which detects hardware
            # and runs the evaluation logic for the model.
            python eval_wrapper.py --benchmark "$BENCHMARK" --model "$MODEL"
        fi
        
        echo ">>> FINISHED $BENCHMARK <<<"
        echo "-----------------------------------------------------------------"
    done
done

echo "All evaluations completed successfully!"
