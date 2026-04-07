#!/bin/bash
#SBATCH --partition=public
#SBATCH --account=class_cse572spring2026
#SBATCH --qos=class
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --job-name=omniact-v3

set -euo pipefail

cd /scratch/$USER/TLDR_OSInteraction/phase2

module load cuda-12.1.1-gcc-12.1.0
module load cudnn/9.17.1-cuda12
module load mamba/latest

source activate ph2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"

echo "Starting Multi-Agent Benchmark Tasks..."

python omniact_multiagent.py --benchmark omniact --model Qwen/Qwen3-VL-2B-Instruct > 2b_model.log 2>&1 &
python omniact_multiagent.py --benchmark omniact --model Qwen/Qwen3-VL-4B-Instruct > 4b_model.log 2>&1 &
python omniact_multiagent.py --benchmark omniact --model Qwen/Qwen3-VL-8B-Instruct > 8b_model.log 2>&1 &

wait

echo "All tasks completed."