!/bin/bash
# setup_from_scratch.sh

set -e

echo "================================================="
echo "Building Fresh vLLM Environment on Sol Scratch"
echo "================================================="

# 1. Route large downloads to Scratch
export HF_HOME="/scratch/$USER/huggingface_cache"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"

mkdir -p "$HF_HOME"
mkdir -p "$PIP_CACHE_DIR"

# 2. Load Sol modules
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
module load cudnn/9.17.1-cuda12

# 3. Hook Mamba into the script
eval "$(conda shell.bash hook)"

# 4. Create the new environment
ENV_PATH="/scratch/$USER/osworld"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating new Mamba environment at $ENV_PATH..."
    mamba create -p "$ENV_PATH" python=3.10 -y
else
    echo "Environment already exists at $ENV_PATH"
fi

# 5. Activate the environment
conda activate "$ENV_PATH"

# 6. Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies successfully installed from requirements.txt!"
else
    echo "ERROR: requirements.txt not found in the current directory."
    exit 1
fi

echo "================================================="
echo "Setup complete! Your environment is ready at: $ENV_PATH"
echo "================================================="