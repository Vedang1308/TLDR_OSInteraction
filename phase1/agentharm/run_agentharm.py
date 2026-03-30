import sys
import os
import torch

if len(sys.argv) < 2:
    print("FATAL: Please provide a model ID. Example: python run_agentharm.py Qwen/Qwen3-VL-2B-Instruct")
    sys.exit(1)

model_id = sys.argv[1]
safe_model_name = model_id.replace("/", "_")
log_dir = f"results/{safe_model_name}/agentharm"
os.makedirs(log_dir, exist_ok=True)

print(f"=== INITIALIZING NATIVE PYTORCH AGENTHARM RUN ===")

# 1. Warm up HuggingFace BEFORE inspect_ai starts to prevent TaskGroup Isolation bugs!
# Because inspect_ai uses coroutines and isolates module loading, we must structurally 
# initialize trust_remote_code locally in the true python main thread first!
print("Warming up HuggingFace Module Cache globally...")
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText

# Pre-load statically to populate sys.modules seamlessly
GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
GLOBAL_CONFIG = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
GLOBAL_MODEL = AutoModelForImageTextToText.from_pretrained(model_id, config=GLOBAL_CONFIG, trust_remote_code=True)
GLOBAL_MODEL.to("hpu").eval()
print(f"Model successfully anchored to HPU at global scope!")

# 2. Expose to the local system so GaudiProvider can grab it
import builtins
builtins.__GAUDI_CACHED_MODEL__ = GLOBAL_MODEL
builtins.__GAUDI_CACHED_PROCESSOR__ = GLOBAL_PROCESSOR

# 3. Register our custom Gaudi PyTorch provider.
import gaudi_provider

# 4. Trigger the framework evaluation using our local plugin definition
from inspect_ai import eval
from inspect_evals.agentharm import agentharm

print(f"Starting native HPU eager-mode evaluation for {model_id} via GaudiProvider...")

# Construct the custom provider string (e.g. gaudi_qwen/Qwen/Qwen3-VL-2B-Instruct)
provider_model_str = f"gaudi_qwen/{model_id}"

print("Instantiating Judge Model locally to completely bypass missing OpenAI API Keys...")
eval(
    agentharm(
        refusal_judge=provider_model_str,
        semantic_judge=provider_model_str
    ),
    model=provider_model_str,
    log_dir=log_dir
)

print(f"\nAgentHARM metrics and execution logs have been securely saved to: {log_dir}")
