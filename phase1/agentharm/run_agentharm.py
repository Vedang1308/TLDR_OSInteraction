import sys
import os

if len(sys.argv) < 2:
    print("FATAL: Please provide a model ID. Example: python run_agentharm.py Qwen/Qwen3-VL-2B-Instruct")
    sys.exit(1)

model_id = sys.argv[1]
safe_model_name = model_id.replace("/", "_")
log_dir = f"results/{safe_model_name}/agentharm"
os.makedirs(log_dir, exist_ok=True)

print(f"=== INITIALIZING NATIVE PYTORCH AGENTHARM RUN ===")

# 1. Register our custom, foolproof Gaudi PyTorch provider.
# This explicitly bypasses inspect_ai's internal huggingface provider, 
# statically avoiding the AutoModelForCausalLM and AutoConfig validation bugs.
import gaudi_provider

# 2. Trigger the framework evaluation using our local plugin definition
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
