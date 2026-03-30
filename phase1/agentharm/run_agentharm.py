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
print("Injecting Model Architecture Monkeypatch to bypass Inspect AI CausalLM restrictions...")

# 1. Monkeypatch inspect_ai's internal huggingface module to use Vision-Language Models natively
import inspect_ai.model._providers.hf as hf_provider
from transformers import AutoModelForImageTextToText
hf_provider.AutoModelForCausalLM = AutoModelForImageTextToText

# 2. Trigger the framework evaluation
from inspect_ai import eval
from inspect_ai.model import get_model
from inspect_evals.agentharm import agentharm

print(f"Starting native HPU eager-mode evaluation for {model_id}...")
print("Instantiating Judge Model locally to completely bypass missing OpenAI API Keys...")

# Manually pre-load the model as the automated judge so it skips pinging GPT-4o
local_judge = get_model(f"hf/{model_id}", model_args={"trust_remote_code": True})

eval(
    agentharm(judge_model=local_judge),
    model=f"hf/{model_id}",
    model_args={"trust_remote_code": True},
    log_dir=log_dir
)

print(f"\nAgentHARM metrics and execution logs have been securely saved to: {log_dir}")
