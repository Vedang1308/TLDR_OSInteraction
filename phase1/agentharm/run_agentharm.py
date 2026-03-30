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

# 1. Bruteforce Monkeypatch inspect_ai's internal huggingface module to use Vision-Language Models natively
#    AND forcefully inject trust_remote_code=True into the very last layer before Transformers crashes.
import inspect_ai.model._providers.hf as hf_provider
from transformers import AutoModelForImageTextToText, AutoConfig

_orig_config_from_pretrained = AutoConfig.from_pretrained
_orig_model_from_pretrained = AutoModelForImageTextToText.from_pretrained

@classmethod
def _patched_config_from_pretrained(cls, *args, **kwargs):
    kwargs["trust_remote_code"] = True
    return _orig_config_from_pretrained(*args, **kwargs)

@classmethod
def _patched_model_from_pretrained(cls, *args, **kwargs):
    kwargs["trust_remote_code"] = True
    if "device_map" not in kwargs:
        kwargs["device"] = "hpu"
    return _orig_model_from_pretrained(*args, **kwargs)

# Inject deep into Transformers logic
AutoConfig.from_pretrained = _patched_config_from_pretrained

hf_provider.AutoModelForCausalLM = AutoModelForImageTextToText
hf_provider.AutoModelForCausalLM.from_pretrained = _patched_model_from_pretrained

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
