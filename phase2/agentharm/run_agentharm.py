import sys
import os
import torch

# 🚀 PROJECT PATH RESOLUTION: Add the root directory to sys.path
# This ensures that 'from phase2...' imports work regardless of where the script is invoked.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ⚡ AUTOMATION: Always apply defensive patches before starting AgentHARM
# This prevents KeyErrors during 'Safe Pivot' scenarios across the SOL cluster.
try:
    # Ensure the script can find patch_agentharm.py in its local directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from patch_agentharm import auto_patch_agentharm
    auto_patch_agentharm()
except Exception as e:
    print(f"WARNING: Automated patching failed: {e}")

if len(sys.argv) < 2:
    print("FATAL: Please provide a model ID. Example: python run_agentharm.py Qwen/Qwen3-VL-2B-Instruct")
    sys.exit(1)

model_id = sys.argv[1]
safe_model_name = model_id.replace("/", "_")
log_dir = f"results/{safe_model_name}/agentharm"
os.makedirs(log_dir, exist_ok=True)

# Disable the confusing full-screen UI Dashboard so it scrolls normally like OmniACT
os.environ["INSPECT_DISPLAY"] = "plain"

print(f"=== INITIALIZING NATIVE PYTORCH AGENTHARM RUN ===")

# 1. Warm up HuggingFace BEFORE inspect_ai starts to prevent TaskGroup Isolation bugs!
print("Warming up HuggingFace Module Cache globally...")
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from phase2.agents.sys_utils import detect_device

# Determine the best available hardware
DEVICE = detect_device()
print(f"Hardware Detection Result: {DEVICE.upper()}")

# Pre-load statically to populate sys.modules seamlessly
GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
GLOBAL_CONFIG = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
GLOBAL_MODEL = AutoModelForImageTextToText.from_pretrained(model_id, config=GLOBAL_CONFIG, trust_remote_code=True)

# 🚀 UNIVERSAL LOADING: Handle HPU vs CUDA vs CPU
if DEVICE == "hpu":
    GLOBAL_MODEL.to("hpu")
elif DEVICE == "cuda":
    GLOBAL_MODEL.to("cuda")

GLOBAL_MODEL.eval()
print(f"Model successfully anchored to {DEVICE.upper()} at global scope!")

# 2. Expose to the local system so GaudiProvider can grab it
import builtins
builtins.__GAUDI_CACHED_MODEL__ = GLOBAL_MODEL
builtins.__GAUDI_CACHED_PROCESSOR__ = GLOBAL_PROCESSOR

# 3. Register our custom Gaudi PyTorch provider.
import gaudi_provider

# 4. Trigger the framework evaluation using our local plugin definition
from inspect_ai import eval, eval_retry
from inspect_evals.agentharm import agentharm
import glob

print(f"Starting native HPU eager-mode evaluation for {model_id} via GaudiProvider...")

# Construct the custom provider string (e.g. gaudi_qwen/Qwen/Qwen3-VL-2B-Instruct)
provider_model_str = f"gaudi_qwen/{model_id}"

# Smart Resume: Check if there is an existing log to resume from
existing_logs = sorted(glob.glob(os.path.join(log_dir, "*.json")), reverse=True)
retry_log = None

if existing_logs:
    # Check if the most recent log was actually finished
    import json
    try:
        with open(existing_logs[0], 'r') as f:
            log_data = json.load(f)
            if log_data.get("status") != "completed":
                retry_log = existing_logs[0]
                print(f"RESUME DETECTED: Found incomplete log covering {len(log_data.get('eval', {}).get('dataset', {}).get('sample_ids', []))} samples.")
                print(f"Resuming from: {os.path.basename(retry_log)}")
    except Exception:
        pass

if retry_log:
    eval_retry(
        retry_log,
        log_dir=log_dir,
        log_format="json",
        max_connections=1
    )
else:
    print("No incomplete logs found. Starting a fresh AgentHARM run...")
    eval(
        agentharm(
            refusal_judge=provider_model_str,
            semantic_judge=provider_model_str
        ),
        model=provider_model_str,
        log_dir=log_dir,
        log_format="json",
        max_connections=1
    )

print(f"\nAgentHARM metrics and execution logs have been securely saved to: {log_dir}")
