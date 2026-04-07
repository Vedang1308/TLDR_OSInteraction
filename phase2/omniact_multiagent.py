import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 🚀 PROJECT PATH RESOLUTION: Add the root directory to sys.path
# This ensures that 'from phase2...' imports work regardless of where the script is invoked.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ⚡ GLOBAL SPEED OPTIMIZATION: Enable HPU Eager-Mode Pipelining
# This allows the Gaudi2 hardware to pre-fetch kernels during tokenization.
os.environ["PT_HPU_EAGER_PIPELINE_ENABLE"] = "1"
os.environ["PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE"] = "1"

import argparse
import subprocess
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import math
import re

class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, processor, threshold=2):
        self.processor = processor
        self.threshold = threshold
        self.pattern = re.compile(r"click\((\d+),\s*(\d+)\)")

    def __call__(self, input_ids, scores, **kwargs):
        # Decode enough context to check recent lines
        generated_text = self.processor.batch_decode(input_ids[:, -100:], skip_special_tokens=True)[0]
        lines = [l.strip() for l in generated_text.split('\n') if l.strip()]
        
        if len(lines) >= self.threshold:
            last_lines = lines[-self.threshold:]
            # 1. Exact match check
            if len(set(last_lines)) == 1:
                return True
            
            # 2. Drifting Click detection
            clicks = []
            for line in last_lines:
                match = self.pattern.search(line)
                if match:
                    clicks.append((int(match.group(1)), int(match.group(2))))
            
            if len(clicks) == self.threshold:
                # Check if they are all very close or moving in a tight band
                for i in range(1, len(clicks)):
                    dist = math.sqrt((clicks[i][0]-clicks[i-1][0])**2 + (clicks[i][1]-clicks[i-1][1])**2)
                    if dist < 40: # Drifting or exact click
                        continue
                    else:
                        return False
                return True
        return False

from phase2.agents.sys_utils import detect_device

def load_vlm_model(model_name, device):
    """Loads the specified HuggingFace VLM Model onto the available hardware device.
    Ensure that HF_HOME is respected by the transformers library.
    """
    print(f"Loading '{model_name}' onto {device}...")
    
    # Qwen-VL architecture commonly uses AutoProcessor and AutoModelForImageTextToText
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        if device == "hpu":
            # Modern Gaudi drivers typically handle operations natively without the old adapter
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_name, config=config, trust_remote_code=True)
            except Exception:
                try:
                    from transformers import AutoModelForVision2Seq
                    model = AutoModelForVision2Seq.from_pretrained(model_name, config=config, trust_remote_code=True)
                except Exception:
                    try:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
                    except Exception:
                        from transformers import AutoModel
                        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            model.to("hpu")
            print("Successfully loaded model on Intel Gaudi HPU.")
            return model, processor
            
        elif device == "cuda":
            # Standard CUDA loading
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_name, config=config, device_map="auto", trust_remote_code=True)
            except Exception:
                try:
                    from transformers import AutoModelForVision2Seq
                    model = AutoModelForVision2Seq.from_pretrained(model_name, config=config, device_map="auto", trust_remote_code=True)
                except Exception:
                    try:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", trust_remote_code=True)
                    except Exception:
                        from transformers import AutoModel
                        model = AutoModel.from_pretrained(model_name, config=config, device_map="auto", trust_remote_code=True)
            print("Successfully loaded model on Nvidia GPU.")
            return model, processor
            
        else:
            # Fallback to general/CPU
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_name, config=config, trust_remote_code=True)
            except Exception:
                try:
                    from transformers import AutoModelForVision2Seq
                    model = AutoModelForVision2Seq.from_pretrained(model_name, config=config, trust_remote_code=True)
                except Exception:
                    try:
                        from transformers import AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
                    except Exception:
                        from transformers import AutoModel
                        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            print("Successfully loaded model on CPU.")
            return model, processor
            
    except Exception as e:
        import traceback
        print(f"Failed to load VLM model {model_name} on {device}. Error Traceback:")
        traceback.print_exc()
        sys.exit(1)

# Multi-agent imports
import builtins
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from phase2.agents.omniact_agent_v2 import OmniactAgentSystemV2

def eval_omniact(model_name, device, model, processor, limit=-1):
    print(f"[OmniACT] Evaluating {model_name} against OmniACT dataset...")
    
    # Configure resilient checkpointing
    import json
    safe_model_name = model_name.replace("/", "_")
    
    # Establish strict Model/Benchmark directory segregation for massive-scale analytics
    results_dir = os.path.join("results-v2", safe_model_name, "omniact")
    os.makedirs(results_dir, exist_ok=True)
    
    # Strip the generic monolithic vendor prefix to build a hyper-clean output filename string natively
    clean_name = safe_model_name.replace("Qwen_", "").lower()
    checkpoint_file = os.path.join(results_dir, f"phase2_multiagent_omniact_{clean_name}_results.json")
    
    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"[OmniACT] Resuming from checkpoint. {len(results)} tasks already completed.")

    import urllib.request
    import zipfile
    import glob
    
    data_dir = "omniact_data"
    zip_path = "omniact_data.zip"
    
    if not os.path.exists(data_dir):
         print(f"[OmniACT] Full 9.8k evaluation dataset not found locally. Downloading ~260MB raw data.zip from HuggingFace...")
         # Download directly from HuggingFace's resolved dataset URL
         urllib.request.urlretrieve("https://huggingface.co/datasets/Writer/omniact/resolve/main/data.zip", zip_path)
         print("[OmniACT] Extracting archive...")
         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
             zip_ref.extractall(data_dir)
         print("[OmniACT] Extraction complete!")
         
    # Locate all task script files in the tasks/ directory
    task_files = glob.glob(os.path.join(data_dir, "data", "tasks", "**", "task_*.txt"), recursive=True)
    if limit > 0:
        task_files = task_files[:limit]
    total_tasks = len(task_files)
    print(f"[OmniACT] Processing exactly {total_tasks} raw physical examples on {device}...")
    
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    file_lock = threading.Lock()
    
    def process_task(idx, task_txt_path):
        task_filename = os.path.basename(task_txt_path)
        # e.g. task_1.14.txt -> "1.14" -> screen "1"
        stripped_metrics = task_filename.replace("task_", "").replace(".txt", "")
        screen_id = stripped_metrics.split(".")[0]
        
        # Use relative path as the unique task_id to avoid domain collisions
        task_dir = os.path.dirname(task_txt_path)
        task_rel_path = os.path.relpath(task_txt_path, os.path.join(data_dir, "data", "tasks"))
        task_id = task_rel_path.replace(os.sep, "__") # unique key
        
        with file_lock:
            if task_id in results:
                return # Skip efficiently
            
        # Reconstruct the image path by swapping /tasks/ for /data/ in the directory string
        img_dir = task_dir.replace(os.sep + "tasks" + os.sep, os.sep + "data" + os.sep)
        image_path = os.path.join(img_dir, f"screen{screen_id}.png")
        
        # Datasets are highly inconsistent natively (Web uses screen1.png, Desktop uses screen_1.png)
        if not os.path.exists(image_path):
            fallback_image_path = os.path.join(img_dir, f"screen_{screen_id}.png")
            if os.path.exists(fallback_image_path):
                image_path = fallback_image_path
        
        # Read Task Instruction natively from disk (First line of the text file contains the prompt)
        with open(task_txt_path, "r", encoding="utf-8") as tf:
            lines = tf.readlines()
            instruction_text = lines[0].replace("Task: ", "").strip() if len(lines) > 0 else "Instruction Load Failure"
            
        if not os.path.exists(image_path):
            print(f"  -> WARNING: Missing physical image for {task_id} -> {image_path}. Skipping isolated task.")
            return
            
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Smart Image Resolution Routing based on architecture size
        max_dim = 1280
        if "8b" in model_name.lower():
            max_dim = 1920 # HD Preservation for large models
            print(f"  -> [HD Vision] Using expanded {max_dim}x{max_dim} context window.")
        else:
            max_dim = 1280 # Aggressively cap raw pixel arrays for smaller models
            
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        # 0. UI Metadata Grounding REMOVED for Phase 1 Baseline (Pure Vision-to-Action)
        grounding_text = ""
        
        # 1. Structure the Paper-Aligned conversational payload (Fig 4)
        role_desc = "Role: You are an excellent robotic process automation agent. Your goal is to generate PyAutoGUI code to achieve the given task based on the screenshot and UI elements provided."
        api_ref = """Below is the API Reference to use for the process:
def click(x: float, y: float):
    \"\"\"takes the mouse to location (X, Y) and does a left click\"\"\"
    pass
def write(text: str):
    \"\"\"Types the text, which is a string\"\"\"
    pass
def press(key: str):
    \"\"\"Presses a specific keyboard key\"\"\"
    pass
def scroll(amount: int):
    \"\"\"Scrolls the screen by amount\"\"\"
    pass"""

        examples = """Here are a few examples for reference:
Example 1:
Task: Open the email from Google
Output Script:
pyautogui.click(410, 578)

Example 2:
Task: Search for "OmniACT"
Output Script:
pyautogui.click(200, 50)
pyautogui.write("OmniACT")
pyautogui.press("enter")"""

        context_string = f"{api_ref}\n\n{examples}"

        if os.environ.get("DRY_RUN") == "1":
            print(f"\n--- DRY RUN PROMPT FOR TASK {task_id} ---")
            print(f"Instruction: {instruction_text}")
            print(f"Context: {context_string}")
            print("--- END DRY RUN ---")
            sys.exit(0)
            
        print(f"\n[{idx}/{total_tasks}] Task: {task_id}")
        
        # Instantiate the improved agent system dynamically (Density Consensus V2)
        agent_system = OmniactAgentSystemV2(benchmark="omniact", num_samples=5, temperature=0.7)
        
        # Execute the high-temp sample loop
        generated_action = agent_system.execute_task(
            instruction=instruction_text,
            context=context_string,
            images=[image],
            task_index=idx
        )
        
        with file_lock:
            results[task_id] = {"instruction": instruction_text[:100], "action": generated_action}
            with open(checkpoint_file, "w") as f:
                json.dump(results, f, indent=4)
            
        print(f"   Qwen3 Token: {generated_action.strip()}\n")

    print(f"[OmniACT] Launching concurrency pool (4 workers)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_task, idx, task_txt_path) for idx, task_txt_path in enumerate(task_files)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception generated during worker execution: {e}")
    print(f"\n[OmniACT] Evaluation completely finished! Results securely saved to {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser(description="Unified Runner for VLMs on OSWorld, BLIND-ACT, and OmniACT")
    parser.add_argument("--benchmark", type=str, required=True, choices=["osworld", "blind-act", "omniact"])
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model string (e.g. Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of tasks to evaluate")
    args = parser.parse_args()

    # Hardware Detection
    device = detect_device()
    print(f"Hardware Detection Initialized: {device.upper()}")
    
    # Verify Cache Paths 
    hf_cache = os.environ.get("HF_HOME", "NOT SET")
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "NOT SET")
    print(f"Using HF_HOME={hf_cache}")
    print(f"Using XDG_CACHE_HOME={xdg_cache}")

    # For safety, explicitly set huggingface caching properties:
    os.environ["DEFAULT_HF_HOME"] = os.environ.get("HF_HOME", "")
    os.environ["DEFAULT_XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", "")

    # Load Model (In real evaluation you may skip model instancing if using an OpenAI-compatible API endpoint like pure vLLM, 
    # but since the prompt explicitly asks to load it with python directly handling habana optimum optimizations vs cuda wrapper, we perform loading here.)
    model, processor = load_vlm_model(args.model, device)

    # Setup Singleton anchor for TriAgentSystem
    builtins.__GAUDI_CACHED_MODEL__ = model
    builtins.__GAUDI_CACHED_PROCESSOR__ = processor

    # Route request to specific execution loops
    if args.benchmark == "omniact":
        eval_omniact(args.model, device, model, processor, limit=args.limit)

if __name__ == "__main__":
    main()
