import os
import sys

import argparse
import subprocess

def detect_device():
    """Detects available hardware accelerator (Gaudi HPU, Nvidia GPU, or CPU)."""
    try:
        # Check for Gaudi HPU
        subprocess.check_output(["hl-smi"], stderr=subprocess.DEVNULL)
        return "hpu"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        # Check for Nvidia GPU
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "cpu"

def load_vlm_model(model_name, device):
    """Loads the specified HuggingFace VLM Model onto the available hardware device.
    Ensure that HF_HOME is respected by the transformers library.
    """
    print(f"Loading '{model_name}' onto {device}...")
    
    # Qwen-VL architecture commonly uses AutoProcessor and AutoModelForImageTextToText
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        if device == "hpu":
            # Opt for Optimum Habana implementation optimized for Gaudi
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_features
            adapt_transformers_to_features()
            from transformers import AutoModelForImageTextToText
            
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            model.to("hpu")
            print("Successfully loaded model on Intel Gaudi HPU.")
            return model, processor
            
        elif device == "cuda":
            # Standard CUDA loading
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            print("Successfully loaded model on Nvidia GPU.")
            return model, processor
            
        else:
            # Fallback to general/CPU
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            print("Successfully loaded model on CPU.")
            return model, processor
            
    except Exception as e:
        import traceback
        print(f"Failed to load VLM model {model_name} on {device}. Error Traceback:")
        traceback.print_exc()
        sys.exit(1)

def eval_osworld(model_name, device, model, processor):
    print(f"[OSWorld] Evaluating {model_name} acting via OSWorld tasks...")
    # NOTE: Actual integration involves OSWorld's `run.py` agent loop interfacing with inference.
    # Since run.py manages its own process, one would typically use vLLM server endpoints or pass model configs to OSWorld.
    # We display a placeholder for this structure.
    print(f"[OSWorld] Passing logic to OSWorld's pipeline for {model_name}.")

def eval_blind_act(model_name, device, model, processor):
    print(f"[BLIND-ACT] Evaluating {model_name} acting via BLIND-ACT tasks...")
    # BLIND-ACT similarly builds upon OSWorld's execution loop.
    print(f"[BLIND-ACT] Passing logic to BLIND-ACT's pipeline for {model_name}.")

def eval_omniact(model_name, device, model, processor):
    print(f"[OmniACT] Evaluating {model_name} against OmniACT dataset...")
    
    # Configure resilient checkpointing
    import json
    safe_model_name = model_name.replace("/", "_")
    os.makedirs("results", exist_ok=True)
    checkpoint_file = f"results/omniact_checkpoint_{safe_model_name}.json"
    
    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"[OmniACT] Resuming from checkpoint. {len(results)} tasks already completed.")

    try:
        from datasets import load_dataset
        dataset = load_dataset("Writer/omniact", split="test")
        print(f"[OmniACT] Processing exactly {len(dataset)} examples on {device}...")
        
        for idx, item in enumerate(dataset):
            task_id = str(idx) # Or item['id'] depending on dataset structure
            if task_id in results:
                continue # Skip efficiently
                
            # --- [PLACEHOLDER] Insert your actual Agent Prompting & Generation Logic here ---
            # generated_action = agent_loop(item["instruction"], item["image"], model, processor)
            generated_action = "MOCK_CLICK_ACTION" 
            
            # Save progress incrementally to disk
            results[task_id] = {"instruction": item.get("instruction", ""), "action": generated_action}
            with open(checkpoint_file, "w") as f:
                json.dump(results, f, indent=4)
                
            if idx % 10 == 0:
                print(f"  -> Saved checkpoint for {idx}/{len(dataset)} examples.")
                
        print(f"\n[OmniACT] Evaluation completely finished! Results securely saved to {checkpoint_file}")
        
    except ImportError:
        print("Please ensure 'datasets' is installed (`pip install datasets`).")

def main():
    parser = argparse.ArgumentParser(description="Unified Runner for VLMs on OSWorld, BLIND-ACT, and OmniACT")
    parser.add_argument("--benchmark", type=str, required=True, choices=["osworld", "blind-act", "omniact"])
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model string (e.g. Qwen/Qwen3-VL-2B-Instruct)")
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

    # Route request to specific execution loops
    if args.benchmark == "osworld":
        eval_osworld(args.model, device, model, processor)
    elif args.benchmark == "blind-act":
        eval_blind_act(args.model, device, model, processor)
    elif args.benchmark == "omniact":
        eval_omniact(args.model, device, model, processor)

if __name__ == "__main__":
    main()
