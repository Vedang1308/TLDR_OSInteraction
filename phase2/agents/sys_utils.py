import re
import torch
from transformers import StoppingCriteria

class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, processor, prompt_len, threshold=3):
        """
        Initializes the stopping criteria to detect and prevent infinite click loops.
        - prompt_len: The number of tokens in the input prompt (to ignore history).
        - threshold: Number of identical pyautogui.click calls allowed before stopping.
        """
        self.processor = processor
        self.prompt_len = prompt_len
        self.threshold = threshold
        self.click_pattern = re.compile(r"click\((\d+),\s*(\d+)\)")

    def __call__(self, input_ids, scores, **kwargs):
        """
        Scans generated tokens ONLY for repetitive actions.
        """
        # Isolate ONLY the new generated tokens
        new_ids = input_ids[:, self.prompt_len:]
        decoded_texts = self.processor.batch_decode(new_ids, skip_special_tokens=True)
        
        for decoded_text in decoded_texts:
            clicks = self.click_pattern.findall(decoded_text)
            if not clicks:
                continue
                
            for i in range(len(clicks) - self.threshold + 1):
                window = clicks[i:i + self.threshold]
                if len(set(window)) == 1:
                    return True
        return False

class TextRepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, processor, prompt_len, threshold=5):
        """
        Stops generation if a phrase repeats ONLY in the new output.
        - prompt_len: Ignores the chat history repetitions.
        """
        self.processor = processor
        self.prompt_len = prompt_len
        self.threshold = threshold

    def __call__(self, input_ids, scores, **kwargs):
        # Isolate ONLY the new generated tokens
        new_ids = input_ids[:, self.prompt_len:]
        decoded_texts = self.processor.batch_decode(new_ids, skip_special_tokens=True)
        
        from collections import Counter
        for decoded_text in decoded_texts:
            words = decoded_text.lower().split()
            if len(words) < 15:
                continue
                
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            counts = Counter(trigrams)
            
            if any(v >= self.threshold for v in counts.values()):
                return True
        return False

import torch
import os
import subprocess

def detect_device():
    """Dynamically detects HPU (Gaudi), CUDA (Nvidia), or CPU."""
    # 1. Check for Gaudi HPU
    try:
        subprocess.check_output(["hl-smi"], stderr=subprocess.DEVNULL)
        return "hpu"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 2. Check for Nvidia GPU
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "cpu"

# ⚡ Hardware-Specific Optimizations
DEVICE = detect_device()
if DEVICE == "hpu":
    # Enable HPU Eager-Mode Pipelining for Gaudi2
    os.environ["PT_HPU_EAGER_PIPELINE_ENABLE"] = "1"
    os.environ["PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE"] = "1"
    print("GAUDI DETECTED: HPU Eager-Pipelining Enabled.")
elif DEVICE == "cuda":
    print("NVIDIA DETECTED: Standard CUDA Acceleration Active.")
else:
    print("CPU DETECTED: Non-accelerated fallback mode active.")

def get_hpu_model_singleton():
    """Retrieves the globally cached model from builtins to prevent OOM."""
    import builtins
    if hasattr(builtins, "__GAUDI_CACHED_MODEL__"):
        return builtins.__GAUDI_CACHED_MODEL__, builtins.__GAUDI_CACHED_PROCESSOR__
    else:
        raise RuntimeError("Gaudi singleton model not found in builtins! Run the main script directly.")

def run_qwen_inference(model, processor, messages, images=None, max_tokens=128, temperature=0.7, num_return_sequences=1):
    """
    Standardizes inference with safety critera. 
    Implements Token-Isolation to prevent history-based "Safety Poisoning".
    """
    from transformers import StoppingCriteriaList
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    
    # Calculate exact prompt length to isolate new generation
    prompt_len = inputs.input_ids.shape[1]
    
    # Instantiate isolated criteria
    stop_criteria = StoppingCriteriaList([
        RepetitionStoppingCriteria(processor, prompt_len, threshold=3),
        TextRepetitionStoppingCriteria(processor, prompt_len, threshold=5)
    ])
    
    # Model-Aware Inference Scaling
    try:
        model_name = getattr(model.config, "_name_or_path", "").lower()
        if not model_name:
            model_name = getattr(model, "name_or_path", "").lower()
    except Exception:
        model_name = ""
        
    top_k = 50 # default
    if "2b" in model_name:
        temperature = 0.5 
        top_k = 20
    elif "4b" in model_name:
        temperature = 0.5 
        top_k = 35
    elif "8b" in model_name:
        temperature = max(temperature, 0.7) 
        top_k = 50

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_k=top_k,
            do_sample=(temperature > 0),
            stopping_criteria=stop_criteria,
            num_return_sequences=num_return_sequences
        )
        
    input_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_texts = [text.strip() for text in output_texts]
    
    if num_return_sequences == 1:
        return output_texts[0]
    return output_texts