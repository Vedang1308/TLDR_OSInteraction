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
        decoded_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        
        clicks = self.click_pattern.findall(decoded_text)
        if not clicks:
            return False
            
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
        decoded_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        
        words = decoded_text.lower().split()
        if len(words) < 15:
            return False
            
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        from collections import Counter
        counts = Counter(trigrams)
        
        if any(v >= self.threshold for v in counts.values()):
            return True
        return False

import torch
import os

# ⚡ GLOBAL SPEED OPTIMIZATION: Enable HPU Eager-Mode Pipelining
# This allows the Gaudi2/HPU hardware to pre-fetch kernels while the CPU is still tokenizing.
os.environ["PT_HPU_EAGER_PIPELINE_ENABLE"] = "1"
os.environ["PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE"] = "1"

def detect_device():
    """Detects available hardware (HPU or CPU)."""
    try:
        import habana_frameworks.torch.core as htcore
        return "hpu"
    except ImportError:
        return "cpu"

def get_hpu_model_singleton():
    """Retrieves the globally cached model from builtins to prevent OOM."""
    import builtins
    if hasattr(builtins, "__GAUDI_CACHED_MODEL__"):
        return builtins.__GAUDI_CACHED_MODEL__, builtins.__GAUDI_CACHED_PROCESSOR__
    else:
        raise RuntimeError("Gaudi singleton model not found in builtins! Run the main script directly.")

def run_qwen_inference(model, processor, messages, images=None, max_tokens=128, temperature=0.7):
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
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            do_sample=(temperature > 0),
            stopping_criteria=stop_criteria
        )
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    return output_text
