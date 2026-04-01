import builtins
import torch
import re
import math
from transformers import StoppingCriteria, StoppingCriteriaList

class RepetitionStoppingCriteria(StoppingCriteria):
    """Stops generation when the model starts looping identical or near-identical click lines."""
    def __init__(self, processor, threshold=3):
        self.processor = processor
        self.threshold = threshold
        self.pattern = re.compile(r"click\((\d+),\s*(\d+)\)")

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.processor.batch_decode(input_ids[:, -200:], skip_special_tokens=True)[0]
        lines = [l.strip() for l in generated_text.split('\n') if l.strip()]
        
        if len(lines) >= self.threshold:
            last_lines = lines[-self.threshold:]
            # Exact match check
            if len(set(last_lines)) == 1:
                return True
            
            # Drifting Click detection
            clicks = []
            for line in last_lines:
                match = self.pattern.search(line)
                if match:
                    clicks.append((int(match.group(1)), int(match.group(2))))
            
            if len(clicks) == self.threshold:
                all_close = True
                for i in range(1, len(clicks)):
                    dist = math.sqrt((clicks[i][0]-clicks[i-1][0])**2 + (clicks[i][1]-clicks[i-1][1])**2)
                    if dist >= 40:
                        all_close = False
                        break
                if all_close:
                    return True
        return False


def get_hpu_model_singleton():
    """
    Retrieves the globally warmed-up HuggingFace Model and Processor from memory.
    This is critical for the TriAgentSystem: three agents (Manager, Executor, Auditor)
    would require 3 x 16GB = 48GB of VRAM if loaded individually.
    Anchoring to the Python builtins acts as a native Singleton Provider.
    """
    if hasattr(builtins, "__GAUDI_CACHED_MODEL__"):
        model = builtins.__GAUDI_CACHED_MODEL__
        processor = builtins.__GAUDI_CACHED_PROCESSOR__
        return model, processor
    else:
        raise RuntimeError(
            "FATAL: HPU Singleton not found! Make sure you initialized the model "
            "globally in `builtins` before instantiating the Multi-Agent System."
        )

def run_qwen_inference(model, processor, messages, images=None, tools=None, max_tokens=512, temperature=0.7):
    """
    Utility wrapper to natively execute Qwen3-VL inferences on the HPU.
    Includes RepetitionStoppingCriteria to prevent infinite click loops.
    """
    # Build text prompt
    text_prompt = processor.apply_chat_template(
        messages, 
        tools=tools,
        add_generation_prompt=True
    )
    
    # We allow the caller to pass images within the message dicts if needed.
    inputs = processor(
        text=[text_prompt], 
        images=images, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    stopping_criteria = StoppingCriteriaList([RepetitionStoppingCriteria(processor)])
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            stopping_criteria=stopping_criteria
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()
