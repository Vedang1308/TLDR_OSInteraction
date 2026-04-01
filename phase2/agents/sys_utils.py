import builtins
import torch

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

def run_qwen_inference(model, processor, messages, images=None, tools=None, max_tokens=1024, temperature=0.7):
    """
    Utility wrapper to natively execute Qwen3-VL inferences on the HPU.
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
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0
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
