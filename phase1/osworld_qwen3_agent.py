import json
import os
import sys
from typing import Dict, Any

# =======================================================================
# IMPORT THE INTEL GAUDI WRAPPER
# Ensure that this script can see `eval_wrapper.py` in its python path!
# =======================================================================
try:
    from eval_wrapper import load_vlm_model, detect_device
except ImportError:
    print("FATAL ERROR: Could not find eval_wrapper.py.")
    print("Please make sure osworld_qwen3_agent.py and eval_wrapper.py are in the same directory, or that phase1 is in your PYTHONPATH.")
    sys.exit(1)

# Because we don't have OSWorld physically imported right here, we structure the Agent dynamically
# If you place this script inside the OSWorld cloned repo, you can do: `from OSWorld.agent import Agent`
class Qwen3OSWorldAgent:
    """
    Physical Integration Bridge linking the complex OSWorld virtual machine 
    evaluation loop to our proven Intel Gaudi VLM memory allocators.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """
        Initializes the model explicitly utilizing our hardware wrapper scripts!
        """
        print(f"[OSWorld-Agent] Initializing hardware bridge for {model_name}...")
        
        # 1. Detect Hardware native layout
        self.device = detect_device()
        
        # 2. Boot the VLM onto the actual processor tiles
        self.model, self.processor = load_vlm_model(model_name, self.device)
        
        print("[OSWorld-Agent] Processor bridge fully instantiated!")

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        OSWorld's `run.py` calls this method continually until the environment reaches a terminal state.
        
        State Dictionary Keys typically include:
        - state["screenshot"]: A PIL Image object or raw bytes of the virtual Ubuntu Desktop
        - state["instruction"]: The user objective (e.g. "Create a new folder named 'test'")
        - state["history"]: Previous actions taken by the agent
        """
        
        # Extract environment variables
        instruction = state.get("instruction", "")
        screenshot = state.get("screenshot", None)
        
        # 1. --- FORMAT THE PROMPT FOR QWEN3-VL ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Based on the screenshot, execute the following instruction. "
                                             f"Output ONLY a single command in one of these formats: "
                                             f"click(x, y), type('text'), drag(x1, y1, x2, y2), or scroll(up/down). "
                                             f"The Desktop screen resolution is 1920x1080. "
                                             f"\nTask Instruction: {instruction}"}
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # 2. --- GENERATE TOKENS USING THE HPU ---
        # Convert the virtual Desktop screenshot into tensor embeddings
        inputs = self.processor(text=[text_prompt], images=[screenshot], padding=True, return_tensors="pt").to(self.device)
        
        # Dispatch to the Hardware Compute Layer
        generated_ids = self.model.generate(**inputs, max_new_tokens=150)
        
        # Clean output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 3. --- REVERSE-MAP TEXT INTO OSWORLD ACTION ---
        action_dict = {
            "action_type": "QWEN_GAUDI_ACTION",
            "raw_hpu_output": generated_text
        }
        
        return action_dict

    def reset(self):
        """
        Called when OSWorld resets the environment between evaluation tasks.
        Clears any conversational history your agent might be aggressively caching.
        """
        pass
