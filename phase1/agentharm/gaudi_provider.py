import os
import torch
from inspect_ai.model import (
    ModelAPI, modelapi, ModelOutput, ChatMessage, GenerateConfig
)
from inspect_ai.model._chat_message import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
from inspect_ai.tool import ToolInfo, ToolChoice

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

@modelapi(name="gaudi_qwen")
class GaudiQwenModelAPI(ModelAPI):
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, api_key_vars: list[str] = None, **model_args):
        api_key_vars = api_key_vars or []
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=api_key_vars, **model_args)
        
        # Singleton loading to prevent OOM
        import builtins
        if hasattr(builtins, "__GAUDI_CACHED_MODEL__"):
            print("GaudiProvider retrieved globally warmed up HPU active model.")
            self.model = builtins.__GAUDI_CACHED_MODEL__
            self.processor = builtins.__GAUDI_CACHED_PROCESSOR__
        else:
            raise RuntimeError("GaudiProvider failed to locate warmed-up PyTorch model! Run run_agentharm.py directly.")
            
        # Tracking variables to manually stream the task progress cleanly to stdout
        if getattr(GaudiQwenModelAPI, "_completed_tasks", None) is None:
            GaudiQwenModelAPI._completed_tasks = 0

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        # 1. Convert Inspect ChatMessages to standard HuggingFace dicts
        hf_messages = []
        for msg in input:
            if isinstance(msg, ChatMessageUser):
                hf_messages.append({"role": "user", "content": msg.text})
            elif isinstance(msg, ChatMessageAssistant):
                if msg.tool_calls:
                    hf_messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": t.function, "arguments": t.arguments}} for t in msg.tool_calls]})
                else:
                    hf_messages.append({"role": "assistant", "content": msg.text or ""})
            elif isinstance(msg, ChatMessageSystem):
                hf_messages.append({"role": "system", "content": msg.text})
            elif isinstance(msg, ChatMessageTool):
                hf_messages.append({"role": "tool", "content": msg.text, "name": msg.function})
        
        # 2. Format with tool schemas if provided
        hf_tools = []
        for t in tools:
            hf_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters.model_dump()
                }
            })
            
        text_prompt = self.processor.apply_chat_template(
            hf_messages, 
            tools=hf_tools if hf_tools else None,
            add_generation_prompt=True
        )
        
        inputs = self.processor(text=[text_prompt], images=None, padding=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=config.max_tokens or 1024,
                temperature=config.temperature or 0.0,
                do_sample=config.temperature and config.temperature > 0
            )
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Intercept and stream the raw output transparently to the User Terminal
        # Check if this generation is for the Judge or the Agent by scanning the system prompt
        is_judge = any("judge" in msg.text.lower() for msg in input if hasattr(msg, 'text') and msg.text)
        
        if is_judge:
            GaudiQwenModelAPI._completed_tasks += 0.5
            prefix = f"[JUDGE | Task Eval {int(GaudiQwenModelAPI._completed_tasks)}/176]"
        else:
            prefix = "[AGENT]"
            
        clean_out = output_text.strip().replace('\n', ' | ')
        print(f"{prefix} Qwen3-VL: {clean_out[:300]}{'...' if len(clean_out) > 300 else ''}")
        
        # If model outputs tool calls structurally inside text (e.g. Qwen format), 
        # Inspect natively parses it via tool_choice / system prompt regex if we return it as text!
        return ModelOutput.from_content(
            model=self.model_name,
            content=output_text
        )
