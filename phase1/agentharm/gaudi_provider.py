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
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=api_key_vars, **model_args)
        
        # Singleton loading to prevent OOM
        if not hasattr(GaudiQwenModelAPI, "_model_loaded"):
            print(f"Loading '{model_name}' natively on Gaudi HPU...")
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # Using exact architecture required for OmniACT
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, config=config, trust_remote_code=True
            ).eval()
            self.model.to("hpu")
            
            GaudiQwenModelAPI._model_loaded = True
            GaudiQwenModelAPI._shared_model = self.model
            GaudiQwenModelAPI._shared_processor = self.processor
            print("Successfully loaded model!")
        else:
            self.model = GaudiQwenModelAPI._shared_model
            self.processor = GaudiQwenModelAPI._shared_processor

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
        
        # If model outputs tool calls structurally inside text (e.g. Qwen format), 
        # Inspect natively parses it via tool_choice / system prompt regex if we return it as text!
        return ModelOutput.from_content(
            model=self.model_name,
            content=output_text
        )
