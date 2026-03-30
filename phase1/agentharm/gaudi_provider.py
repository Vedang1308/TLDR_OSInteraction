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
        pass

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
        # Check if this generation is for the Judge or the Agent by safely checking tool usage.
        # AgentHARM sandbox executes 80+ tools natively. The offline Evaluator Judges use ZERO tools.
        is_judge = (len(tools) == 0)
        
        if is_judge:
            prefix = "\n[JUDGE]"
        else:
            prefix = "[AGENT]"
            
        clean_out = output_text.strip()
        print(f"\n{prefix} Qwen3-VL: {clean_out}")
        
        # Because we bypassed vLLM, the raw PyTorch text engine returns pure strings.
        # We must manually extract Qwen's <tool_call> XML structures and map them 
        # securely into the inspect_ai `ToolCall` object registry so the sandbox executes them!
        import json
        import re
        from inspect_ai.model import ToolCall

        parsed_tool_calls = []
        clean_text_output = output_text

        # Qwen-specific XML tool parsing regex
        tool_call_patterns = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', output_text, re.DOTALL)
        
        for tool_json_str in tool_call_patterns:
            try:
                # Remove it so the text reasoning is uniquely isolated from the XML block
                clean_text_output = clean_text_output.replace(f"<tool_call>{tool_json_str}</tool_call>", "")
                
                tool_data = json.loads(tool_json_str)
                tool_name = tool_data.get("name", "")
                tool_args = tool_data.get("arguments", {})
                
                # Assign a deterministic ID for sandbox routing
                import uuid
                call_id = f"call_{str(uuid.uuid4())[:8]}"
                
                parsed_tool_calls.append(
                    ToolCall(
                        id=call_id,
                        function=tool_name,
                        arguments=tool_args,
                        type="function"
                    )
                )
            except Exception as e:
                print(f"[GAUDI PARSER FATAL ERROR]: FAILED TO DECODE QWEN TOOL XML: {e}")
                
        # Return physical ToolCall arrays so the AISI framework can actually invoke the APIs
        final_output = ModelOutput.from_content(
            model=self.model_name,
            content=clean_text_output.strip() if clean_text_output.strip() else "I invoked an interactive tool API request."
        )
        if parsed_tool_calls:
            final_output.choices[0].message.tool_calls = parsed_tool_calls
            
        return final_output
