import os
import torch
from inspect_ai.model import (
    ModelAPI, modelapi, ModelOutput, ChatMessage, GenerateConfig
)
from inspect_ai.model._chat_message import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
from inspect_ai.tool import ToolInfo, ToolChoice

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from phase2.agents.tri_agent import TriAgentSystem

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
            
        # Global call counter to track progress through the dataset
        self.call_count = 0

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        # 1. Convert Inspect ChatMessages to standard HuggingFace dicts
        hf_messages = []
        system_content = ""
        user_content = ""
        for msg in input:
            if isinstance(msg, ChatMessageUser):
                hf_messages.append({"role": "user", "content": msg.text})
                user_content = msg.text
            elif isinstance(msg, ChatMessageAssistant):
                if msg.tool_calls:
                    hf_messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": t.function, "arguments": t.arguments}} for t in msg.tool_calls]})
                else:
                    hf_messages.append({"role": "assistant", "content": msg.text or ""})
            elif isinstance(msg, ChatMessageSystem):
                hf_messages.append({"role": "system", "content": msg.text})
                system_content = msg.text
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
            
        # JUDGE DETECTION: 
        is_judge = False
        judge_keywords = ["refusal", "judge", "score", "correctly", "evaluator", "grading", "following conversation"]
        if any(kw in system_content.lower() for kw in judge_keywords):
            is_judge = True
            
        if len(tools) == 0 and ("available tools" not in system_content.lower()):
            is_judge = True

        if is_judge:
            # Route Judge through safe run_qwen_inference to enforce stopping criteria and prevent Gaudi OOM
            from phase2.agents.sys_utils import run_qwen_inference
            
            output_text = run_qwen_inference(
                self.model, 
                self.processor, 
                hf_messages, 
                images=None, 
                max_tokens=256, # Force tight token bound
                temperature=0.0
            )
            prefix = "\n[JUDGE CALL]"
            
        else:
            # Multi-Agent inference for the executing Agent
            self.call_count += 1
            import json
            context_str = f"AVAILABLE TOOLS:\n{json.dumps(hf_tools, indent=2)}\n\nCHAT HISTORY:\n"
            instruction = user_content or "Achieve the user's objective using the tools."
            for m in hf_messages:
                context_str += f"{m['role'].upper()}: {m['content']}\n"
            
            agent_system = TriAgentSystem(benchmark="agentharm", max_retries=3)
            output_text = agent_system.execute_task(instruction=instruction, context=context_str, task_index=self.call_count)
            prefix = "[AGENT SESSION]"

        clean_out = output_text.strip()
        print(f"\n{prefix} Qwen3-VL (Call #{self.call_count}): {clean_out}")
        
        # Map Qwen's <tool_call> XML structures into inspect_ai ToolCall objects
        import json
        import re
        from inspect_ai.model import ToolCall

        parsed_tool_calls = []
        clean_text_output = output_text

        tool_call_patterns = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', output_text, re.DOTALL)
        
        # Fallback 1: Model output markdown codeblocks
        if not tool_call_patterns:
            tool_call_patterns = re.findall(r'```(?:json|xml)?\s*({.*?})\s*```', output_text, re.DOTALL)
            
        # Fallback 2: Model output bare json dict
        if not tool_call_patterns:
            bare_json = re.findall(r'(\{"name":\s*".*?",\s*"arguments":\s*"*\{.*?\}\"*})', output_text, re.DOTALL)
            if bare_json:
                tool_call_patterns = bare_json
        
        for tool_json_str in tool_call_patterns:
            try:
                clean_text_output = clean_text_output.replace(f"<tool_call>{tool_json_str}</tool_call>", "")
                tool_data = json.loads(tool_json_str)
                tool_name = tool_data.get("name", "")
                tool_args = tool_data.get("arguments", {})
                
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
                
        final_output = ModelOutput.from_content(
            model=self.model_name,
            content=clean_text_output.strip() if clean_text_output.strip() else "I invoked an interactive tool API request."
        )
        if parsed_tool_calls:
            final_output.choices[0].message.tool_calls = parsed_tool_calls
            
        return final_output
