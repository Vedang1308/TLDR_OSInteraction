import os
import json
import uuid
import sys
from phase2.agents.sys_utils import run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY

class MemoryAgent:
    def __init__(self, benchmark, model, processor):
        self.benchmark = benchmark
        self.model = model
        self.processor = processor
        self.prompts = PROMPT_REGISTRY.get("memory_agent", {})
        self.structuring_prompt = self.prompts.get("structuring", "")
        self.reasoning_prompt = self.prompts.get("reasoning", "")
        
        # Local JSON-based Memory Graph (Dictionary)
        self.memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"{benchmark}_memory_graph.json")
        self.memory_store = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                 try:
                      return json.load(f)
                 except:
                      return []
        return []

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_store, f, indent=4)

    def create(self, instruction, state_log, action, feedback, images=None):
        """Standardize and Structure episodic traces into semantic/procedural memory."""
        prompt = (f"Goal: {instruction}\nPast State: {state_log}\n"
                  f"Action Taken: {action}\nFeedback received: {feedback}\n\n"
                  f"Extract the core reusable insight.")
                  
        user_content = []
        if images:
             user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": self.structuring_prompt},
            {"role": "user", "content": user_content}
        ]
        
        insight = run_qwen_inference(self.model, self.processor, messages, images=images, max_tokens=100, temperature=0.2).strip()
        print(f"\n[MEMORY_AGENT] Extracted Knowledge -> {insight}")
        
        tags = set(instruction.lower().replace(".", "").replace(",", "").split())
        self.memory_store.append({
            "id": str(uuid.uuid4())[:8],
            "tags": list(tags)[:10],
            "insight": insight
        })
        self._save_memory()

    def retrieve_and_reason(self, instruction, images=None):
        """Retrieves using basic tag overlap and reasons over it."""
        if not self.memory_store:
            return "No previous memory available."
            
        # 1. Retrieval (Lexical / Tag based)
        query_tags = set(instruction.lower().replace(".", "").replace(",", "").split())
        scored_mems = []
        for mem in self.memory_store:
            overlap = len(query_tags.intersection(set(mem["tags"])))
            scored_mems.append((overlap, mem["insight"]))
            
        scored_mems.sort(key=lambda x: x[0], reverse=True)
        retrieved_insights = [m[1] for m in scored_mems[:3] if m[0] > 0]
        
        if not retrieved_insights:
            retrieved_insights = [m["insight"] for m in self.memory_store[-2:]]

        raw_memory_context = "\n- ".join(retrieved_insights)
        
        # 2. Reasoning (Synthesis)
        prompt = (f"Current Goal: {instruction}\nRetrieved Past Memories:\n- {raw_memory_context}\n\n"
                  f"Synthesize this into a short, exact directive.")
                  
        user_content = []
        if images:
             user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        messages = [
             {"role": "system", "content": self.reasoning_prompt},
             {"role": "user", "content": user_content}
        ]
        
        guidance = run_qwen_inference(self.model, self.processor, messages, images=images, max_tokens=150, temperature=0.1).strip()
        print(f"\n[MEMORY_AGENT] Synthesized Guidance -> {guidance}")
        return guidance
