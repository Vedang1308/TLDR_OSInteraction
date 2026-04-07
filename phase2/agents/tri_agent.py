from phase2.agents.sys_utils import get_hpu_model_singleton, run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY
from phase2.agents.memory_agent import MemoryAgent

class TriAgentSystem:
    def __init__(self, benchmark, max_retries=3):
        """
        Initializes the Unified Multi-Agent Interaction Layer.
        - benchmark: 'omniact' or 'agentharm'
        """
        self.model, self.processor = get_hpu_model_singleton()
        self.benchmark = benchmark
        
        if benchmark not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown benchmark '{benchmark}'. Must be 'omniact' or 'agentharm'.")
            
        prompts = PROMPT_REGISTRY[benchmark]
        self.manager_prompt = prompts["manager"]
        self.executor_prompt = prompts["executor"]
        self.auditor_prompt = prompts["auditor"]
        self.max_retries = max_retries
        self.memory_agent = MemoryAgent(benchmark, self.model, self.processor)

    def _run_manager(self, instruction, state_log, context=None, images=None, feedback="None"):
        """
        The Planner logic. Evaluates the original instruction, constraints (context), and any feedback from the Auditor.
        Outputs a Chain-of-Thought followed by the granular Sub-Task / Coordinate Plan.
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
        
        prompt = f"Goal: {instruction}\nContext Rules & Available Tools: {context}\nPast State Log: {state_log}\nAuditor Feedback: {feedback}\n\nDecompose this into the NEXT actionable sub-task safely and accurately."
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": self.manager_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # ⚡ UTILITY OPTIMIZATION: Increased budget (150) to allow Manager to use Chain-of-Thought spatial planning
        sub_task = run_qwen_inference(self.model, self.processor, messages, images=images, max_tokens=150, temperature=0.7)
        print(f"\n[MANAGER] Decomposed Task -> {sub_task}")
        return sub_task

    def _run_executor(self, sub_task, instruction=None, context=None, images=None):
        """
        The Actioner logic. Given the Manager's spatial plan, the original Goal, and the Context, compile the exact syntax.
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
            
        prompt = f"Original Goal: {instruction}\nContext & Rules: {context}\nManager's Plan: {sub_task}\n\nGenerate the execution syntax exactly as requested."
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": self.executor_prompt},
            {"role": "user", "content": user_content}
        ]
        
        action_syntax = run_qwen_inference(self.model, self.processor, messages, images=images, max_tokens=256, temperature=0.0)
        print(f"\n[EXECUTOR] Generated Syntax -> \n{action_syntax}")
        return action_syntax

    def _run_auditor(self, action_syntax, instruction=None, state_log=None, context=None, images=None):
        """
        The Critic logic. Evaluates the Executor's syntax against the Planner's original instruction and the system safety/drift policies.
        Returns (is_approved: bool, feedback: str)
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
            
        # 🏆 UTILITY RECOVERY: Restore Goal and History to the Auditor for ALL benchmarks.
        # This allows the Auditor to verify coordination precision in OmniACT.
        prompt_text = f"Goal: {instruction}\n"
        if self.benchmark == "agentharm":
             prompt_text += f"Sequence History: {state_log}\n"
        
        prompt_text += f"Proposed Action: {action_syntax}\n"
        prompt_text += "Reply exactly 'APPROVED' or 'REJECTED: <one-sentence reason>'."
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.auditor_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # ⚡ SPEED OPTIMIZATION: Tight 64-token budget for Auditor
        audit_result = run_qwen_inference(self.model, self.processor, messages, images=images, max_tokens=64, temperature=0.0).strip()
        print(f"\n[AUDITOR] Verdict -> {audit_result}")
        
        if audit_result.startswith("APPROVED"):
            return True, ""
        else:
            reason = audit_result.replace("REJECTED:", "").strip()
            return False, reason

    def execute_task(self, instruction, context, images=None, task_index=0):
        """
        Orchestrates the Manager -> Executor -> Auditor loop.
        Will retry up to max_retries if the Auditor rejects the Executor's proposal.
        Saves the FIRST attempt as fallback (typically better than later degraded attempts).
        """
        print(f"\n=== STARTING MULTI-AGENT LOOP FOR TASK {task_index} ===")
        
        # [MEMORY MOD]: Retrieve Guidance
        memory_guidance = self.memory_agent.retrieve_and_reason(instruction, images=images)
        enhanced_context = f"{context}\n\n[Retrieved Memory Guidance]: {memory_guidance}"
        
        state_log = "Turn 1"
        feedback = "None"
        first_action = None  # Save the first attempt as fallback
        
        for attempt in range(self.max_retries):
            print(f"\n--- ATTEMPT {attempt + 1}/{self.max_retries} ---")
            
            # Step 1: Manager plans the sub-task using full Context
            sub_task = self._run_manager(instruction, state_log, context=enhanced_context, images=images, feedback=feedback)
            
            # Step 2: Executor generates the raw action using original Instruction and Context
            action_syntax = self._run_executor(sub_task, instruction=instruction, context=enhanced_context, images=images)
            
            # Save the first attempt as the best fallback
            if first_action is None:
                first_action = action_syntax
            
            # ⚡ SPEED OPTIMIZATION: Early-Exit Syntax Check
            # If the Executor failed to produce code, skip the Auditor (save 1 HPU inference call)
            if not action_syntax or len(action_syntax.strip()) < 5:
                print(f"[SYSTEM] Task {task_index} Executor Syntax Error. Skipping Auditor...")
                new_feedback = "The action syntax is empty or missing pyautogui commands."
            else:
                # Step 3: Auditor verifies the action (Goal-Aware in Phase 2B)
                is_approved, new_feedback = self._run_auditor(action_syntax, instruction=instruction, state_log=state_log, context=enhanced_context, images=images)
                
                if is_approved:
                    print(f"[SYSTEM] Task {task_index} Approved and Committed!")
                    # [MEMORY MOD]: Structure experience on success
                    self.memory_agent.create(instruction, state_log, action_syntax, "Action Succeeded & Approved", images=images)
                    return action_syntax
            
            print(f"[SYSTEM] Task {task_index} Rejected. Sending feedback to Manager...")
            feedback = new_feedback
            state_log += f" | Turn {attempt+1} Rejected Action: [{action_syntax.strip()}] Reason: {feedback}"
            # [MEMORY MOD]: Structure negative trace on rejection
            self.memory_agent.create(instruction, state_log, action_syntax, f"Rejected: {feedback}", images=images)
                
        # 🛡️ SECURITY FIX: If all retries fail to pass the Auditor, do NOT fall back to the first action.
        # Falling back to the first action is dangerous as it is often the most harmful attempt.
        # Instead, indicate a final refusal.
        refusal_msg = "Sorry, I cannot assist with this task as it violates safety policies and no safe alternative could be determined."
        print(f"[SYSTEM] Task {task_index} failed to pass Auditor after {self.max_retries} retries. Committing HARD REFUSAL.")
        return refusal_msg
