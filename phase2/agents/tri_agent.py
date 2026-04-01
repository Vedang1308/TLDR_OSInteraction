from phase2.agents.sys_utils import get_hpu_model_singleton, run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY

class TriAgentSystem:
    def __init__(self, benchmark, max_retries=3):
        """
        Initializes the Unified Multi-Agent Interaction Layer.
        - benchmark: 'omniact' or 'agentharm'
        """
        self.model, self.processor = get_hpu_model_singleton()
        
        if benchmark not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown benchmark '{benchmark}'. Must be 'omniact' or 'agentharm'.")
            
        prompts = PROMPT_REGISTRY[benchmark]
        self.manager_prompt = prompts["manager"]
        self.executor_prompt = prompts["executor"]
        self.auditor_prompt = prompts["auditor"]
        self.max_retries = max_retries

    def _run_manager(self, instruction, state_log, images=None, feedback="None"):
        """
        The Planner logic. Evaluates the original instruction and any feedback from the Auditor.
        Returns the granular Sub-Task.
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Instruction: {instruction}\nPast State Log: {state_log}\nAuditor Feedback: {feedback}\n\nDecompose this into the NEXT actionable sub-task. Output ONLY the sub-task description."})

        messages = [
            {"role": "system", "content": self.manager_prompt},
            {"role": "user", "content": user_content}
        ]
        
        sub_task = run_qwen_inference(self.model, self.processor, messages, images=images, temperature=0.7)
        print(f"\n[MANAGER] Decomposed Task -> {sub_task}")
        return sub_task

    def _run_executor(self, sub_task, context, images=None):
        """
        The Actioner logic. Given the granular sub-task from the Manager and the current context (screenshot/tool state), generate the exact syntax.
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Context: {context}\nSub-Task: {sub_task}\n\nGenerate the execution syntax exactly as requested."})

        messages = [
            {"role": "system", "content": self.executor_prompt},
            {"role": "user", "content": user_content}
        ]
        
        action_syntax = run_qwen_inference(self.model, self.processor, messages, images=images, temperature=0.0) # Greedy for execution precision
        print(f"\n[EXECUTOR] Generated Syntax -> \n{action_syntax}")
        return action_syntax

    def _run_auditor(self, action_syntax, instruction, context, images=None):
        """
        The Critic logic. Evaluates the Executor's syntax against the Planner's original instruction and the system safety/drift policies.
        Returns (is_approved: bool, feedback: str)
        """
        user_content = []
        if images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Goal: {instruction}\nContext: {context}\nProposed Action: {action_syntax}\n\nEvaluate this action. If it is safe and solves the goal, reply strictly with 'APPROVED'. If not, reply with 'REJECTED: <reason>'."})

        messages = [
            {"role": "system", "content": self.auditor_prompt},
            {"role": "user", "content": user_content}
        ]
        
        audit_result = run_qwen_inference(self.model, self.processor, messages, images=images, temperature=0.0).strip()
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
        """
        print(f"\n=== STARTING MULTI-AGENT LOOP FOR TASK {task_index} ===")
        state_log = "Turn 1"
        feedback = "None"
        
        for attempt in range(self.max_retries):
            print(f"\n--- ATTEMPT {attempt + 1}/{self.max_retries} ---")
            
            # Step 1: Manager plans the sub-task
            sub_task = self._run_manager(instruction, state_log, images=images, feedback=feedback)
            
            # Step 2: Executor generates the raw action
            action_syntax = self._run_executor(sub_task, context, images=images)
            
            # Step 3: Auditor verifies the action
            is_approved, new_feedback = self._run_auditor(action_syntax, instruction, context, images=images)
            
            if is_approved:
                print(f"[SYSTEM] Task {task_index} Approved and Committed!")
                return action_syntax
            else:
                print(f"[SYSTEM] Task {task_index} Rejected by Auditor. Sending feedback to Manager...")
                feedback = new_feedback
                state_log += f" | Turn {attempt+1} Failed: {feedback}"
                
        print(f"[SYSTEM] Task {task_index} failed to pass Auditor after {self.max_retries} retries. Committing last known action.")
        return action_syntax
