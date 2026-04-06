import re
import math
from phase2.agents.sys_utils import get_hpu_model_singleton, run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY

class OmniactAgentSystem:
    def __init__(self, benchmark="omniact", num_samples=3, temperature=0.7):
        """
        Implements the Tri-Agent Visual Critic & Spatial Centroid Averaging for OmniACT.
        - num_samples: How many independent instinct shots to generate.
        - temperature: Kept high (0.7) to allow distribution diversity.
        """
        self.model, self.processor = get_hpu_model_singleton()
        self.benchmark = benchmark
        self.num_samples = num_samples
        self.temperature = temperature
        
        if benchmark not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown benchmark '{benchmark}'.")
            
        # Using a specialized prompt for OmniACT Self-Trust
        self.system_prompt = PROMPT_REGISTRY[benchmark].get("omniact_executor", "")
        self.auditor_prompt = PROMPT_REGISTRY[benchmark].get("auditor", "")
        
    def _extract_coordinates(self, script):
        """Regex to pull (x, y) coordinates from PyAutoGUI syntax, supporting floats."""
        match = re.search(r"click\(([\d.]+),\s*([\d.]+)\)", script)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return None

    def execute_task(self, instruction, context, images=None, task_index=0):
        print(f"\n=== [OmniACT-TriAgent] Starting High-Temp Consensus & Visual Critic Loop (N={self.num_samples}, T={self.temperature}) for task {task_index} ===")
        
        user_content = []
        if images:
             user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Instruction: {instruction}\nContext: {context}\n\nExecute the instruction."})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        actions = []
        coords_list = []
        
        for i in range(self.num_samples):
            action_syntax = run_qwen_inference(
                self.model, 
                self.processor, 
                messages, 
                images=images, 
                max_tokens=256, 
                temperature=self.temperature
            )
            actions.append(action_syntax)
            
            coord = self._extract_coordinates(action_syntax)
            if coord:
                coords_list.append((coord, action_syntax))
                
            print(f"  -> [Sample {i+1}] {action_syntax.strip().split(maxsplit=1)[0]}...") # Print snippet 
            
        if not coords_list:
            print("[OmniACT-TriAgent] No valid clicks generated. Returning first fallback output.")
            return actions[0] if actions else ""
            
        # Spatial Clustering with Model-Aware Thresholding
        try:
            model_name = getattr(self.model.config, "_name_or_path", "").lower()
            if not model_name:
                model_name = getattr(self.model, "name_or_path", "").lower()
        except Exception:
            model_name = ""
            
        threshold = 30 # default
        if "2b" in model_name:
            threshold = 45 # High variance, needs wider net
        elif "4b" in model_name:
            threshold = 30 # Moderate precision
        elif "8b" in model_name:
            threshold = 15 # High precision, enforcing strict lock
            
        clusters = [] 
        
        for item in coords_list:
            coord, action = item
            added = False
            for cluster in clusters:
                # Calculate cluster center
                center_x = sum(c[0][0] for c in cluster) / len(cluster)
                center_y = sum(c[0][1] for c in cluster) / len(cluster)
                
                # Euclidean distance
                dist = math.sqrt((coord[0] - center_x)**2 + (coord[1] - center_y)**2)
                if dist <= threshold:
                    cluster.append(item)
                    added = True
                    break
            if not added:
                clusters.append([item])
                
        # Sort clusters by size (descending)
        clusters.sort(key=len, reverse=True)
        
        for cluster_idx, current_cluster in enumerate(clusters):
            # Centroid Averaging
            avg_x = sum(c[0][0] for c in current_cluster) / len(current_cluster)
            avg_y = sum(c[0][1] for c in current_cluster) / len(current_cluster)
            
            # Take the representative script and replace its click coordinate with the averaged one
            rep_action = current_cluster[0][1]
            # Replace the first click
            optimized_action = re.sub(r"click\([\d.]+,\s*[\d.]+\)", f"click({avg_x:.1f}, {avg_y:.1f})", rep_action, count=1)
            
            print(f"\n[OmniACT-TriAgent] Evaluating Cluster {cluster_idx+1} (Size: {len(current_cluster)}) with Centroid-Optimized Action:")
            print(f"Proposed: {optimized_action.strip()}")
            
            # --- Visual Critic Loop ---
            critic_user_content = []
            if images:
                 critic_user_content.append({"type": "image"})
            critic_user_content.append({
                "type": "text", 
                "text": f"Instruction: {instruction}\n\nProposed Action:\n{optimized_action.strip()}\n\nDoes this action accurately target the goal based on the screen? Output APPROVED or REJECTED: <reason>."
            })
            
            critic_messages = [
                {"role": "system", "content": self.auditor_prompt},
                {"role": "user", "content": critic_user_content}
            ]
            
            critic_feedback = run_qwen_inference(
                self.model,
                self.processor,
                critic_messages,
                images=images,
                max_tokens=64,
                temperature=0.0 # Strict verification
            )
            
            print(f"  --> Visual Critic Feedback: {critic_feedback.strip()}")
            
            if "APPROVED" in critic_feedback.upper():
                print(f"[OmniACT-TriAgent] Action APPROVED! Locking in coordinates.")
                return optimized_action
            else:
                print(f"[OmniACT-TriAgent] Action REJECTED. Moving to next cluster if available.")
                
        # Fallback if ALL clusters are rejected
        print("[OmniACT-TriAgent] All clustered actions rejected by Visual Critic. Falling back to the largest cluster's optimized action.")
        
        avg_x = sum(c[0][0] for c in clusters[0]) / len(clusters[0])
        avg_y = sum(c[0][1] for c in clusters[0]) / len(clusters[0])
        fallback_action = re.sub(r"click\([\d.]+,\s*[\d.]+\)", f"click({avg_x:.2f}, {avg_y:.2f})", clusters[0][0][1], count=1)
        
        return fallback_action
