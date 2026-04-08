import re
import math
from phase2.agents.sys_utils import get_hpu_model_singleton, run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY

class OmniactAgentSystemV2:
    def __init__(self, benchmark="omniact", num_samples=5, temperature=0.7):
        """
        Implements the V2 Architecture: Density-Based Consensus with strict Integer Coordinates.
        Replaces the blind Visual Critic that caused performance drops.
        """
        self.model, self.processor = get_hpu_model_singleton()
        self.benchmark = benchmark
        self.num_samples = num_samples
        self.temperature = temperature
        
        if benchmark not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown benchmark '{benchmark}'.")
            
        try:
            model_name = getattr(self.model.config, "_name_or_path", "").lower()
            if not model_name:
                model_name = getattr(self.model, "name_or_path", "").lower()
        except Exception:
            model_name = ""
            
        use_prompt = "omniact_executor_v2"
        if benchmark == "omniact":
            if "2b" in model_name:
                use_prompt = "omniact_executor_2b"
                self.num_samples = 10
            elif "4b" in model_name:
                use_prompt = "omniact_executor_4b"
                self.num_samples = 5
            elif "8b" in model_name:
                use_prompt = "omniact_executor_8b"
                self.num_samples = 3
                
        prompts = PROMPT_REGISTRY[benchmark]
        self.system_prompt = prompts.get(use_prompt, prompts.get("omniact_executor_v2", ""))

    def _extract_coordinates(self, script):
        """Regex to pull (x, y) coordinates from PyAutoGUI syntax."""
        match = re.search(r"click\(([\d.]+),\s*([\d.]+)\)", script)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return None

    def execute_task(self, instruction, context, images=None, task_index=0, ui_map=None):
        if ui_map is None:
            ui_map = {}
            
        print(f"\n=== [OmniACT-V2] Starting High-Temp Density Consensus Loop (N={self.num_samples}, T={self.temperature}) for task {task_index} ===")
        
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
                temperature=self.temperature,
                force_temperature=True
            )
            actions.append(action_syntax)
            
            # Intercept Set-of-Mark tag structure
            tag_match = re.search(r"click\((\d+)\)", action_syntax)
            if tag_match:
                tag_id = int(tag_match.group(1))
                if tag_id in ui_map:
                    cx, cy = ui_map[tag_id]
                    coord = (float(cx), float(cy))
                    coords_list.append((coord, action_syntax))
                    print(f"  -> [Sample {i+1}] (SoM Tag [{tag_id}] Intercepted) {action_syntax.strip().split(maxsplit=1)[0]}...") 
                    continue
            
            coord = self._extract_coordinates(action_syntax)
            if coord:
                coords_list.append((coord, action_syntax))
                
            print(f"  -> [Sample {i+1}] {action_syntax.strip().split(maxsplit=1)[0]}...") 
            
        if not coords_list:
            print("[OmniACT-V2] No valid clicks generated. Returning best-effort text output.")
            return actions[0] if actions else ""
            
        # Hardware-Aware Spatial Clustering 
        try:
            model_name = getattr(self.model.config, "_name_or_path", "").lower()
            if not model_name:
                model_name = getattr(self.model, "name_or_path", "").lower()
        except Exception:
            model_name = ""
            
        threshold = 30
        if "2b" in model_name:
            threshold = 45 
        elif "4b" in model_name:
            threshold = 25 # Optimized sweet-spot for 4B model performance
        elif "8b" in model_name:
            threshold = 15 # Tightened 8B threshold to prevent false cluster splitting
            
        clusters = [] 
        
        for item in coords_list:
            coord, action = item
            added = False
            for cluster in clusters:
                center_x = sum(c[0][0] for c in cluster) / len(cluster)
                center_y = sum(c[0][1] for c in cluster) / len(cluster)
                
                dist = math.sqrt((coord[0] - center_x)**2 + (coord[1] - center_y)**2)
                if dist <= threshold:
                    cluster.append(item)
                    added = True
                    break
            if not added:
                clusters.append([item])
                
        # Sort clusters by Density (size), resolve ties by lowest variance (tighter cluster)
        def cluster_variance(cluster):
            if len(cluster) == 1:
                return 0.0
            cx = sum(c[0][0] for c in cluster) / len(cluster)
            cy = sum(c[0][1] for c in cluster) / len(cluster)
            return sum((c[0][0] - cx)**2 + (c[0][1] - cy)**2 for c in cluster) / len(cluster)

        clusters.sort(key=lambda c: (len(c), -cluster_variance(c)), reverse=True)
        
        best_cluster = clusters[0]
        avg_x = sum(c[0][0] for c in best_cluster) / len(best_cluster)
        avg_y = sum(c[0][1] for c in best_cluster) / len(best_cluster)
        
        # Cast to int to perfectly map to official parsers and avoid float regressions
        int_x, int_y = int(round(avg_x)), int(round(avg_y))
        rep_action = best_cluster[0][1]
        
        # Rigorous integer substitution for both tagged and untagged
        optimized_action = re.sub(r"click\(\d+\)", f"click({int_x}, {int_y})", rep_action, count=1)
        optimized_action = re.sub(r"click\([\d.]+,\s*[\d.]+\)", f"click({int_x}, {int_y})", optimized_action, count=1)
        
        print(f"[OmniACT-V2] Selected Cluster 1 (Size: {len(best_cluster)}). Final Action: {optimized_action.strip()}")
        return optimized_action
