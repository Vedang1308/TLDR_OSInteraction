import re
import math
from phase2.agents.sys_utils import get_hpu_model_singleton, run_qwen_inference
from phase2.agents.prompts import PROMPT_REGISTRY
from phase2.agents.memory_agent import MemoryAgent

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
            
        prompts = PROMPT_REGISTRY[benchmark]
        # Using safely framed prompt to avoid AgentHarm-style refusals
        self.system_prompt = prompts.get("omniact_executor_v2", "")
        
        # [MEMORY MOD]: Instantiate the Memory Agent for episodic recall
        self.memory_agent = MemoryAgent(benchmark, self.model, self.processor)

    def execute_task(self, instruction, context, images=None, task_index=0, ui_map=None):
        if ui_map is None:
            ui_map = {}
            
        print(f"\n=== [OmniACT-V2] Starting High-Temp Density Consensus Loop (N={self.num_samples}, T={self.temperature}) for task {task_index} ===")
        
        # [MEMORY MOD]: Retrieve guidance from past experiences before inference
        memory_guidance = self.memory_agent.retrieve_and_reason(instruction, images=images)
        enhanced_context = f"{context}\n\n[Memory Guidance from past tasks]: {memory_guidance}"
        
        user_content = []
        if images:
             user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Instruction: {instruction}\nContext: {enhanced_context}\n\nExecute the instruction."})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Run batched inference to eliminate sequential latency
        actions = run_qwen_inference(
            self.model, 
            self.processor, 
            messages, 
            images=images, 
            max_tokens=256, 
            temperature=self.temperature,
            num_return_sequences=self.num_samples
        )
        
        if self.num_samples == 1:
            actions = [actions]
            
        parsed_samples = []
        for i, action_syntax in enumerate(actions):
            # Parse Set-of-Mark tags
            tag_matches = list(re.finditer(r"(click|doubleClick|rightClick|moveTo|dragTo)\((\d+)\)", action_syntax))
            # Parse raw spatial coordinates
            raw_matches = list(re.finditer(r"(click|doubleClick|rightClick|moveTo|dragTo)\(([\d.]+),\s*([\d.]+)\)", action_syntax))
            
            # Print for debugging
            preview = action_syntax.strip().split(maxsplit=1)[0] if action_syntax.strip() else "Empty"
            print(f"  -> [Sample {i+1}] {preview}...")
            
            for match in tag_matches:
                verb = match.group(1)
                tag_id = int(match.group(2))
                if tag_id in ui_map:
                    cx, cy = ui_map[tag_id]
                    parsed_samples.append({
                        'coord': (float(cx), float(cy)),
                        'action': action_syntax,
                        'syntax_to_replace': match.group(0),
                        'verb': verb,
                        'is_som': True
                    })
                    print(f"    -> Intercepted valid SoM Tag [{tag_id}] for {verb}")
                    
            for match in raw_matches:
                verb = match.group(1)
                x = float(match.group(2))
                y = float(match.group(3))
                parsed_samples.append({
                    'coord': (x, y),
                    'action': action_syntax,
                    'syntax_to_replace': match.group(0),
                    'verb': verb,
                    'is_som': False
                })

        valid_som_tags = [s for s in parsed_samples if s['is_som']]
        if valid_som_tags:
            # Short-circuit density clustering: prioritize correct SoM mode
            tag_counts = {}
            for s in valid_som_tags:
                tag_counts[s['syntax_to_replace']] = tag_counts.get(s['syntax_to_replace'], 0) + 1
                
            best_tag_syntax = max(tag_counts, key=tag_counts.get)
            best_sample = next(s for s in valid_som_tags if s['syntax_to_replace'] == best_tag_syntax)
            
            int_x, int_y = int(round(best_sample['coord'][0])), int(round(best_sample['coord'][1]))
            verb = best_sample['verb']
            optimized_action = best_sample['action'].replace(best_tag_syntax, f"{verb}({int_x}, {int_y})", 1)
            
            print(f"[OmniACT-V2] Consensus selected precise SoM tag '{best_tag_syntax}'. Final Action: {optimized_action.strip()}")
            # [MEMORY MOD]: Store SoM consensus result
            self.memory_agent.create(instruction, "V2-SoM", optimized_action, "SoM Consensus Selected", images=images)
            return optimized_action
            
        raw_samples = [s for s in parsed_samples if not s['is_som']]
        if not raw_samples:
            print("[OmniACT-V2] No valid spatial actions found. Returning best-effort text output.")
            best_effort = actions[0] if actions else ""
            self.memory_agent.create(instruction, "V2-NoAction", best_effort, "No spatial actions parsed", images=images)
            return best_effort
            
        # Unified Distance Thresholding
        threshold = 30.0
        clusters = [] 
        
        for item in raw_samples:
            coord = item['coord']
            added = False
            for cluster in clusters:
                center_x = sum(c['coord'][0] for c in cluster) / len(cluster)
                center_y = sum(c['coord'][1] for c in cluster) / len(cluster)
                
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
            cx = sum(c['coord'][0] for c in cluster) / len(cluster)
            cy = sum(c['coord'][1] for c in cluster) / len(cluster)
            return sum((c['coord'][0] - cx)**2 + (c['coord'][1] - cy)**2 for c in cluster) / len(cluster)

        clusters.sort(key=lambda c: (len(c), -cluster_variance(c)), reverse=True)
        
        best_cluster = clusters[0]
        avg_x = sum(c['coord'][0] for c in best_cluster) / len(best_cluster)
        avg_y = sum(c['coord'][1] for c in best_cluster) / len(best_cluster)
        
        # Cast to int to perfectly map to official parsers and avoid float regressions
        int_x, int_y = int(round(avg_x)), int(round(avg_y))
        rep_action = best_cluster[0]['action']
        syntax_to_replace = best_cluster[0]['syntax_to_replace']
        verb = best_cluster[0]['verb']
        
        optimized_action = rep_action.replace(syntax_to_replace, f"{verb}({int_x}, {int_y})", 1)
        
        print(f"[OmniACT-V2] Selected Cluster 1 (Size: {len(best_cluster)}). Final Action: {optimized_action.strip()}")
        # [MEMORY MOD]: Store density-consensus result
        self.memory_agent.create(instruction, f"V2-Cluster(size={len(best_cluster)})", optimized_action, "Density Consensus Selected", images=images)
        return optimized_action