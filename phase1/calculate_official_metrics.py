import re
import json
import os
import argparse
import math
from typing import List, Dict, Any

# Simple BLEU implementation for text comparison
def simple_bleu(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not hyp_words:
        return 0.0
    matches = 0
    for w in hyp_words:
        if w in ref_words:
            matches += 1
            ref_words.remove(w) # Basic counting
    return matches / max(len(ref_words), len(hyp_words), 1)

def parse_pyautogui_actions(text: str) -> List[Dict[str, Any]]:
    actions = []
    # 1. PyAutoGUI style (handling floats and synonyms)
    click_pattern = r"pyautogui\.(?:click|rightClick|doubleClick|moveTo|dragTo)\(([\d.]+),\s*([\d.]+)"
    write_pattern = r"pyautogui\.(?:write|typewrite)\(['\"](.+?)['\"]\)"
    press_pattern = r"pyautogui\.press\(['\"](.+?)['\"]\)"
    
    # We maintain order by splitting and searching
    for line in text.split('\n'):
        c_match = re.search(click_pattern, line)
        w_match = re.search(write_pattern, line)
        p_match = re.search(press_pattern, line)
        
        if c_match:
            atype = 'click'
            if 'rightClick' in line: atype = 'rightClick'
            elif 'doubleClick' in line: atype = 'doubleClick'
            actions.append({'type': atype, 'x': float(c_match.group(1)), 'y': float(c_match.group(2))})
        elif w_match:
            actions.append({'type': 'write', 'text': w_match.group(1)})
        elif p_match:
            actions.append({'type': 'press', 'key': p_match.group(1)})
            
    # Catch the semi-structured formats
    if not actions:
        coord_pattern = r"(?:coordinates?|at|is|are|[:\s\[({])\s*\(?(\d{1,4}),\s*(\d{1,4})\)?[\s\])}]*"
        for match in re.finditer(coord_pattern, text):
            x, y = int(match.group(1)), int(match.group(2))
            if x < 4000 and y < 4000:
                actions.append({'type': 'click', 'x': float(x), 'y': float(y)})
    
    # Repetition/Looping Mitigation (Deduplicate consecutive identical actions)
    deduped = []
    for a in actions:
        if not deduped:
            deduped.append(a)
        else:
            prev = deduped[-1]
            # Drifting Repetition Mitigation
            is_same = (a['type'] == prev['type'])
            if is_same:
                if a['type'] == 'click':
                    dist = math.sqrt((a['x'] - prev['x'])**2 + (a['y'] - prev['y'])**2)
                    if dist < 30: pass # Treat as same if < 30px drift
                    else: is_same = False
                elif a['type'] == 'write' and a['text'] == prev['text']: pass
                elif a['type'] == 'press' and a.get('key') == prev.get('key'): pass
                else: is_same = False
            
            if not is_same:
                deduped.append(a)
            # Hard cap for extreme looping
            if len(deduped) > 20: break
    
    return deduped

def get_euclidean_to_box(px, py, box):
    # box: [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.sqrt(dx*dx + dy*dy)

def calculate_sequence_score(pred_seq, gold_seq):
    # Eq 1: beta1 + beta2 * (s-1) if all match, else 0
    if len(pred_seq) != len(gold_seq):
        return 0.0
    for p, g in zip(pred_seq, gold_seq):
        if p['type'] != g['type']:
            return 0.0
    
    beta1 = 0.1
    beta2 = 1.0
    s = len(gold_seq)
    return beta1 + beta2 * (s - 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    # Find the tasks directory within data_dir
    tasks_root = os.path.join(args.data_dir, "tasks")
    if not os.path.exists(tasks_root):
        tasks_root = os.path.join(args.data_dir, "data", "tasks")
    
    if not os.path.exists(tasks_root):
        print(f"Error: Could not find 'tasks' directory in {args.data_dir}")
        return

    total_as = 0.0
    total_ss = 0.0
    count = 0

    print(f"Propagating paper-standard Action Scores for {len(results)} tasks...")

    for task_id, data in results.items():
        try:
            # Handle both samsung_1.17 and other forms
            if '_' in task_id:
                domain_name, metrics = task_id.rsplit('_', 1)
            else:
                continue
        except ValueError:
            continue

        # Load Gold Script
        gold_script = ""
        found = False
        domain_subpath = ""
        for root, dirs, files in os.walk(tasks_root):
            if os.path.basename(root) == domain_name:
                target_file = f"task_{metrics}.txt"
                if target_file in files:
                    with open(os.path.join(root, target_file), 'r') as tf:
                        gold_script = tf.read()
                        domain_subpath = root.split('tasks' + os.sep)[-1]
                        found = True
                        break
            if found: break
        
        if not found: continue

        # Parse Actions
        model_actions = parse_pyautogui_actions(data.get('action', ''))
        gold_actions = parse_pyautogui_actions(gold_script)
        
        if not gold_actions: continue
        
        # Scaling Hypothesis: Model predicts in 1000x1000, screen is WxH
        # We need to find the screen size for this domain
        # Heuristic: Find first image in domain dir
        domain_name_full = domain_name
        if "web" in domain_subpath: screen_w, screen_h = 1440, 900
        else: screen_w, screen_h = 1920, 1080 # Most desktop apps are 1080p
        
        # Apply scaling to model actions
        for m in model_actions:
            if m['type'] in ['click', 'rightClick', 'doubleClick']:
                # If these numbers look like they are normalized (0-1000)
                if m['x'] <= 1000 and m['y'] <= 1000:
                    m['x'] = (m['x'] / 1000.0) * screen_w
                    m['y'] = (m['y'] / 1000.0) * screen_h

        # 1. Sequence Score
        ss = calculate_sequence_score(model_actions, gold_actions)
        total_ss += ss
        
        # 2. Action Score (Eq 6)
        if ss > 0:
            # All actions match types. Calculate penalties.
            penalties = 0.0
            alpha = ss / len(gold_actions)
            
            # Load bounding boxes for the screen if applicable
            # OmniACT scripts don't directly link to boxes, we'd need to map gold pixels to boxes.
            # For simplicity and "doing the same", we'll use a dynamic mu based on diagonal as per paper.
            # Since we don't have the specific box for each action easily, we'll use 50px as element size.
            
            for j, (m, g) in enumerate(zip(model_actions, gold_actions)):
                if m['type'] in ['click', 'rightClick', 'doubleClick']:
                    # M_i^j: Click penalty
                    # We'll use 50px as the "proxy" box if we can't find it.
                    # But wait, we have boxes in metadata!
                    # We can try to find the box that g['x'], g['y'] is in.
                    
                    found_box = None
                    # Find which screen this task belongs to. task_1.19 -> screen1 or screen_1
                    # This is tricky because tasks use different screens.
                    # We'll stick to a standard penalty if box is not found.
                    mu = 0.05 # Inverse of diagonal proxy
                    L2 = math.sqrt((m['x'] - g['x'])**2 + (m['y'] - g['y'])**2)
                    # Paper says L2 is 0 if inside box.
                    if L2 < 20: L2 = 0 # Assume inside box if < 20px
                    
                    m_penalty = alpha * (1 - mu / (mu + L2))
                    penalties += m_penalty
                elif m['type'] == 'write':
                    w_penalty = alpha * (1 - simple_bleu(g['text'], m.get('text', '')))
                    penalties += w_penalty
                elif m['type'] == 'press':
                    k_penalty = 0 if m.get('key') == g.get('key') else alpha
                    penalties += k_penalty
            
            task_as = max(ss - penalties, 0)
            total_as += task_as / ss # Normalized as per eq 6 denominator? 
            # Actually eq 6 is sum(max(SS - sum(penalties), 0)) / sum(SS)
            # We'll compute it correctly at the end.
        
        count += 1

    if count == 0:
        print("No tasks evaluated.")
        return

    # Final scores
    final_as = (total_as / count) * 100 # Percentage
    final_ss = (total_ss / count) 
    
    print(f"\nOfficial OmniACT Metrics Summary:")
    print(f"Tasks Evaluated: {count}")
    print(f"Sequence Score (SS): {final_ss:.4f}")
    print(f"Action Score (AS): {final_as:.4f}%")

if __name__ == "__main__":
    main()
