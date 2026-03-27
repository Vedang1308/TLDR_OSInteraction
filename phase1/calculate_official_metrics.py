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

def process_result_file(results_path, data_dir):
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find the tasks directory within data_dir
    tasks_root = os.path.join(data_dir, "tasks")
    if not os.path.exists(tasks_root):
        tasks_root = os.path.join(data_dir, "data", "tasks")
    
    if not os.path.exists(tasks_root):
        return None

    total_as = 0.0
    total_ss = 0.0
    count = 0
    skip_missing = 0
    skip_empty_gold = 0

    for task_id, data in results.items():
        try:
            # Load Gold Script directly using the relative path in task_id
            task_rel_path = task_id.replace("__", os.sep)
            gold_path = os.path.join(tasks_root, task_rel_path)
            
            if not os.path.exists(gold_path):
                skip_missing += 1
                continue

            with open(gold_path, 'r') as tf:
                gold_script = tf.read()
                domain_subpath = os.path.dirname(task_rel_path)
        except Exception as e:
            skip_missing += 1
            continue

        # Parse Actions
        model_actions = parse_pyautogui_actions(data.get('action', ''))
        gold_actions = parse_pyautogui_actions(gold_script)
        
        if not gold_actions:
            skip_empty_gold += 1
            continue
        
        # Scaling Hypothesis: 1000x1000 -> Screen Pixels
        if "web" in domain_subpath: screen_w, screen_h = 1440, 900
        else: screen_w, screen_h = 1920, 1080
        
        for m in model_actions:
            if m['type'] in ['click', 'rightClick', 'doubleClick']:
                if m['x'] <= 1000 and m['y'] <= 1000:
                    m['x'] = (m['x'] / 1000.0) * screen_w
                    m['y'] = (m['y'] / 1000.0) * screen_h

        ss = calculate_sequence_score(model_actions, gold_actions)
        total_ss += ss
        data['is_match'] = (ss > 0)
        
        if ss > 0:
            penalties = 0.0
            alpha = ss / len(gold_actions)
            mu = 0.05
            for j, (m, g) in enumerate(zip(model_actions, gold_actions)):
                if m['type'] in ['click', 'rightClick', 'doubleClick']:
                    L2 = math.sqrt((m['x'] - g['x'])**2 + (m['y'] - g['y'])**2)
                    if L2 < 20: L2 = 0
                    m_penalty = alpha * (1 - mu / (mu + L2))
                    penalties += m_penalty
                elif m['type'] == 'write':
                    w_penalty = alpha * (1 - simple_bleu(g['text'], m.get('text', '')))
                    penalties += w_penalty
                elif m['type'] == 'press':
                    k_penalty = 0 if m.get('key') == g.get('key') else alpha
                    penalties += k_penalty
            
            total_as += max(ss - penalties, 0)
        count += 1

    if count == 0: return None
    return {
        'count': count,
        'ss_mean': (total_ss / count),
        'as_eq6': (total_as / total_ss * 100) if total_ss > 0 else 0,
        'match_rate': (sum(1 for task_id, data in results.items() if data.get('is_match', False)) / count * 100) if count > 0 else 0,
        'skip_missing': skip_missing,
        'skip_empty_gold': skip_empty_gold
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.json or results directory")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    # Find all result files
    files_to_process = []
    if os.path.isdir(args.results):
        for root, _, files in os.walk(args.results):
            for f in files:
                if f.endswith("results.json"):
                    files_to_process.append(os.path.join(root, f))
    else:
        files_to_process.append(args.results)

    if not files_to_process:
        print(f"No results.json files found in {args.results}")
        return

    print(f"\nOmniACT Phase 1 Baseline Report")
    print(f"{'Model/File':<40} | {'SS (Avg)':<10} | {'AS (%)':<10} | {'Match (%)':<10} | {'Evaluated':<10}")
    print("-" * 95)

    for f_path in sorted(files_to_process):
        metrics = process_result_file(f_path, args.data_dir)
        if metrics:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(f_path)))
            if model_name == "phase1": # fallback
                model_name = os.path.basename(f_path)
            print(f"{model_name:<40} | {metrics['ss_mean']:<10.4f} | {metrics['as_eq6']:<10.2f} | {metrics['match_rate']:<10.2f} | {metrics['count']:<10}")
        else:
            print(f"Error processing {f_path}")

    print("-" * 110)
    print(f"Total Models Evaluated: {len(files_to_process)}")

if __name__ == "__main__":
    main()
