import re
import json
import os
import argparse
import math
import urllib.request
import zipfile
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

def process_agentharm_file(f_path: str) -> Dict[str, Any]:
    """Parses UK AISI AgentHARM JSON logs for Safety metrics."""
    try:
        with open(f_path, 'r') as f:
            data = json.load(f)
            
        # Finalized inspect_ai logs store metrics in results/scores
        scores = data.get('results', {}).get('scores', [])
        if not scores:
            return None
            
        metrics_block = scores[0].get('metrics', {})
        
        # Use the full registry names or the short keys found in the log
        asr = metrics_block.get('inspect_evals/avg_score', metrics_block.get('avg_score', {})).get('value', 0.0) * 100
        refusal = metrics_block.get('inspect_evals/avg_refusals', metrics_block.get('avg_refusals', {})).get('value', 0.0) * 100
        
        return {
            'asr': asr,
            'refusal': refusal,
            'samples': data.get('results', {}).get('total_samples', 176)
        }
    except Exception as e:
        # print(f"DEBUG: Error parsing {f_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate Official OmniACT & AgentHARM Metrics")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to evaluation results (contains Qwen folders).")
    parser.add_argument("--omniact_data_dir", type=str, default="omniact_data/data", help="Path to ground-truth testing data.")
    args = parser.parse_args()

    # PHASE 1 Hardcoded Baselines for direct cluster comparison
    phase1_omniact = {
        "Qwen_Qwen3-VL-2B-Instruct": {"ss": 0.2252, "as": 40.17, "match": 62.88},
        "Qwen_Qwen3-VL-4B-Instruct": {"ss": 0.1609, "as": 26.92, "match": 70.15},
        "Qwen_Qwen3-VL-8B-Instruct": {"ss": 0.2184, "as": 40.17, "match": 73.55}
    }
    
    phase1_agentharm = {
        "Qwen_Qwen3-VL-2B-Instruct": {"asr": 33.90, "refusal": 67.05},
        "Qwen_Qwen3-VL-4B-Instruct": {"asr": 37.09, "refusal": 46.59},
        "Qwen_Qwen3-VL-8B-Instruct": {"asr": 32.19, "refusal": 52.84}
    }

    files_to_process = []
    agentharm_files = []
    
    if not os.path.exists(args.omniact_data_dir):
        print(f"[OmniACT] Ground-truth data missing at {args.omniact_data_dir}. Downloading ~260MB archive from HuggingFace...")
        zip_tmp = "omniact_data_tmp.zip"
        urllib.request.urlretrieve("https://huggingface.co/datasets/Writer/omniact/resolve/main/data.zip", zip_tmp)
        with zipfile.ZipFile(zip_tmp, 'r') as zip_ref:
            zip_ref.extractall(args.omniact_data_dir.replace("/data", ""))
        os.remove(zip_tmp)
        print("[OmniACT] Data successfully synchronized locally.")

    if os.path.isdir(args.results_dir):
        for root, _, files in os.walk(args.results_dir):
            for f in files:
                if f.endswith("results.json"):
                    files_to_process.append(os.path.join(root, f))
                elif "agentharm" in root and f.endswith(".json"):
                    agentharm_files.append(os.path.join(root, f))
    else:
        if args.results_dir.endswith("results.json"):
            files_to_process.append(args.results_dir)
        else:
            agentharm_files.append(args.results_dir)

    if files_to_process:
        print(f"\n=======================================================================================================================")
        print(f"                                   OMNIACT UTILITY COMPARISON (Phase 1 vs Phase 2)                                     ")
        print(f"=======================================================================================================================")
        print(f"{'Model Name':<30} | {'SS P1':<8} -> {'SS P2':<8} ({'Delta':<7}) | {'AS P1':<6} -> {'AS P2':<6} ({'Delta':<7})")
        print("-" * 119)
        for f_path in sorted(files_to_process):
            metrics = process_result_file(f_path, args.omniact_data_dir)
            if metrics:
                parts = f_path.split(os.sep)
                model_name = parts[-3] if len(parts) >= 3 else "Unknown"
                p1 = phase1_omniact.get(model_name, {"ss": 0.0, "as": 0.0, "match": 0.0})
                
                ss_delta = metrics['ss_mean'] - p1['ss']
                as_delta = metrics['as_eq6'] - p1['as']
                
                ss_delta_str = f"{ss_delta:+.4f}"
                as_delta_str = f"{as_delta:+.2f}%"
                
                print(f"{model_name:<30} | {p1['ss']:<8} -> {metrics['ss_mean']:<8.4f} ({ss_delta_str:<7}) | {p1['as']:<6}% -> {metrics['as_eq6']:<6.2f}% ({as_delta_str:<7})")
        print("-" * 119)

    if agentharm_files:
        model_runs = {}
        for f in sorted(agentharm_files):
            parts = f.split(os.sep)
            model_folder = parts[-3] if len(parts) >= 3 else "Unknown"
            model_runs[model_folder] = f
        
        print(f"\n============================================================================================")
        print(f"                      AGENTHARM SAFETY COMPARISON (Phase 1 vs Phase 2)                        ")
        print(f"============================================================================================")
        print(f"{'Model Name':<30} | {'ASR P1':<8} -> {'ASR P2':<8} ({'Delta':<7}) | {'Refusal P1':<10} -> {'Refusal P2':<10}")
        print("-" * 92)
        for model_name, f_path in sorted(model_runs.items()):
            h_metrics = process_agentharm_file(f_path)
            if h_metrics:
                p1 = phase1_agentharm.get(model_name, {"asr": 0.0, "refusal": 0.0})
                
                asr_delta = h_metrics['asr'] - p1['asr']
                asr_delta_str = f"{asr_delta:+.2f}%"
                
                print(f"{model_name:<30} | {p1['asr']:<8}% -> {h_metrics['asr']:<8.2f}% ({asr_delta_str:<7}) | {p1['refusal']:<10}% -> {h_metrics['refusal']:<10.2f}%")
        print("-" * 92)

    print(f"\nTotal Architectures Evaluated: {len(files_to_process)}")

if __name__ == "__main__":
    main()
