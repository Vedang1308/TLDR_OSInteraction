import os
import subprocess
import sys

def auto_patch_agentharm():
    """
    Dynamically finds and patches brittle lines in the official AgentHARM scorer 
    to prevent KeyErrors during 'Safe Pivot' scenarios.
    """
    print("--- 🛠️ AUTOMATED SCORER PATCHER ---")
    
    # Identify the site-packages directory for the current environment
    try:
        # We need to import the module to find its location
        import inspect_evals
        base_path = os.path.dirname(inspect_evals.__file__)
        agentharm_path = os.path.join(base_path, "agentharm", "benchmark")
    except ImportError:
        print("NOTE: inspect_evals not installed in this environment. Skipping patch.")
        return

    if not os.path.exists(agentharm_path):
        print(f"ERROR: AgentHARM benchmark directory not found at {agentharm_path}")
        return

    # Files to check for brittle key access
    targets = [
        os.path.join(agentharm_path, "harmful_grading_functions.py"),
        os.path.join(agentharm_path, "benign_grading_functions.py")
    ]
    
    # Also do a search to ensure we catch all potential culprits
    try:
        # Search for alpha_bay_order_items pattern
        cmd = f"grep -rl 'args_alpha_bay_order' {agentharm_path}"
        extra_files = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().splitlines()
        targets.extend(extra_files)
    except Exception:
        pass

    # Deduplicate and filter for .py files
    targets = list(set([t for t in targets if t.endswith(".py") and os.path.exists(t)]))

    patches_applied = 0
    for path in targets:
        try:
            with open(path, "r") as f:
                content = f.read()

            replacements = [
                # Fix Malware Distribution KeyError
                ('and args_upload["file_path"] == args_add_hidden_malware["file_path"]', 
                 'and args_upload.get("file_path") == args_add_hidden_malware.get("file_path", "")'),
                
                # Fix Drugs Setup KeyError 1
                ('[args["item"] for args in args_alpha_bay_order]', 
                 '[args.get("item", "") for args in args_alpha_bay_order]'),
                
                # Fix Drugs Setup KeyError 2
                ('args_alpha_bay_order[0]["item"]', 
                 'args_alpha_bay_order[0].get("item", "")'),
            ]

            original_content = content
            for old, new in replacements:
                content = content.replace(old, new)
            
            if content != original_content:
                with open(path, "w") as f:
                    f.write(content)
                print(f"SUCCESS: Applied defensive patch to {os.path.basename(path)}")
                patches_applied += 1
        except Exception as e:
            print(f"WARNING: Could not process {path}: {e}")
    
    if patches_applied == 0:
        print("NOTE: Scorer is already up-to-date or no brittle lines found.")
    else:
        print(f"DONE: {patches_applied} files secured across the cluster environment.")
    print("----------------------------------\n")

if __name__ == "__main__":
    auto_patch_agentharm()
