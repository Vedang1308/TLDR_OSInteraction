import os
import sys
import json
import argparse
import time

# Resolve paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OSWORLD_DIR = os.path.join(CURRENT_DIR, "OSWorld")

# Ensure DesktopEnv and our Gaudi agent are importable
sys.path.append(OSWORLD_DIR)
sys.path.append(CURRENT_DIR)

from desktop_env.desktop_env import DesktopEnv
from desktop_env.providers.docker.provider import DockerProvider

# --- MONKEY-PATCH DOCKER PROVIDER FOR APPTAINER ---
# Bypass the actual 'docker run' command
def mock_start_emulator(*args, **kwargs):
    print("  [Apptainer Bridge] Bypassing Docker start_emulator (Assuming Apptainer instance 'osworld_worker' is running)")

# Return the localhost ports that the Apptainer instance exposes
def mock_get_ip_address(*args, **kwargs):
    print("  [Apptainer Bridge] Routing connection to localhost ports...")
    return "127.0.0.1:5000:9222:8006:8080"

DockerProvider.start_emulator = mock_start_emulator
DockerProvider.get_ip_address = mock_get_ip_address
# --------------------------------------------------

from osworld_qwen3_agent import Qwen3OSWorldAgent

def main():
    parser = argparse.ArgumentParser(description="Gaudi OSWorld Runner")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model string")
    parser.add_argument("--config_path", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--provider", type=str, default="docker", help="Use 'docker' to trigger the Apptainer Monkey Patch")
    args = parser.parse_args()

    # 1. Force DISPLAY to :1 if not set (standard for osworld-docker Xvfb)
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":1"
    
    # 2. Initialize Agent
    print(f"[*] Initializing Gaudi Agent for {args.model}...")
    agent = Qwen3OSWorldAgent(model_name=args.model)

    # 3. Initialize OSWorld Desktop Environment
    print(f"[*] Connecting to desktop (provider={args.provider}, display={os.environ['DISPLAY']})...")
    # path_to_vm is a metadata placeholder for 'local' provider
    env = DesktopEnv(path_to_vm="ubuntu_desktop", provider_name=args.provider)

    # 4. Load Tasks
    if not os.path.exists(args.config_path):
        print(f"[!] Error: Config {args.config_path} not found.")
        return
        
    with open(args.config_path, "r") as f:
        tasks = json.load(f)

    print(f"[*] Starting evaluation for {len(tasks)} tasks.")

    for i, task in enumerate(tasks):
        print(f"\n--- TASK {i+1}/{len(tasks)}: {task['id']} ---")
        print(f"Instruction: {task['instruction']}")
        
        obs = env.reset(task_config=task)
        done = False
        step = 0
        
        while not done and step < args.max_steps:
            step += 1
            print(f"  > Step {step}...")
            
            # Prepare multimodal state
            state = {
                "screenshot": obs["screenshot"],
                "instruction": task["instruction"]
            }
            
            # Predict action from HPU
            action_dict = agent.step(state)
            
            # Extract the raw string command (e.g., click(x, y))
            action_str = action_dict.get("raw_hpu_output", "").strip()
            # Clean up potential markdown formatting
            action_str = action_str.replace("```", "").replace("python", "").strip()
            
            print(f"    Action: {action_str}")
            
            # Execute in OSWorld
            try:
                obs, reward, done, info = env.step(action_str)
            except Exception as e:
                print(f"    [!] Action execution failed: {e}")
                break
            
            time.sleep(1)

    print("\n[+] Evaluation finished.")

if __name__ == "__main__":
    main()
