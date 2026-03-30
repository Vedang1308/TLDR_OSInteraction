import sys
from transformers import AutoConfig

model_name = "Qwen/Qwen3-VL-2B-Instruct"
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
