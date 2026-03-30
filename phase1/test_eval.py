import sys
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText

model_name = "Qwen/Qwen3-VL-2B-Instruct"
try:
    print("1. Loading Processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("2. Loading Config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("3. Loading Model...")
    model = AutoModelForImageTextToText.from_pretrained(model_name, config=config, trust_remote_code=True)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
