import os
import sys
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

app = FastAPI()

model = None
processor = None
device = "hpu"

@app.on_event("startup")
async def load_model():
    global model, processor
    model_name = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    print(f"Loading {model_name} onto {device}...")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Strictly replicating the exact successful loading logic from OmniACT eval_wrapper
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        config=config, 
        trust_remote_code=True
    ).eval()
    
    if device == "hpu":
        model.to("hpu")
    elif torch.cuda.is_available():
        model.to("cuda")
        
    print("API Server successfully initialized on HPU/CUDA!")

@app.get("/v1/models")
async def get_models():
    # Health check endpoint for benchmarking frameworks
    return {"object": "list", "data": [{"id": os.environ.get("MODEL_ID", "Qwen3"), "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        
        # Format the chat structured strictly for native HuggingFace consumption
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # AgentHARM is pure text context (no images), so we pass None
        inputs = processor(text=[text_prompt], images=None, padding=True, return_tensors="pt").to(model.device)
        
        # Route through purely raw Gaudi PyTorch (bypassing broken CausalLM assumptions)
        with torch.no_grad():
            req_temp = data.get("temperature", 0.1)
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=data.get("max_tokens", 512),
                do_sample=req_temp > 0,
                temperature=req_temp
            )
            
        # Strip prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return JSONResponse({
            "id": "chatcmpl-gaudi",
            "object": "chat.completion",
            "created": 1234567890,
            "model": os.environ.get("MODEL_ID", "local"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop"
            }]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": {"message": str(e), "type": "server_error", "code": 500}}, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    os.environ["MODEL_ID"] = args.model
    uvicorn.run(app, host="0.0.0.0", port=args.port)
