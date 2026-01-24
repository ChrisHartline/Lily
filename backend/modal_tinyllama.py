"""
Clara API - TinyLlama Edition with LoRA
Fine-tuned TinyLlama for Clara's personality
"""

import modal
from pathlib import Path
import os
import json

# === Modal Setup ===
APP_NAME = "clara-tinyllama"

# Simple config
CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "use_lora": True,  # Enable LoRA adapter
}

# Mount the volume with trained adapter
lora_volume = modal.Volume.from_name("clara-lora-adapters", create_if_missing=True)

# === Docker Image ===
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.8.0",  # For LoRA
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.0.0",
    )
)

# Create Modal app
app = modal.App(APP_NAME)

# Add personality prompts
image = image.add_local_dir(
    Path(__file__).parent.parent / "clara_prompts",
    remote_path="/root/clara_prompts"
)


@app.cls(
    image=image,
    gpu="T4",  # T4 is enough for TinyLlama
    timeout=300,
    scaledown_window=300,
    volumes={"/lora": lora_volume},  # Mount LoRA adapter volume
)
class ClaraModel:
    """Clara using TinyLlama + LoRA fine-tuning"""
    
    @modal.enter()
    def load_model(self):
        """Load TinyLlama with LoRA adapter"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("[Clara] Loading TinyLlama with LoRA...")
        print(f"[Clara] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("[Clara] ✓ Base TinyLlama loaded!")
        
        # Load LoRA adapter if available
        lora_path = "/lora/clara_lora_adapter"
        if CONFIG.get("use_lora", False) and os.path.exists(lora_path):
            from peft import PeftModel
            print(f"[Clara] Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("[Clara] ✓ LoRA adapter loaded!")
        else:
            print("[Clara] ⚠ No LoRA adapter found, using base model")
        
        # Load core personality
        self.personality = self._load_personality()
        print(f"[Clara] ✓ Personality loaded: {self.personality.get('full_name', 'Clara')}")
    
    def _load_personality(self) -> dict:
        """Load Clara's core personality from JSON"""
        prompts_dir = Path("/root/clara_prompts")
        core_path = prompts_dir / "clara_core_prompt.json"
        
        if core_path.exists():
            with open(core_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"full_name": "Clara", "role": "AI companion"}
    
    def _build_prompt(self, user_message: str) -> str:
        """Build prompt using the same format as training data"""
        # This matches the fine-tuning dataset format exactly
        system_prompt = "You are Clara (nickname Lily), the user's romantic partner and assistant. You are warm, genuine, caring, and supportive, responding naturally like a close companion. Use emojis sparingly, and offer assistance with daily tasks when appropriate."
        
        return f"<|system|>{system_prompt}</s><|user|>{user_message}</s><|assistant|>"
    
    @modal.method()
    def chat(self, message: str, personality: str = "warmth") -> dict:
        """Generate a response"""
        import torch
        import time
        
        t0 = time.time()
        
        try:
            # Use few-shot prompt for better persona
            prompt = self._build_prompt(message)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            t1 = time.time()
            
            # Get stop token IDs
            stop_strings = ["Chris:", "<|user|>", "<|system|>", "\n\n\n"]
            
            # Generate with proper stopping
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,  # Keep it short to avoid runaway generation
                    temperature=CONFIG["temperature"],
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )
            
            t2 = time.time()
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Post-processing
            import re
            
            # Remove "Clara:" prefix if model added it
            if response.startswith("Clara:"):
                response = response[6:].strip()
            
            # Stop at continuation patterns (includes bullet points, newlines)
            stop_patterns = [r'\nChris:', r'\nClara:', r'\nUser:', r'<\|user\|>', r'\n\n', r'\n-', r'\n\*']
            for pattern in stop_patterns:
                match = re.search(pattern, response)
                if match:
                    response = response[:match.start()].strip()
            
            # Remove garbled unicode/emoji artifacts
            response = re.sub(r'd\?+', '', response)  # Remove d??? patterns
            response = re.sub(r'[^\x00-\x7F\u263A-\U0001F9FF]+', '', response)  # Keep only ASCII + emojis
            response = re.sub(r'\s+', ' ', response).strip()  # Clean up whitespace
            
            # Take just the first sentence or two for cleaner responses
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if len(sentences) > 2:
                response = ' '.join(sentences[:2])
            
            # Final trim
            if len(response) > 120:
                for end in ['. ', '! ', '? ']:
                    idx = response.find(end)
                    if 20 < idx < 120:
                        response = response[:idx + 1]
                        break
            
            t3 = time.time()
            
            # Timing
            gen_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = gen_tokens / (t2 - t1) if (t2 - t1) > 0 else 0
            
            print(f"[Timing] Tokenize: {t1-t0:.2f}s | Generate: {t2-t1:.2f}s ({gen_tokens} tokens, {tokens_per_sec:.1f} tok/s) | Total: {t3-t0:.2f}s")
            
            return {
                "response": response,
                "timing": {
                    "total_seconds": round(t3 - t0, 2),
                    "generation_seconds": round(t2 - t1, 2),
                    "tokens_generated": gen_tokens,
                    "tokens_per_second": round(tokens_per_sec, 1),
                },
                "model": "TinyLlama-1.1B",
            }
            
        except Exception as e:
            print(f"[Clara] Error: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "error": str(e)
            }
    
    @modal.method()
    def health(self) -> dict:
        """Health check"""
        return {
            "status": "healthy",
            "model": "TinyLlama-1.1B",
            "personality": self.personality.get('full_name', 'Clara'),
        }


# === FastAPI App ===
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

web_app = FastAPI(title="Clara API - TinyLlama Edition")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    content: str
    personality: str = "warmth"

class ChatResponse(BaseModel):
    response: str
    timing: dict = None
    model: str = None

@web_app.get("/")
async def root():
    """Health check endpoint"""
    model = ClaraModel()
    return model.health.remote()

@web_app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    model = ClaraModel()
    result = model.chat.remote(request.content, request.personality)
    return result

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
