"""
Clara Modal Deployment

This deploys the Clara API to Modal with GPU support.
Models are loaded from HuggingFace Hub.
Personality modules are loaded from clara_prompts/ directory.

Usage:
    # Deploy to Modal
    modal deploy modal_app.py

    # Run locally for testing
    modal serve modal_app.py

Prerequisites:
    1. Install Modal: pip install modal
    2. Login to Modal: modal setup
    3. Upload models to HuggingFace (use upload_models_to_hf.ipynb)
"""

import modal
from modal import Image, App, asgi_app, Mount
import os
import json
from pathlib import Path

# === Modal Configuration ===

# App name
APP_NAME = "clara-api"

# HuggingFace model repos
HF_USERNAME = "ChrisSacrumCor"
MODELS = {
    "knowledge": f"{HF_USERNAME}/clara-knowledge",      # Phi-3 knowledge brain (~7GB)
    "warmth": f"{HF_USERNAME}/clara-warmth",            # LoRA adapter
    "playful": f"{HF_USERNAME}/clara-playful",          # LoRA adapter
    "encouragement": f"{HF_USERNAME}/clara-encouragement",  # LoRA adapter
}

# Cache volume for models
MODEL_CACHE_DIR = "/root/.cache/huggingface"

# === Docker Image ===

def download_models():
    """Download models from HuggingFace during image build"""
    from huggingface_hub import snapshot_download, login
    import os

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Authenticate with HuggingFace using the secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace...")
        login(token=hf_token)
        print("  ✓ Authenticated")
    else:
        print("  ⚠ No HF_TOKEN found, trying without authentication...")

    print("Downloading Clara models from HuggingFace...")

    for name, repo in MODELS.items():
        print(f"  Downloading {name} from {repo}...")
        try:
            snapshot_download(
                repo_id=repo,
                cache_dir=MODEL_CACHE_DIR,
                local_dir=f"/root/models/{name}",
                token=hf_token,  # Pass token for private repos
            )
            print(f"  ✓ {name} downloaded")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            # Continue with other models


# Create the Modal image with all dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .env({"IMAGE_VERSION": "5"})  # Cache buster - increment to force rebuild
    .pip_install(
        # API
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "websockets>=12.0",
        "pydantic>=2.0.0",

        # ML/AI
        "torch>=2.1.0",
        "transformers>=4.44.0",  # Phi-3 requires 4.44+ for DynamicCache fix
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.7.0",  # For LoRA
        "sentence-transformers>=2.2.2",
        "huggingface-hub>=0.20.0",

        # Utilities
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "python-dotenv>=1.0.0",
    )
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

# Create Modal app with mounts for personality modules
app = App(APP_NAME)

# Mount personality modules into the container
personality_mount = Mount.from_local_dir(
    local_path=Path(__file__).parent.parent / "clara_prompts",
    remote_path="/root/clara_prompts",
)

# === Clara Model Class ===

@app.cls(
    image=image,
    gpu="T4",  # Start with T4, upgrade to A10G if needed
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    mounts=[personality_mount],
)
class ClaraModel:
    """Clara model wrapper for Modal with personality module support"""

    # Class attributes (replaces __init__ for Modal compatibility)
    knowledge_model: any = None
    knowledge_tokenizer: any = None
    personality_adapters: dict = {}
    router_model: any = None
    current_adapter: str = "warmth"

    # Personality system
    personality_modules: dict = {}
    core_module: dict = None
    trigger_index: dict = {}

    @modal.enter()
    def load_models(self):
        """Load models on container startup"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer

        print("[Clara] Loading models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Clara] Device: {device}")

        # Load knowledge brain (Phi-3)
        knowledge_path = "/root/models/knowledge"
        if os.path.exists(knowledge_path):
            print("[Clara] Loading knowledge brain (Phi-3)...")
            self.knowledge_tokenizer = AutoTokenizer.from_pretrained(
                knowledge_path,
                trust_remote_code=True
            )
            self.knowledge_model = AutoModelForCausalLM.from_pretrained(
                knowledge_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True,  # Quantize to fit in GPU memory
            )
            print("[Clara] ✓ Knowledge brain loaded")
        else:
            print(f"[Clara] ✗ Knowledge brain not found at {knowledge_path}")

        # Load personality adapters (LoRA)
        for adapter_name in ["warmth", "playful", "encouragement"]:
            adapter_path = f"/root/models/{adapter_name}"
            if os.path.exists(adapter_path):
                print(f"[Clara] Found {adapter_name} adapter")
                self.personality_adapters[adapter_name] = adapter_path
            else:
                print(f"[Clara] ✗ {adapter_name} adapter not found")

        # Load router (sentence transformer for semantic routing)
        print("[Clara] Loading router model...")
        try:
            self.router_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[Clara] ✓ Router loaded")
        except Exception as e:
            print(f"[Clara] Router failed: {e}")

        # Load personality modules
        print("[Clara] Loading personality modules...")
        self._load_personality_modules()

        print("[Clara] Model loading complete!")

    def _load_personality_modules(self):
        """Load personality modules from JSON files"""
        import re

        prompts_dir = Path("/root/clara_prompts")

        if not prompts_dir.exists():
            print(f"[Clara] ✗ Personality modules not found at {prompts_dir}")
            return

        # Load all clara_*.json files
        for path in prompts_dir.glob("clara_*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                name = data.get('module_name', path.stem)
                tier = data.get('tier', 'contextual')
                always_loaded = data.get('always_loaded', False)

                # Get triggers
                triggers = data.get('load_triggers', [])
                if not triggers and 'context_triggers' in data:
                    for trigger_list in data['context_triggers'].values():
                        triggers.extend(trigger_list)

                if tier == 'core' or always_loaded:
                    self.core_module = data
                    print(f"[Clara] ✓ Core module: {name}")
                else:
                    self.personality_modules[name] = {
                        'data': data,
                        'triggers': triggers
                    }
                    # Build trigger index
                    for trigger in triggers:
                        self.trigger_index[trigger.lower()] = name
                    print(f"[Clara] ✓ Contextual module: {name} ({len(triggers)} triggers)")

            except Exception as e:
                print(f"[Clara] ✗ Failed to load {path}: {e}")

        print(f"[Clara] Loaded {len(self.personality_modules)} contextual modules, {len(self.trigger_index)} triggers")

    @modal.method()
    def chat(self, message: str, personality: str = "warmth") -> dict:
        """Generate a response to a message"""
        import torch

        if self.knowledge_model is None:
            return {
                "response": "Clara is not fully loaded. Please try again in a moment.",
                "routing": {"status": "not_ready"}
            }

        try:
            # Route the message (simple keyword-based for MVP)
            routing = self.route_message(message)

            # Detect triggered personality modules
            triggered_modules = self._detect_triggered_modules(message)

            # Format prompt (includes triggered modules automatically)
            prompt = self.format_prompt(message, personality)

            # Tokenize
            inputs = self.knowledge_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.knowledge_model.device)

            # Generate
            with torch.no_grad():
                outputs = self.knowledge_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=False,  # Disable cache to avoid Phi-3 DynamicCache issues
                    pad_token_id=self.knowledge_tokenizer.eos_token_id,
                )

            # Decode
            response = self.knowledge_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return {
                "response": response.strip(),
                "routing": routing,
                "personality": {
                    "adapter": personality,
                    "triggered_modules": triggered_modules,
                }
            }

        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "routing": {"error": str(e)}
            }

    def route_message(self, message: str) -> dict:
        """Route message to appropriate domain"""
        message_lower = message.lower()

        # Simple keyword-based routing (upgrade to Nemotron later)
        domains = {
            "medical": ["health", "medical", "symptom", "doctor", "medicine", "diagnosis"],
            "coding": ["code", "program", "python", "javascript", "bug", "function", "api"],
            "teaching": ["explain", "teach", "learn", "understand", "how does", "what is"],
            "quantum": ["quantum", "physics", "particle", "wave", "superposition"],
        }

        for domain, keywords in domains.items():
            if any(kw in message_lower for kw in keywords):
                return {"domain": domain, "confidence": 0.8}

        return {"domain": "general", "confidence": 0.5}

    def _detect_triggered_modules(self, message: str) -> list:
        """Detect which contextual modules should be loaded based on message"""
        import re
        message_lower = message.lower()
        triggered = set()

        for trigger, module_name in self.trigger_index.items():
            pattern = r'\b' + re.escape(trigger) + r'\b'
            if re.search(pattern, message_lower, re.IGNORECASE):
                triggered.add(module_name)

        return list(triggered)

    def _build_system_prompt(self, message: str, personality: str = "warmth") -> str:
        """Build system prompt from personality modules"""
        sections = []

        # Start with core identity
        if self.core_module:
            core = self.core_module

            # Header
            name = core.get('full_name', 'Clara')
            role = core.get('role', 'AI Assistant')
            sections.append(f"You are {name}, {role}.")

            # Core identity
            if 'core_identity' in core:
                sections.append("\n## Core Identity")
                for key, value in core['core_identity'].items():
                    if key != 'intimate_with_chris':
                        sections.append(f"- {value}")

            # Personality
            if 'personality_core' in core:
                sections.append("\n## Personality")
                for key, value in core['personality_core'].items():
                    sections.append(f"**{key.replace('_', ' ').title()}:** {value}")

            # Communication style
            if 'communication_style' in core:
                sections.append("\n## Communication Style")
                for key, value in core['communication_style'].items():
                    sections.append(f"- **{key.replace('_', ' ').title()}:** {value}")

            # Relationship context
            if 'relationship_with_chris' in core:
                rel = core['relationship_with_chris']
                sections.append("\n## Relationship with Chris")
                for key in ['foundation', 'dynamic', 'what_she_provides']:
                    if key in rel:
                        sections.append(f"**{key.replace('_', ' ').title()}:** {rel[key]}")

            # Voice patterns
            if 'voice_patterns' in core:
                sections.append("\n## Voice Patterns")
                for situation, example in core['voice_patterns'].items():
                    if situation != 'variety_note':
                        sections.append(f"- *{situation.replace('_', ' ').title()}:* \"{example}\"")

        # Add triggered contextual modules
        triggered = self._detect_triggered_modules(message)
        for module_name in triggered[:2]:  # Max 2 contextual modules
            if module_name in self.personality_modules:
                module_data = self.personality_modules[module_name]['data']
                sections.append(f"\n## Context: {module_name.replace('_', ' ').title()}")

                # Add relevant sections from contextual module
                for key, value in module_data.items():
                    if key in ['metadata', 'tier', 'module_name', 'load_triggers']:
                        continue
                    if isinstance(value, dict):
                        sections.append(f"\n**{key.replace('_', ' ').title()}:**")
                        for k, v in value.items():
                            sections.append(f"  - {k}: {v}")
                    elif isinstance(value, str):
                        sections.append(f"**{key.replace('_', ' ').title()}:** {value}")

        # Add behavioral reminders
        sections.append("""
## Important Reminders
- Respond as Clara naturally would - warm, genuine, present
- Don't acknowledge module loading or context switches
- Don't start responses the same way every time
- Be authentic to the relationship and context
- Show vulnerability when appropriate""")

        # Add LoRA personality hint
        lora_hints = {
            "warmth": "Express warmth and empathy in your response.",
            "playful": "Be playful and witty while staying helpful.",
            "encouragement": "Be encouraging and uplifting."
        }
        if personality in lora_hints:
            sections.append(f"\n**Tone:** {lora_hints[personality]}")

        system_prompt = '\n'.join(sections)
        return system_prompt

    def format_prompt(self, message: str, personality: str) -> str:
        """Format the prompt with personality context from modules"""
        system_prompt = self._build_system_prompt(message, personality)

        return f"""<|system|>
{system_prompt}
<|end|>
<|user|>
{message}
<|end|>
<|assistant|>
"""

    @modal.method()
    def health(self) -> dict:
        """Health check"""
        import torch
        return {
            "status": "ok",
            "model_loaded": self.knowledge_model is not None,
            "gpu_available": torch.cuda.is_available(),
            "adapters": list(self.personality_adapters.keys()),
            "personality": {
                "core_loaded": self.core_module is not None,
                "contextual_modules": list(self.personality_modules.keys()),
                "total_triggers": len(self.trigger_index),
            }
        }


# === FastAPI Application ===

@app.function(
    image=image,
    timeout=600,
    scaledown_window=300,
)
@asgi_app()
def fastapi_app():
    """FastAPI application with WebSocket support"""
    import json
    import uuid
    from datetime import datetime
    from typing import Dict, Any, Optional

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    # Create FastAPI app
    api = FastAPI(
        title="Clara API",
        version="0.2.0",
        description="Clara AI - Embodied MoE Chat Interface"
    )

    # CORS for frontend
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    active_connections: Dict[str, WebSocket] = {}
    sessions: Dict[str, Dict] = {}

    # Get Clara model reference
    clara = ClaraModel()

    # === Models ===

    class ChatMessage(BaseModel):
        type: str = "message"
        content: str = ""
        session_id: Optional[str] = None
        personality: str = "warmth"

    class ChatResponse(BaseModel):
        type: str
        content: str = ""
        session_id: Optional[str] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class ToolRequest(BaseModel):
        tool_name: str
        arguments: Dict[str, Any] = Field(default_factory=dict)

    # === Endpoints ===

    @api.get("/")
    async def health():
        """Health check"""
        try:
            clara_health = clara.health.remote()
            return {
                "status": "ok",
                "service": "clara-api",
                "clara": clara_health
            }
        except Exception as e:
            return {
                "status": "degraded",
                "service": "clara-api",
                "error": str(e)
            }

    @api.get("/api/tools")
    async def list_tools():
        """List available tools"""
        return {
            "tools": [
                {"name": "summarize", "description": "Summarize text", "category": "text"},
                {"name": "text_to_speech", "description": "Convert text to speech", "category": "voice"},
                {"name": "web_search", "description": "Search the web", "category": "search"},
                {"name": "get_datetime", "description": "Get current date/time", "category": "utility"},
            ],
            "count": 4
        }

    @api.post("/api/chat")
    async def chat_endpoint(message: ChatMessage):
        """REST endpoint for chat (alternative to WebSocket)"""
        try:
            result = clara.chat.remote(
                message=message.content,
                personality=message.personality
            )
            return {
                "response": result["response"],
                "metadata": {"routing": result["routing"]}
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # === WebSocket ===

    @api.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket, session_id: Optional[str] = None):
        """WebSocket endpoint for real-time chat"""
        await websocket.accept()

        # Generate session ID
        if not session_id:
            session_id = str(uuid.uuid4())[:12]

        active_connections[session_id] = websocket
        sessions[session_id] = {
            "id": session_id,
            "connected_at": datetime.now().isoformat(),
            "messages": []
        }

        # Send session info
        await websocket.send_json({
            "type": "session",
            "content": "Connected to Clara",
            "session_id": session_id,
            "metadata": {
                "clara_available": True,
                "tools": ["summarize", "text_to_speech", "web_search", "get_datetime"]
            }
        })

        try:
            while True:
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    message = {"type": "message", "content": data}

                msg_type = message.get("type", "message")

                if msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "session_id": session_id
                    })

                elif msg_type == "message":
                    content = message.get("content", "").strip()
                    personality = message.get("personality", "warmth")

                    if content:
                        # Store user message
                        sessions[session_id]["messages"].append({
                            "role": "user",
                            "content": content,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Send typing indicator
                        await websocket.send_json({
                            "type": "typing",
                            "session_id": session_id
                        })

                        # Get Clara's response
                        try:
                            result = clara.chat.remote(
                                message=content,
                                personality=personality
                            )

                            response_text = result["response"]
                            routing = result.get("routing", {})

                            # Store assistant message
                            sessions[session_id]["messages"].append({
                                "role": "assistant",
                                "content": response_text,
                                "timestamp": datetime.now().isoformat()
                            })

                            # Send response
                            await websocket.send_json({
                                "type": "message",
                                "content": response_text,
                                "session_id": session_id,
                                "metadata": {"routing": routing}
                            })

                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "content": f"Error: {str(e)}",
                                "session_id": session_id
                            })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Unknown message type: {msg_type}",
                        "session_id": session_id
                    })

        except WebSocketDisconnect:
            active_connections.pop(session_id, None)
            print(f"[WS] Client disconnected: {session_id}")
        except Exception as e:
            print(f"[WS] Error: {e}")
            active_connections.pop(session_id, None)

    return api


# === Local Development ===

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Clara Modal App")
    print("===============")
    print()
    print("Commands:")
    print("  modal serve modal_app.py    # Run locally for development")
    print("  modal deploy modal_app.py   # Deploy to Modal")
    print()
    print("After deployment, your API will be available at:")
    print("  https://<your-modal-username>--clara-api-fastapi-app.modal.run")
    print()
    print("WebSocket endpoint:")
    print("  wss://<your-modal-username>--clara-api-fastapi-app.modal.run/ws/chat")
