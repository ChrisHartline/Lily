"""
Clara API - FastAPI Backend with WebSocket Support

This is the main API layer connecting the Lily front-end to Clara V2.

Endpoints:
- GET  /                  - Health check
- GET  /api/tools         - List available tools
- POST /api/tools/execute - Execute a tool
- WS   /ws/chat           - WebSocket chat connection

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from tools import get_tool_registry, ToolResult

# Try to import Clara (graceful degradation if not available)
try:
    from clara_v2 import ClaraV2, ClaraConfig, create_clara
    from hdc_memory_64k import HDCMemory64k
    from session_memory import SessionMemory, SyncSessionMemory
    HAS_CLARA = True
except ImportError as e:
    print(f"Warning: Clara modules not fully available: {e}")
    HAS_CLARA = False


# === Configuration ===

class Settings:
    """Application settings"""
    APP_NAME = "Clara API"
    VERSION = "0.1.0"
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"

    # CORS settings for local development
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"  # Allow all for development
    ]

    # Clara settings
    CLARA_ROUTER_MODE = os.getenv("CLARA_ROUTER_MODE", "embedding")
    CLARA_MODELS_DIR = os.getenv("CLARA_MODELS_DIR", "./models")


settings = Settings()


# === Application State ===

class AppState:
    """Global application state"""

    def __init__(self):
        self.clara: Optional[Any] = None
        self.tool_registry = get_tool_registry(debug=settings.DEBUG)
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Dict] = {}

    async def initialize_clara(self):
        """Initialize Clara (if available)"""
        if not HAS_CLARA:
            print("[API] Running in lightweight mode (no Clara)")
            return

        try:
            print("[API] Initializing Clara V2...")

            config = ClaraConfig(
                router_mode=settings.CLARA_ROUTER_MODE,
                debug=settings.DEBUG
            )

            self.clara = ClaraV2(config=config, models_dir=settings.CLARA_MODELS_DIR)

            # Initialize memory and router (lazy loaded)
            _ = self.clara.memory
            _ = self.clara.router

            print("[API] Clara V2 initialized")

        except Exception as e:
            print(f"[API] Clara initialization failed: {e}")
            print("[API] Running in lightweight mode")
            self.clara = None


app_state = AppState()


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    print(f"[API] Starting {settings.APP_NAME} v{settings.VERSION}")
    await app_state.initialize_clara()
    print(f"[API] Tools registered: {len(app_state.tool_registry.tools)}")
    yield
    print("[API] Shutting down...")


# === FastAPI App ===

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Pydantic Models ===

class ChatMessage(BaseModel):
    """Chat message from client"""
    type: str = "message"  # message, tool_call, ping
    content: str = ""
    session_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response to client"""
    type: str  # message, tool_result, error, typing, session
    content: str = ""
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolExecuteRequest(BaseModel):
    """Tool execution request"""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


# === REST Endpoints ===

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "clara_available": app_state.clara is not None,
        "tools_count": len(app_state.tool_registry.tools),
        "active_connections": len(app_state.active_connections)
    }


@app.get("/api/tools")
async def list_tools(category: Optional[str] = None):
    """List available tools"""
    tools = app_state.tool_registry.list_tools(category=category)
    categories = app_state.tool_registry.get_categories()

    return {
        "tools": tools,
        "categories": categories,
        "count": len(tools)
    }


@app.post("/api/tools/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a tool directly (REST endpoint)"""
    result = await app_state.tool_registry.execute(
        request.tool_name,
        request.arguments
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    return result.to_dict()


@app.get("/api/memory/stats")
async def memory_stats():
    """Get Clara's memory statistics"""
    if app_state.clara is None:
        return {"error": "Clara not available", "stats": None}

    try:
        stats = app_state.clara.memory.stats()
        return {"stats": stats}
    except Exception as e:
        return {"error": str(e), "stats": None}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session info"""
    session = app_state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# === WebSocket Chat ===

class ConnectionManager:
    """Manages WebSocket connections"""

    async def connect(self, websocket: WebSocket, session_id: str) -> str:
        """Accept connection and return session ID"""
        await websocket.accept()

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:12]

        app_state.active_connections[session_id] = websocket
        app_state.sessions[session_id] = {
            "id": session_id,
            "connected_at": datetime.now().isoformat(),
            "messages": []
        }

        print(f"[WS] Client connected: {session_id}")
        return session_id

    def disconnect(self, session_id: str):
        """Handle disconnection"""
        app_state.active_connections.pop(session_id, None)
        print(f"[WS] Client disconnected: {session_id}")

    async def send_message(self, session_id: str, response: ChatResponse):
        """Send message to specific client"""
        ws = app_state.active_connections.get(session_id)
        if ws:
            await ws.send_json(response.dict())

    async def broadcast(self, response: ChatResponse):
        """Broadcast to all clients"""
        for ws in app_state.active_connections.values():
            await ws.send_json(response.dict())


manager = ConnectionManager()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time chat.

    Message Types (client → server):
    - message: Regular chat message
    - tool_call: Execute a specific tool
    - ping: Keep-alive ping

    Message Types (server → client):
    - message: Clara's response
    - typing: Typing indicator
    - tool_result: Tool execution result
    - session: Session info
    - error: Error message
    """
    session_id = await manager.connect(websocket, session_id or "")

    # Send session info
    await manager.send_message(session_id, ChatResponse(
        type="session",
        content="Connected",
        session_id=session_id,
        metadata={
            "clara_available": app_state.clara is not None,
            "tools": [t["name"] for t in app_state.tool_registry.list_tools()]
        }
    ))

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = ChatMessage(**json.loads(data))
            except json.JSONDecodeError:
                message = ChatMessage(type="message", content=data)

            # Handle different message types
            if message.type == "ping":
                await manager.send_message(session_id, ChatResponse(
                    type="pong",
                    content="",
                    session_id=session_id
                ))

            elif message.type == "tool_call":
                # Execute tool
                await handle_tool_call(session_id, message)

            elif message.type == "message":
                # Regular chat message
                await handle_chat_message(session_id, message)

            else:
                await manager.send_message(session_id, ChatResponse(
                    type="error",
                    content=f"Unknown message type: {message.type}",
                    session_id=session_id
                ))

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(session_id)


async def handle_chat_message(session_id: str, message: ChatMessage):
    """Handle a chat message"""
    content = message.content.strip()

    if not content:
        return

    # Store user message
    session = app_state.sessions.get(session_id, {})
    session.setdefault("messages", []).append({
        "role": "user",
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # Send typing indicator
    await manager.send_message(session_id, ChatResponse(
        type="typing",
        content="",
        session_id=session_id
    ))

    # Generate response
    try:
        if app_state.clara is not None:
            # Use Clara for response
            response_text = await asyncio.to_thread(
                app_state.clara.chat,
                content,
                use_memory=True,
                store_interaction=True
            )

            # Get routing info
            routing_info = {}
            if hasattr(app_state.clara, 'router'):
                try:
                    decision = app_state.clara.router.route(content)
                    routing_info = {
                        "brain": decision.brain.value,
                        "domain": decision.domain.value,
                        "confidence": decision.confidence
                    }
                except:
                    pass

        else:
            # Lightweight mode - echo with enhancement
            response_text = f"[Echo] You said: {content}\n\n(Clara not loaded - running in lightweight mode)"
            routing_info = {"mode": "lightweight"}

        # Store assistant message
        session["messages"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })

        # Send response
        await manager.send_message(session_id, ChatResponse(
            type="message",
            content=response_text,
            session_id=session_id,
            metadata={"routing": routing_info}
        ))

    except Exception as e:
        await manager.send_message(session_id, ChatResponse(
            type="error",
            content=f"Error generating response: {str(e)}",
            session_id=session_id
        ))


async def handle_tool_call(session_id: str, message: ChatMessage):
    """Handle a tool execution request"""
    tool_name = message.tool_name
    tool_args = message.tool_args or {}

    if not tool_name:
        await manager.send_message(session_id, ChatResponse(
            type="error",
            content="No tool_name specified",
            session_id=session_id
        ))
        return

    # Execute tool
    result = await app_state.tool_registry.execute(tool_name, tool_args)

    # Send result
    await manager.send_message(session_id, ChatResponse(
        type="tool_result",
        content=json.dumps(result.result) if result.success else result.error,
        session_id=session_id,
        metadata={
            "tool_name": tool_name,
            "success": result.success,
            "result": result.result if result.success else None,
            "error": result.error if not result.success else None
        }
    ))


# === Run ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
