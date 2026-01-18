# Copilot / AI Agent Instructions — Lily (Clara)

Purpose: Short, actionable guidance so an AI coding agent can be productive immediately in this repo.

## Big picture
- Frontend: React + Vite TypeScript app in project root (see `App.tsx`, `index.tsx`, `config.ts`). Built UI for a chat assistant called Clara.
- Backend: FastAPI-based Clara API in `backend/` (see `backend/main.py` and `backend/modal_app.py`). Supports REST and WebSocket chat. Clara (ML) logic lives in `backend/clara_v2.py` (heavy ML deps) and memory/routing in `backend/hdc_memory_64k.py`, `backend/nemotron_router.py`.
- Deployments: Two modes — local lightweight mode (no models loaded) and Modal deployment (GPU, models from HuggingFace) using `backend/modal_app.py`. See `backend/DEPLOY.md` for the deploy flow.

## Quick dev & debug commands
- Frontend (Vite):
  - Install & run: `npm install` then `npm run dev` (default port 5173)
  - Build: `npm run build` and `npm run preview`
- Backend (local lightweight dev):
  - Create virtualenv: `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
  - Install: `pip install -r backend/requirements.txt` (skip heavy ML deps if you want lightweight mode)
  - Run API: `cd backend && uvicorn main:app --reload --port 8000`
- Modal (production-like with GPUs): follow instructions in `backend/DEPLOY.md` (upload models to HF, set secrets, `modal deploy modal_app.py`).

## Important configuration
- `config.ts` (root): centralizes API base URLs and WS URL (local vs Modal). Update `MODAL_URL` after deploying.
- Env vars used by backend:
  - `DEBUG` (true/false) — toggles verbose logging
  - `CLARA_ROUTER_MODE` (embedding|llm|hybrid)
  - `CLARA_MODELS_DIR` — where to find model files locally
  - `HF_TOKEN` (Modal secret) — used for model downloads in Modal image

## Runtime behavior / gotchas
- Clara is optional: `backend/main.py` performs a guarded import of Clara modules and sets `HAS_CLARA=False` on ImportError. In this case the backend runs in a *lightweight* echo mode — useful for fast iteration without huge ML downloads.
- Modal deployment (`modal_app.py`) provides `POST /api/chat` (sync wrapper to Modal method) and a WebSocket server; frontend `hooks/useClaraChat.ts` expects `/api/chat` by default.
- Tools: `backend/tools.py` uses a `ToolRegistry` pattern. Tools expose a JSON Schema `parameters` field and can be executed via REST (`POST /api/tools/execute`) or via WebSocket tool_call messages.

## Protocol & example messages
- WebSocket endpoint: `/ws/chat`
  - Client → Server
    - Message: `{ "type": "message", "content": "Hello", "session_id": "<optional>" }`
    - Tool call: `{ "type": "tool_call", "tool_name": "summarize", "tool_args": {"text": "..."} }`
    - Ping: `{ "type": "ping" }`
  - Server → Client
    - Typing: `{ "type": "typing", "content": "", "session_id": "..." }`
    - Message: `{ "type": "message", "content": "...", "session_id": "..." }`
    - Tool result: `{ "type": "tool_result", "content": "json-string-or-error", "metadata": {...} }`

- REST examples
  - Chat (Modal): `POST /api/chat` body `{ "content": "Explain recursion", "personality": "warmth" }` (see `backend/modal_app.py`)
  - List tools: `GET /api/tools`
  - Execute tool: `POST /api/tools/execute` body `{ "tool_name": "summarize", "arguments": { "text": "..." } }`

## Code conventions & patterns to follow
- Tools: register via `ToolRegistry.register(ToolDefinition(...))` in `backend/tools.py`. Follow the schema shape used there when adding parameters.
- Chat/session models: Pydantic models used for REST + WebSocket exchanges in backend; mirror shapes in frontend `types.ts`.
- Lazy & guarded loading: Heavy ML pieces are lazy-loaded or guarded. For quick edits/test runs, rely on lightweight mode or mocked responses.
- Frontend: functional React components + Tailwind CSS. Hook `useClaraChat` is the central integration point — update it if you change the chat interface or transports.

## Helpful files to inspect first
- `config.ts` — where the frontend points to the backend
- `App.tsx`, `hooks/useClaraChat.ts` — UI & chat integration
- `backend/main.py` — REST & WebSocket server (local / minimal Clara)
- `backend/modal_app.py` — Modal/GPU deployment and `/api/chat` REST endpoint used by frontend
- `backend/tools.py` — tool registry & handler patterns
- `backend/clara_v2.py` — Clara architecture (HDC memory, router, dual-brain)
- `backend/DEPLOY.md` & `backend/upload_models_to_hf.ipynb` — deployment and HF upload flow

## When you change something
- If you add a new tool: add it to `ToolRegistry` and ensure its `parameters` schema is accurate; update `backend/main.py` or `modal_app.py` as needed.
- If you change message shapes: update both backend Pydantic models and `frontend/types.ts` + `useClaraChat`.
- If you modify the REST/WS API surface: update `config.ts` and `App`/hooks accordingly and test both REST and WS flows.

---

If any part is unclear or you want more/less detail (e.g., examples for testing tools, or a runnable mini-integration test), say which area and I’ll iterate. ✅
