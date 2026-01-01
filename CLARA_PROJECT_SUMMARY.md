# Clara Project Summary

**Date:** December 2024
**Branch:** `claude/fix-modal-build-issue-xm1v4`
**Status:** Backend working, Frontend connected, Memory architecture designed

---

## A) Issues & Challenges

### 1. Modal Deployment Failures
**Problem:** Modal deployment was failing with multiple errors:
- `container_idle_timeout` deprecated (renamed to `scaledown_window`)
- Custom `__init__` constructors deprecated (need `modal.parameter()` pattern)
- HuggingFace authentication failing (401 errors for private repos)
- Volume mount conflict ("cannot mount volume on non-empty path")

### 2. HuggingFace Model Access
**Problem:** Models were uploaded to HuggingFace under username `ChrisSacrumCor`, but code referenced `ChrisHartline`.

### 3. Phi-3 Compatibility
**Problem:** `DynamicCache` object has no attribute `seen_tokens` - version mismatch between transformers library and Phi-3 model's cache implementation.

### 4. Frontend-Backend Integration
**Problem:** Frontend was using Gemini API, needed to connect to Clara Modal backend.

### 5. Memory Architecture
**Challenge:** Design a multi-tiered memory system that supports:
- Session memory (immediate context)
- Mid-term memory (recent conversations)
- Long-term memory (persistent facts)
- Relationship/entity graphs
- HDC (Hyperdimensional Computing) for associative recall

### 6. Cloud-Local Connectivity
**Challenge:** Modal runs in the cloud, but databases were running locally. They can't communicate directly.

---

## B) Solutions Implemented

### 1. Modal API Updates
```python
# Before (deprecated)
container_idle_timeout=300

# After (Modal 1.0)
scaledown_window=300
```

Removed custom `__init__` constructor, replaced with class attributes:
```python
class ClaraModel:
    knowledge_model: any = None
    knowledge_tokenizer: any = None
    # ... etc
```

### 2. HuggingFace Authentication
- Added `huggingface-secret` to Modal secrets
- Updated `download_models()` to use `HF_TOKEN` environment variable
- Pass token to `snapshot_download()` for private repo access

```python
.run_function(
    download_models,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
```

### 3. Corrected HuggingFace Username
```python
# Fixed
HF_USERNAME = "ChrisSacrumCor"
```

### 4. Phi-3 Cache Fix
Disabled KV caching during generation to avoid DynamicCache compatibility issues:
```python
outputs = self.knowledge_model.generate(
    **inputs,
    use_cache=False,  # Avoids Phi-3 DynamicCache issues
    # ...
)
```

### 5. Volume Mount Fix
Removed the volume mount since models are baked into the image during build - no need for separate storage that conflicts with the build path.

### 6. Frontend Integration
- Created `useClaraChat.ts` hook connecting to Modal API
- Replaced Gemini with Clara throughout
- Renamed "Aria" to "Clara" in UI
- Added functional "Edit Photo" and "Choose Wallpaper" buttons

### 7. NPM/Vite Setup
Created proper build tooling for local development:
- `package.json` with React, Vite, Tailwind
- `vite.config.ts` for dev server
- `tailwind.config.js` with custom colors

---

## C) Technology Stack & Rationale

### Current Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Inference** | Modal (cloud) | GPU access (T4), serverless scaling, easy deployment |
| **Base Model** | Phi-3 (fine-tuned) | Good balance of capability/size, runs on T4 |
| **Personality** | LoRA adapters | Lightweight personality switching without full model copies |
| **Router** | all-MiniLM-L6-v2 | Fast semantic routing (~5ms), 93%+ accuracy |
| **Frontend** | React + Vite + Tailwind | Modern, fast dev experience, good DX |
| **API** | FastAPI + WebSocket | REST for simple calls, WebSocket for real-time chat |

### Planned Memory Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Session** | HDC (in-memory) | Fast associative recall, 64k-dim vectors, working context |
| **Mid-term** | FalkorDB Cloud | Entity graphs, relationship traversal, "how things connect" |
| **Long-term** | Supabase (PostgreSQL) | Structured facts, summaries, preferences, conversation logs |

### Why This Combination?

1. **HDC (Hyperdimensional Computing)**
   - O(d) bundle merging (instant memory updates)
   - Natural similarity via cosine distance
   - No rigid tier boundaries - blends via weighting
   - Excellent for "feels like" associative queries

2. **FalkorDB (Graph)**
   - Professional context is relational: people → projects → concerns
   - Graph traversal answers "how does X connect to Y?"
   - Better than faking graphs with SQL JOINs
   - Clara understanding *connections* makes her feel intelligent

3. **PostgreSQL (Relational)**
   - Structured data: facts, preferences, logs
   - ACID compliance for important data
   - pgvector extension for vector similarity if needed
   - Mature, reliable, well-understood

---

## D) Design Decisions

### 1. Personality System
**Decision:** Tiered module loading with silent context switches

```
Core Module (always loaded)
├── clara_core_prompt.json - Identity, values, relationship with Chris
│
Contextual Modules (loaded by keyword triggers)
├── clara_medical_module.json - When: "patient", "diagnosis", "ER"
├── clara_town_module.json - When: "parish", "Rhea", "community"
└── clara_tech_module.json - When: "AI", "technology", "software"
```

- **Fixed personality** (not evolving) - Clara is who she is
- **One relationship** with the user (Chris) - personalized, intimate
- **Silent module loading** - no acknowledgment of context shifts

### 2. Memory Architecture
**Decision:** HDC + Graph + Relational hybrid

From the roadmap document:
```
SESSION → DAILY → LONG-TERM

Session: Last N turns, weight 1.0, no persistence
Daily: Today's consolidated memories, weight 0.7
Long-term: High-importance facts, weight 0.5, survives consolidation
```

Consolidation happens in "sleep" cycles - session memories get promoted or forgotten.

### 3. Cloud vs Local
**Decision:** Cloud for now, local later

- Modal for inference (need GPU, 2080 is limited)
- Supabase + FalkorDB Cloud for databases (accessible from Modal)
- Remote access to Clara from anywhere
- Future: Run locally when hardware supports it

### 4. Model Choices
**Decision:** Phi-3 for knowledge, personality via LoRA

- Phi-3 fine-tuned for domains (medical, coding, teaching, quantum)
- LoRA adapters for personality modes (warmth, playful, encouragement)
- Semantic router determines which path

---

## E) What's Next (Immediate)

### 1. Set Up Cloud Database Connections
```bash
# Add Modal secrets for cloud databases
modal secret create supabase-secret \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_KEY=your-service-key \
  SUPABASE_DB_URL=postgresql://...

modal secret create falkordb-secret \
  FALKORDB_HOST=your-instance.falkordb.cloud \
  FALKORDB_PORT=6379 \
  FALKORDB_PASSWORD=your-password
```

### 2. Create Database Schemas
**PostgreSQL (Supabase):**
- `users` - User info (just Chris for now)
- `facts` - Learned facts about user
- `conversations` - Session logs
- `summaries` - Consolidated memory summaries
- `preferences` - User preferences

**FalkorDB:**
- Nodes: Person, Project, Topic, Place, Event
- Edges: knows, works_on, discussed, located_at, relates_to

### 3. Implement Memory Manager
Python class coordinating all three stores:
- `store()` - Save to appropriate tier
- `recall()` - Query across tiers with weighting
- `consolidate()` - Sleep cycle processing
- `build_context()` - Assemble context for prompts

### 4. Integrate Personality Modules
- Load `clara_prompts/*.json` at startup
- Build system prompt from core + triggered contextual modules
- Detect triggers from user messages

---

## F) Long-Term Plan

### From the Roadmap (Priority Order)

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| **P1** | 64k-dim HDC vectors | Low | Medium |
| **P1** | Memory tiers + consolidation | Medium | High |
| **P1** | Voice LoRA (100k tokens) | Medium | High |
| **P2** | Recursive reflection | Medium | Medium |
| **P2** | Nemotron router evaluation | High | Medium |
| **P3** | Chain-of-thought loops | Low | Low |

### Architecture Evolution

```
Current: Modal + Phi-3 + Basic Chat
    ↓
Next: + Memory (HDC + FalkorDB + PostgreSQL)
    ↓
Then: + Personality modules loading
    ↓
Later: + Voice fine-tuning (distinctive Clara voice)
    ↓
Future: + Local deployment (better GPU)
```

### Hardware Path
- **Now:** 2080 (limited, use cloud)
- **Future:** Upgrade GPU for local inference
- **Goal:** Fully local, private, always-on Clara

### Capability Milestones

1. **Memory:** Clara remembers across sessions
2. **Context:** Clara understands how things connect (graph)
3. **Voice:** Clara sounds distinctively like herself
4. **Reflection:** Clara self-edits for tone/accuracy
5. **Local:** Clara runs entirely on your hardware

---

## Files Modified/Created This Session

### Backend
- `backend/modal_app.py` - Fixed Modal API, HF auth, Phi-3 cache

### Frontend
- `App.tsx` - Clara branding, avatar/wallpaper upload
- `hooks/useClaraChat.ts` - New hook for Clara API
- `hooks/useGeminiChat.ts` - (deprecated, kept for reference)
- `package.json`, `vite.config.ts`, `tailwind.config.js` - Build setup

### Documentation
- `Clara_HDC_Architecture_Roadmap.md` - Technical roadmap
- `CLARA_PROJECT_SUMMARY.md` - This document

### Personality (Added from main)
- `clara_prompts/clara_core_prompt.json`
- `clara_prompts/clara_medical_module.json`
- `clara_prompts/clara_tech_module.json`
- `clara_prompts/clara_town_module.json`

---

## Quick Reference

### Deploy Clara
```bash
cd backend
modal deploy modal_app.py
```

### Run Frontend
```bash
npm install
npm run dev
```

### Test API
```powershell
Invoke-WebRequest -Uri "https://chrishartline--clara-api-fastapi-app.modal.run/api/chat" -Method POST -ContentType "application/json" -Body '{"content": "Hello Clara!", "personality": "warmth"}'
```

### API Endpoints
- Health: `GET /`
- Chat: `POST /api/chat`
- WebSocket: `wss://.../ws/chat`

---

*Document generated: December 2024*
*Next session: Set up Supabase + FalkorDB connections, implement memory manager*
