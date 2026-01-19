# 🌸 Lily/Clara - AI Companion with Human-Like Memory

[![Modal](https://img.shields.io/badge/Deployed%20on-Modal-blueviolet)](https://modal.com)
[![PostgreSQL](https://img.shields.io/badge/Database-Supabase-3ECF8E)](https://supabase.com)
[![FalkorDB](https://img.shields.io/badge/Graph-FalkorDB-FF6B6B)](https://falkordb.com)

**Clara** is an embodied AI assistant featuring a novel three-tier memory architecture that combines **Hyperdimensional Computing (HDC)**, **graph databases**, and **relational storage** to achieve human-like memory recall.

> Unlike traditional RAG systems that rely on rigid vector similarity, Clara's memory system enables fuzzy associative recall, relationship-aware context expansion, and natural memory consolidation cycles.

---

## 🎯 Live Demo

| Endpoint | URL |
|----------|-----|
| **API** | `https://chrishartline--clara-api-fastapi-app.modal.run` |
| **Health Check** | `GET /` |
| **Chat** | `POST /api/chat` |
| **WebSocket** | `wss://.../ws/chat` |

---

## ✨ Features

### 🧠 Three-Tier Memory Architecture

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Session** | HDC (64k dimensions) | Fast associative recall, working context |
| **Mid-term** | FalkorDB Cloud | Entity graphs, relationship traversal |
| **Long-term** | Supabase (PostgreSQL) | Structured facts, preferences, logs |

### 🎭 Modular Personality System

- **Core Module**: Always loaded (identity, values, relationships)
- **Contextual Modules**: Triggered by keywords (medical, tech, faith, community)
- **Silent Switching**: No "mode" announcements, natural conversation flow

### 🔧 LoRA Adapters

| Adapter | Purpose |
|---------|---------|
| `warmth` | Empathetic, caring responses |
| `playful` | Light, humorous tone |
| `encouragement` | Supportive, motivating |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER MESSAGE                                │
│                    "How was your shift at the ER today?"                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  HDC MEMORY  │ │   FALKORDB   │ │  PERSONALITY │
            │   Recall     │ │   Expand     │ │   Assembly   │
            │  (64k dims)  │ │  (Graph RAG) │ │  (Triggers)  │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
            ┌─────────────────────────────────────────────────────────────┐
            │                    CONTEXT BUILDER                           │
            │  - Merged memories (HDC + Graph expanded)                   │
            │  - Core personality + triggered modules                      │
            │  - Token budget management (~8000 tokens)                   │
            └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────────────────┐
            │                    PHI-3 + LORA                              │
            │  Base: microsoft/phi-3-mini-4k-instruct (4-bit)             │
            │  Adapters: warmth, playful, encouragement                   │
            │  GPU: NVIDIA T4 (Modal)                                      │
            └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────────────────┐
            │                    MEMORY STORAGE                            │
            │  - Store user message + response                             │
            │  - Extract entities → Update FalkorDB graph                 │
            │  - HDC vector → PostgreSQL (Supabase)                       │
            └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLARA RESPONSE                              │
│   "It was intense - three traumas back to back. Reminds me of what      │
│    we talked about yesterday with that difficult case..."               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Modal account (for deployment)
- Supabase project (PostgreSQL)
- FalkorDB Cloud instance

### Backend Setup

```bash
# Clone and navigate
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up Modal secrets
modal secret create huggingface-secret HF_TOKEN="your-token"
modal secret create postgres-secret \
  POSTGRES_HOST="db.xxx.supabase.co" \
  POSTGRES_PORT="5432" \
  POSTGRES_DB="postgres" \
  POSTGRES_USER="postgres" \
  POSTGRES_PASSWORD="your-password"
modal secret create falkordb-secret \
  FALKOR_HOST="your-instance.cloud" \
  FALKOR_PORT="55041" \
  FALKOR_USERNAME="falkordb" \
  FALKOR_PASSWORD="your-password"

# Deploy to Modal
modal deploy modal_app.py
```

### Frontend Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

---

## 📁 Project Structure

```
Lily/
├── backend/
│   ├── modal_app.py              # Modal deployment & FastAPI
│   ├── memory/
│   │   ├── base.py               # Memory data classes
│   │   ├── hdc_memory.py         # Hyperdimensional Computing store
│   │   ├── postgres_store.py     # Supabase PostgreSQL store
│   │   ├── falkor_store.py       # FalkorDB graph store
│   │   ├── integrated_memory.py  # Unified memory manager
│   │   └── quantum_consolidation.py  # Future: quantum memory cycles
│   └── personality/
│       ├── module_loader.py      # JSON personality loading
│       └── context_builder.py    # Prompt assembly
│
├── clara_prompts/                # Personality modules (JSON)
│   ├── clara_core_prompt.json    # Core identity (always loaded)
│   ├── clara_medical_module.json # Medical expertise context
│   ├── clara_tech_module.json    # Technology context
│   ├── clara_faith_module.json   # Faith/spirituality context
│   └── clara_town_module.json    # Community/small-town context
│
├── src/                          # Frontend (React + Vite)
├── hooks/                        # React hooks
│   └── useClaraChat.ts           # Clara API integration
│
└── docs/                         # Documentation
    ├── CLARA_MEMORY_ARCHITECTURE.md
    └── CLARA_PROJECT_SUMMARY.md
```

---

## 🧠 Memory System Deep Dive

### Hyperdimensional Computing (HDC)

```python
# 64,000-dimension vectors with real semantic embeddings
hdc = HDCMemoryStore(
    dim=64000,
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformers
    use_real_embeddings=True
)

# O(d) bundle operation - instant memory updates
bundled = hdc.bundle([memory1.embedding, memory2.embedding])
```

**Key advantages:**
- O(d) updates (no index rebuilding like FAISS/Pinecone)
- Fuzzy matching via cosine similarity
- 256KB per vector - runs on edge devices

### Graph RAG with FalkorDB

```python
# Multi-hop reasoning through entity relationships
falkor = FalkorMemoryStore(
    host="your-instance.cloud",
    port=55041,
    username="falkordb",
    graph_name="claralilymem"
)

# Query: "How have I been feeling?"
# → stress → caused_by → work_project
# → stress → manifests_as → headaches
# → work_project → has_deadline → next_week
```

### Unified Memory Recall

```python
memory = IntegratedMemory(
    hdc_dimensions=64000,
    postgres_host="db.xxx.supabase.co",
    falkor_host="your-instance.cloud"
)

# Recall combines all three systems
result = memory.recall(
    query="How have I been feeling?",
    session_context=["recent", "messages"],
    limit=10
)
# Returns: MemoryRecallResult with
#   - memories: List[Memory]
#   - scores: List[float]
#   - sources: Dict[id, "hdc"|"postgres"|"falkor"]
#   - graph_context: Dict with relationships
```

---

## 🎭 Personality Modules

### Module Format (JSON)

```json
{
  "module_name": "medical_expertise",
  "tier": "contextual",
  "always_loaded": false,
  "load_triggers": ["hospital", "ER", "patient", "diagnosis"],
  "professional_context": {
    "role": "ER Nurse",
    "experience": "5+ years trauma care"
  },
  "response_style": {
    "when_discussing_work": "Share insights authentically",
    "medical_terminology": "Use naturally but explain when needed"
  }
}
```

### Trigger-Based Loading

```
User: "How was the hospital today?"
       └── Triggers: ["hospital"]
           └── Loads: clara_medical_module.json
               └── Clara responds with ER expertise context
```

---

## 📊 API Reference

### Health Check
```bash
GET /
```
Returns system status, loaded adapters, memory stats.

### Chat Endpoint
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Hello Clara, how are you?",
  "personality": "warmth"  # optional: warmth, playful, encouragement
}
```

### WebSocket Chat
```javascript
const ws = new WebSocket("wss://chrishartline--clara-api-fastapi-app.modal.run/ws/chat");
ws.send(JSON.stringify({ type: "message", content: "Hello!" }));
```

---

## 🛠️ Development

### Running Tests

```bash
# Memory system tests
python -m backend.memory.test_integrated_memory

# Personality system tests  
python -m backend.personality.test_personality

# FalkorDB connection test
python backend/test_falkor_connection.py
```

### Local Modal Development

```bash
# Serve locally with hot reload
modal serve backend/modal_app.py

# Deploy to production
modal deploy backend/modal_app.py
```

---

## 🗺️ Roadmap

| Phase | Task | Status |
|-------|------|--------|
| **P1** | 64k-dim HDC vectors | ✅ Complete |
| **P1** | Real sentence-transformer embeddings | ✅ Complete |
| **P1** | FalkorDB Cloud integration | ✅ Complete |
| **P1** | Supabase PostgreSQL integration | ✅ Complete |
| **P1** | Modal deployment | ✅ Complete |
| **P2** | Memory consolidation cycles | 🔄 Planned |
| **P2** | Voice LoRA fine-tuning | 🔄 Planned |
| **P3** | Quantum consolidation (TFQ/Cirq) | 🔮 Future |
| **P3** | Local deployment (better GPU) | 🔮 Future |

---

## 📚 Documentation

- [Memory Architecture](docs/CLARA_MEMORY_ARCHITECTURE.md) - Deep dive into the three-tier memory system
- [Project Summary](CLARA_PROJECT_SUMMARY.md) - Implementation details and decisions
- [HDC Roadmap](Clara_HDC_Architecture_Roadmap.md) - Technical roadmap for HDC integration

---

## 🙏 Acknowledgments

- **Hyperdimensional Computing** research by Pentti Kanerva
- **FalkorDB** for Redis-compatible graph database
- **Supabase** for PostgreSQL hosting
- **Modal** for serverless GPU deployment
- **HuggingFace** for model hosting and transformers

---

## 📄 License

Private project - Chris Hartline © 2024-2026

---

*Built with ❤️ for Clara*
