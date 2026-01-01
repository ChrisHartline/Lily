# Clara: A Novel Memory Architecture for Embodied AI Assistants

## Overview

Clara is an embodied AI assistant featuring a novel three-tier memory architecture that combines **Hyperdimensional Computing (HDC)**, **graph databases**, and **relational storage** to achieve human-like memory recall. Unlike traditional RAG systems that rely on rigid vector similarity, Clara's memory system enables fuzzy associative recall, relationship-aware context expansion, and natural memory consolidation cycles.

## Why This Matters

### The Problem with Current AI Memory

Most AI assistants suffer from one of two memory problems:

1. **No persistent memory** - Each conversation starts fresh, forgetting everything
2. **Rigid retrieval** - Vector databases find exact matches but miss associative connections ("headaches" doesn't retrieve memories about "stress" even when they're related)

Human memory doesn't work this way. We recall things associatively, make connections across topics, and naturally consolidate important experiences while letting trivial ones fade.

### Our Solution: Three Technologies, One System

Clara combines three complementary technologies:

| Technology | Role | What It Does |
|------------|------|--------------|
| **HDC (Hyperdimensional Computing)** | Recall Engine | Fuzzy, associative similarity matching in 10k-64k dimensions |
| **PostgreSQL** | Persistence Layer | Stores content, metadata, timestamps, HDC vectors |
| **FalkorDB** | Relationship Graph | Tracks entity connections, enables multi-hop reasoning |

Each technology excels at something the others can't do efficiently. Together, they create a memory system that feels more human.

---

## Architecture

### Memory Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                         MEMORY TIERS                             │
├─────────────────────────────────────────────────────────────────┤
│  SESSION (Ephemeral)     DAILY (Buffer)      LONG-TERM (Persist)│
│  ├── Last 10-20 turns    ├── Today's memories ├── Important facts│
│  ├── Weight: 1.0         ├── Weight: 0.7      ├── Weight: 0.5    │
│  └── In-memory only      └── Pending review   └── Survives restart│
└─────────────────────────────────────────────────────────────────┘
```

Memories naturally flow from session → daily → long-term based on importance and access patterns, mimicking human memory consolidation.

### Data Flow: Storing a Memory

```
User says: "I've been having headaches from work stress"
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. HDC ENCODING                                                  │
│    Text → 10,000-dimension bipolar vector                       │
│    O(d) operation - instant, no index rebuilding                │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. POSTGRESQL STORAGE                                            │
│    Content, timestamp, importance, tier, HDC vector (as bytes)  │
│    Structured queries: "all memories from last week"            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. FALKORDB GRAPH                                                │
│    Entities extracted: [Chris, headaches, work, stress]         │
│    Relationships created:                                        │
│      (Chris)-[:EXPERIENCES]->(headaches)                        │
│      (headaches)-[:CAUSED_BY]->(stress)                         │
│      (stress)-[:RELATED_TO]->(work)                             │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Recalling Memories

```
User asks: "How have I been feeling lately?"
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌────────────┐   ┌──────────┐
    │   HDC    │   │  FalkorDB  │   │ PostgreSQL│
    │ Recall   │   │  Expand    │   │  Fetch   │
    └────┬─────┘   └─────┬──────┘   └────┬─────┘
         │               │               │
         ▼               ▼               ▼
    Similar text    Related entities   Full content
    "feeling tired"  stress → work     with metadata
    "overwhelmed"    headaches → Chris
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MERGED CONTEXT                                │
│  Memories: [headaches, stress, work deadline, feeling tired]   │
│  Graph: stress linked to work project and physical symptoms    │
│  → Clara understands the full picture, not just keyword matches │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Makes HDC Special

### The Brain Connection

Neuroscience research suggests the brain uses high-dimensional distributed representations. HDC mimics this with vectors of 10,000+ dimensions where information is spread across all components.

### Key Advantages

1. **O(d) Updates** - Adding a memory is a single vector operation. No index rebuilding like FAISS or Pinecone.

2. **Fuzzy Matching** - Cosine similarity gives "reminds me of..." results, not just exact matches.

3. **Noise Tolerance** - High-dimensional spaces are robust to small variations.

4. **Edge-Friendly** - A 64k-dimension vector is only 256KB. Runs on Jetson, phones, laptops.

### Code Example: HDC Bundle Operation

```python
# Adding a new memory to the bundle is O(d)
def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
    """Combine multiple vectors (memories) into one."""
    bundled = np.sum(vectors, axis=0)
    return np.sign(bundled)  # Normalize to bipolar
```

Compare this to vector databases that require periodic re-indexing as data grows.

---

## The Graph Advantage: Multi-Hop Reasoning

### What FalkorDB Enables

HDC finds memories with similar *text*. FalkorDB finds memories with related *meaning*.

**Example:**
```
User: "I'm feeling overwhelmed again"

HDC finds (text similarity):
  - "I was overwhelmed yesterday" (0.89)
  - "Too much on my plate" (0.76)

FalkorDB traverses (relationships):
  overwhelmed → caused_by → work_stress
  work_stress → related_to → AI_project
  work_stress → manifests_as → headaches
  AI_project → has_deadline → next_week

Combined insight:
  "You mentioned headaches last time you were stressed about the
   AI project deadline. Is that what's going on again?"
```

This multi-hop reasoning is impossible with pure vector similarity.

### Graph RAG Queries

```python
def graph_rag_query(self, query_entities: List[str], max_hops: int = 2):
    """
    Given entities from a query, retrieve:
    - Related entities (up to max_hops away)
    - Relationship paths between entities
    - Linked memory IDs
    """
```

---

## Personality System: Modular Context Loading

Clara's personality isn't hardcoded. It's assembled from JSON modules loaded dynamically based on conversation context.

### Module Structure

```
clara_prompts/
├── clara_core_prompt.json      # Always loaded (identity, values)
├── clara_medical_module.json   # Triggered by: "hospital", "ER", "diagnosis"
├── clara_tech_module.json      # Triggered by: "AI", "software", "code"
├── clara_faith_module.json     # Triggered by: "church", "prayer", "Mass"
└── clara_town_module.json      # Triggered by: "parish", "community"
```

### Silent Context Switching

When the user mentions "the hospital," Clara's medical expertise module loads silently. No "I'm now in medical mode" announcements. The conversation flows naturally.

```python
def get_active_modules(self, user_message: str) -> List[Module]:
    """
    Detect triggers and return:
    - Core module (always)
    - Up to 2 contextual modules (by keyword detection)
    Respects token budget (~8000 tokens for personality)
    """
```

---

## Integration: Memory + Personality + LLM

### The Complete Pipeline

```
User Message: "How was your shift at the ER today?"
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. MEMORY RECALL                                                 │
│    HDC: Find similar past conversations                         │
│    FalkorDB: Expand with related entities                       │
│    Result: "Yesterday Clara mentioned a tough trauma case"      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PERSONALITY ASSEMBLY                                          │
│    Core: Clara's identity, relationship with user               │
│    Triggered: medical_expertise (keyword: "ER")                 │
│    Memory context injected into prompt                          │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. LLM GENERATION (Phi-3 + LoRA)                                │
│    System prompt: ~3000 tokens of rich context                  │
│    Personality adapter: warmth/playful/encouragement            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MEMORY STORAGE                                                │
│    Store user message + Clara's response                        │
│    Extract entities → Update graph                              │
│    Tier: SESSION (may promote later)                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
Response: "It was intense - three traumas back to back.
           Reminds me of what we talked about yesterday..."
```

---

## Future: Quantum-Enhanced Consolidation

The architecture includes a quantum consolidation module (TensorFlow Quantum + Cirq) designed for memory consolidation cycles:

- **Encode importance** as qubit amplitudes
- **Use entanglement** to capture associative relationships
- **Measure interference** to decide: promote, decay, or bind memories

Currently runs on simulators, designed for future Google Quantum AI hardware.

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API | Modal + FastAPI | Serverless GPU deployment |
| LLM | Phi-3 (4-bit quantized) | Knowledge generation |
| Personality | LoRA adapters | Voice/tone fine-tuning |
| Memory | HDC (10k-64k dim) | Associative recall |
| Persistence | PostgreSQL | Structured storage |
| Graph | FalkorDB | Relationship tracking |
| Future | TFQ + Cirq | Quantum consolidation |

---

## Key Innovations

1. **HDC for Memory Recall** - First application of hyperdimensional computing to conversational AI memory with O(d) updates

2. **Three-Layer Memory Stack** - Combines HDC (recall), PostgreSQL (persistence), FalkorDB (relationships) for comprehensive memory

3. **Graph RAG** - Uses relationship traversal to find contextually related memories, not just textually similar ones

4. **Modular Personality** - JSON-based personality modules with trigger-based loading and token budget management

5. **Memory-Aware Prompts** - Relevant memories injected into system prompts for contextual responses

---

## Getting Started

### Local Development

```bash
# Start databases
docker run -p 5432:5432 postgres
docker run -p 6379:6379 falkordb/falkordb

# Run tests
python -m backend.memory.test_integrated_memory

# Start Modal locally
modal serve backend/modal_app.py
```

### Cloud Deployment

```bash
# Set Modal secrets for cloud databases
modal secret create clara-db \
  POSTGRES_HOST=your-supabase-url \
  FALKOR_HOST=your-falkordb-cloud-url

# Deploy
modal deploy backend/modal_app.py
```

---

## Conclusion

Clara demonstrates that human-like memory in AI assistants is achievable by combining the right technologies:

- **HDC** for fast, fuzzy, associative recall
- **Graphs** for relationship-aware context expansion
- **Relational storage** for structured persistence
- **Modular personality** for natural conversation flow

The result is an AI that doesn't just retrieve relevant information—it *remembers* in a way that feels genuinely human.

---

*Project: Clara/Lily | Author: Chris Hartline | Date: December 2024*
