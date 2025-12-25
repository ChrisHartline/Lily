"""
Hyperdimensional Computing (HDC) Memory Implementation

HDC uses high-dimensional vectors (10k-64k dimensions) for:
- O(1) similarity computation
- O(d) bundle merging (instant memory updates)
- Natural "fuzzy" recall via cosine similarity
- No rigid tier boundaries - weighted blending

Key operations:
- Encoding: Text → High-dimensional bipolar vector
- Bundling: Combine vectors (superposition)
- Binding: Associate vectors (XOR-like operation)
- Similarity: Cosine distance for recall

Reference: Clara_HDC_Architecture_Roadmap.md
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json

from .base import MemoryStore, Memory, MemoryTier, ConsolidationEngine


class HDCMemory(MemoryStore):
    """
    HDC-based memory store for Clara.

    Configurable dimensions:
    - 10k-dim: Good for edge devices (Jetson), <1000 memories
    - 64k-dim: Server deployment, 10k+ memories, better noise tolerance

    Tier weights control recall priority:
    - Session: 1.0 (highest priority, current context)
    - Daily: 0.7 (today's consolidated)
    - Long-term: 0.5 (permanent facts)
    """

    def __init__(
        self,
        dim: int = 10000,
        tier_weights: Optional[Dict[str, float]] = None,
        importance_threshold: float = 0.7,
        decay_days: int = 7,
        seed: int = 42
    ):
        """
        Initialize HDC memory.

        Args:
            dim: Vector dimensions (10000 for edge, 64000 for server)
            tier_weights: Recall weights per tier
            importance_threshold: Min importance to promote to longterm
            decay_days: Days before low-importance memories decay
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.tier_weights = tier_weights or {
            'session': 1.0,
            'daily': 0.7,
            'longterm': 0.5,
        }
        self.importance_threshold = importance_threshold
        self.decay_days = decay_days

        # Random projection matrix for text encoding
        np.random.seed(seed)
        self.projection = np.random.randn(dim, 768)  # Assuming 768-dim input embeddings

        # Memory storage: id -> (Memory, hypervector)
        self.memories: Dict[str, Tuple[Memory, np.ndarray]] = {}

        # Bindings: associative connections between memories
        self.bindings: Dict[str, List[str]] = {}

    def _text_to_hv(self, text: str) -> np.ndarray:
        """
        Encode text to hyperdimensional vector.

        Simple approach: Hash-based encoding with random projections.
        For production: Use sentence-transformers embedding first.

        Args:
            text: Input text

        Returns:
            Bipolar hypervector (-1, +1) of shape (dim,)
        """
        # Create deterministic seed from text
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash % (2**32))

        # Generate base embedding (simulated - replace with real embedder)
        base_embedding = np.random.randn(768)

        # Project to high-dimensional space
        hv = self.projection @ base_embedding

        # Bipolarize: sign function
        hv = np.sign(hv)
        hv[hv == 0] = 1  # Handle zeros

        return hv

    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute cosine similarity between hypervectors.

        Args:
            hv1, hv2: Hypervectors to compare

        Returns:
            Similarity score in [-1, 1], typically [0, 1] for similar items
        """
        dot = np.dot(hv1, hv2)
        norm = np.linalg.norm(hv1) * np.linalg.norm(hv2)
        if norm == 0:
            return 0.0
        return dot / norm

    def _bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors (superposition).

        This is HDC's "killer feature" - O(d) memory update.
        Bundling creates a vector similar to all inputs.

        Args:
            hvs: List of hypervectors to bundle

        Returns:
            Bundled hypervector
        """
        if not hvs:
            return np.zeros(self.dim)

        # Element-wise sum + sign
        bundled = np.sum(hvs, axis=0)
        bundled = np.sign(bundled)
        bundled[bundled == 0] = 1

        return bundled

    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (association).

        Binding creates a vector dissimilar to both inputs
        but can be "unbound" with the same operation.

        XOR for bipolar: element-wise multiplication

        Args:
            hv1, hv2: Hypervectors to bind

        Returns:
            Bound hypervector
        """
        return hv1 * hv2

    def _recency_weight(self, memory: Memory) -> float:
        """
        Calculate recency weight for memory scoring.

        More recent memories get higher weight.

        Args:
            memory: Memory to score

        Returns:
            Recency weight in [0, 1]
        """
        age = datetime.now() - memory.timestamp
        hours = age.total_seconds() / 3600

        # Exponential decay with 24-hour half-life
        return np.exp(-hours / 24)

    def store(self, memory: Memory) -> str:
        """Store a memory with its hypervector encoding."""
        # Generate ID if not provided
        if not memory.id:
            memory.id = hashlib.sha256(
                f"{memory.content}{memory.timestamp}".encode()
            ).hexdigest()[:16]

        # Encode to hypervector
        hv = self._text_to_hv(memory.content)
        memory.embedding = hv

        # Store
        self.memories[memory.id] = (memory, hv)

        return memory.id

    def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories similar to query.

        Scoring combines:
        - Semantic similarity (HDC cosine)
        - Tier weight
        - Importance
        - Recency

        No rigid boundaries - all tiers searched with weighting.
        """
        query_hv = self._text_to_hv(query)

        results = []
        for memory_id, (memory, hv) in self.memories.items():
            # Apply tier filter if specified
            if tier_filter and memory.tier not in tier_filter:
                continue

            # Base similarity
            base_sim = self._similarity(query_hv, hv)

            # Get tier weight
            tier_weight = self.tier_weights.get(memory.tier.value, 0.5)

            # Recency weight
            recency = self._recency_weight(memory)

            # Combined score
            score = base_sim * tier_weight * memory.importance * (0.5 + 0.5 * recency)

            results.append((memory, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            # Clean up bindings
            self.bindings.pop(memory_id, None)
            for related_ids in self.bindings.values():
                if memory_id in related_ids:
                    related_ids.remove(memory_id)
            return True
        return False

    def consolidate(self) -> Dict[str, int]:
        """
        Run consolidation cycle.

        1. Session → Daily (all session memories)
        2. Daily → Long-term (high importance only)
        3. Forget old low-importance daily memories
        4. Extract patterns (future: quantum binding)
        """
        stats = {"promoted": 0, "forgotten": 0, "compressed": 0}
        now = datetime.now()

        for memory_id, (memory, hv) in list(self.memories.items()):
            # Session → Daily (promote all)
            if memory.tier == MemoryTier.SESSION:
                memory.tier = MemoryTier.DAILY
                stats["promoted"] += 1

            # Daily → Long-term (high importance only)
            elif memory.tier == MemoryTier.DAILY:
                if memory.importance >= self.importance_threshold:
                    memory.tier = MemoryTier.LONGTERM
                    stats["promoted"] += 1
                elif (now - memory.timestamp).days > self.decay_days:
                    # Forget old low-importance memories
                    self.forget(memory_id)
                    stats["forgotten"] += 1

        return stats

    def get_all_memories(self, tier: Optional[MemoryTier] = None) -> List[Memory]:
        """Get all memories, optionally filtered by tier."""
        memories = [mem for mem, _ in self.memories.values()]
        if tier:
            memories = [m for m in memories if m.tier == tier]
        return memories

    def save(self, path: str):
        """Save memory state to file."""
        data = {
            "dim": self.dim,
            "tier_weights": self.tier_weights,
            "memories": {
                mid: {
                    "memory": mem.to_dict(),
                    "hv": hv.tolist()
                }
                for mid, (mem, hv) in self.memories.items()
            },
            "bindings": self.bindings
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load memory state from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.dim = data["dim"]
        self.tier_weights = data["tier_weights"]
        self.bindings = data["bindings"]

        self.memories = {}
        for mid, item in data["memories"].items():
            mem = Memory.from_dict(item["memory"])
            hv = np.array(item["hv"])
            mem.embedding = hv
            self.memories[mid] = (mem, hv)


class ClassicalConsolidation(ConsolidationEngine):
    """
    Classical (non-quantum) consolidation engine.

    Uses importance scoring and pattern extraction
    without quantum entanglement.

    Good for:
    - Development/testing
    - Edge deployment without quantum simulation
    - Baseline comparison
    """

    def __init__(self, importance_threshold: float = 0.7):
        self.importance_threshold = importance_threshold

    def process(
        self,
        memories: List[Memory]
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """Process memories through classical consolidation."""
        stats = {"promoted": 0, "forgotten": 0, "patterns": 0}

        processed = []
        for memory in memories:
            if memory.importance >= self.importance_threshold:
                # Promote to longterm
                memory.tier = MemoryTier.LONGTERM
                stats["promoted"] += 1
            processed.append(memory)

        return processed, stats

    def create_bindings(
        self,
        memories: List[Memory]
    ) -> Dict[str, List[str]]:
        """Create simple keyword-based bindings."""
        bindings = {}

        for i, mem1 in enumerate(memories):
            related = []
            words1 = set(mem1.content.lower().split())

            for j, mem2 in enumerate(memories):
                if i == j:
                    continue
                words2 = set(mem2.content.lower().split())

                # Simple overlap check
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                if overlap > 0.2:
                    related.append(mem2.id)

            if related:
                bindings[mem1.id] = related

        return bindings
