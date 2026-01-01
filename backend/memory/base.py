"""
Base interfaces for Clara's memory system.

These abstract classes define the contract that any memory implementation
must follow, enabling easy swapping of backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class MemoryTier(Enum):
    """Memory tiers with different persistence and recall characteristics."""
    SESSION = "session"      # Working memory, current conversation
    DAILY = "daily"          # Today's consolidated memories
    LONGTERM = "longterm"    # Permanent, high-importance facts


@dataclass
class Memory:
    """
    A single memory unit.

    Attributes:
        id: Unique identifier
        content: The actual memory content (text)
        tier: Which memory tier (session/daily/longterm)
        importance: 0.0-1.0 score for consolidation decisions
        timestamp: When the memory was created
        metadata: Additional context (entities, emotions, topics)
        embedding: Optional vector representation (HDC or traditional)
    """
    id: str
    content: str
    tier: MemoryTier = MemoryTier.SESSION
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            content=data["content"],
            tier=MemoryTier(data["tier"]),
            importance=data["importance"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class MemoryStore(ABC):
    """
    Abstract base class for memory storage backends.

    Implement this interface to create swappable memory systems:
    - HDCMemory (hyperdimensional computing)
    - VectorMemory (traditional embeddings + similarity)
    - HybridMemory (combination approaches)
    """

    @abstractmethod
    def store(self, memory: Memory) -> str:
        """
        Store a memory and return its ID.

        Args:
            memory: The Memory object to store

        Returns:
            The memory's unique ID
        """
        pass

    @abstractmethod
    def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[List[MemoryTier]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Recall memories similar to the query.

        Args:
            query: The search query (text)
            top_k: Maximum number of memories to return
            tier_filter: Optional filter by memory tier

        Returns:
            List of (Memory, similarity_score) tuples, sorted by relevance
        """
        pass

    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """
        Remove a memory by ID.

        Args:
            memory_id: The ID of the memory to remove

        Returns:
            True if removed, False if not found
        """
        pass

    @abstractmethod
    def consolidate(self) -> Dict[str, int]:
        """
        Run consolidation cycle (sleep).

        This promotes important memories between tiers and
        forgets low-importance old memories.

        Returns:
            Stats dict: {"promoted": N, "forgotten": M, "compressed": K}
        """
        pass

    def get_context(
        self,
        query: str,
        max_tokens: int = 1000
    ) -> str:
        """
        Build context string for prompt injection.

        Args:
            query: The current query to find relevant memories for
            max_tokens: Approximate token budget

        Returns:
            Formatted string of relevant memories for prompt injection
        """
        memories = self.recall(query, top_k=10)

        context_parts = []
        estimated_tokens = 0

        for memory, score in memories:
            # Rough token estimate: ~4 chars per token
            mem_tokens = len(memory.content) // 4
            if estimated_tokens + mem_tokens > max_tokens:
                break

            context_parts.append(f"[{memory.tier.value}] {memory.content}")
            estimated_tokens += mem_tokens

        if not context_parts:
            return ""

        return "Relevant memories:\n" + "\n".join(context_parts)


class ConsolidationEngine(ABC):
    """
    Abstract base class for memory consolidation.

    Consolidation is the "sleep" process that:
    - Promotes important session memories to daily/longterm
    - Compresses and summarizes old memories
    - Creates associative bindings between related concepts

    Implementations:
    - ClassicalConsolidation: Traditional importance-based promotion
    - QuantumConsolidation: TFQ/Cirq entanglement for fuzzy binding
    """

    @abstractmethod
    def process(
        self,
        memories: List[Memory]
    ) -> Tuple[List[Memory], Dict[str, Any]]:
        """
        Process memories through consolidation.

        Args:
            memories: List of memories to consolidate

        Returns:
            Tuple of (processed_memories, stats_dict)
        """
        pass

    @abstractmethod
    def create_bindings(
        self,
        memories: List[Memory]
    ) -> Dict[str, List[str]]:
        """
        Create associative bindings between memories.

        Args:
            memories: Memories to find associations between

        Returns:
            Dict mapping memory_id -> [related_memory_ids]
        """
        pass
