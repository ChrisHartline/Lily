"""
Integrated Memory Manager

Combines HDC, PostgreSQL, and FalkorDB into a unified memory system.

Data Flow:
1. STORE: Content → HDC encode → PostgreSQL persist → FalkorDB graph
2. RECALL: Query → HDC similarity → FalkorDB expand → PostgreSQL fetch
3. CONSOLIDATE: Quantum/Classical process → Update tiers → Prune graph

This is the main interface for Clara's memory operations.
"""

import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import Memory, MemoryTier, ConsolidationEngine
from .hdc_memory import HDCMemory
from .postgres_store import PostgresMemoryStore, POSTGRES_AVAILABLE
from .falkor_store import FalkorMemoryStore, Entity, FALKOR_AVAILABLE


@dataclass
class RecallResult:
    """Result from memory recall with full context."""
    memories: List[Memory]
    scores: List[float]
    graph_context: Dict[str, Any]
    sources: Dict[str, str]  # memory_id -> source (hdc, graph, direct)


class IntegratedMemory:
    """
    Unified memory manager combining three technologies:

    - HDC (Hyperdimensional Computing): Fast associative recall
    - PostgreSQL: Persistent structured storage
    - FalkorDB: Relationship graph for context expansion

    Each serves a distinct purpose:
    - HDC answers: "What memories are similar to this?"
    - FalkorDB answers: "What else is related through relationships?"
    - PostgreSQL answers: "Give me the full memory content"
    """

    def __init__(
        self,
        # HDC settings
        hdc_dimensions: int = 10000,
        # PostgreSQL settings
        postgres_host: str = "localhost",
        postgres_port: int = 5432,
        postgres_db: str = "clara",
        postgres_user: str = "postgres",
        postgres_password: str = None,
        # FalkorDB settings
        falkor_host: str = "localhost",
        falkor_port: int = 6379,
        falkor_graph: str = "clara_memory",
        # Feature flags
        enable_postgres: bool = True,
        enable_falkor: bool = True,
        enable_graph_expansion: bool = True,
    ):
        """
        Initialize integrated memory system.

        Args:
            hdc_dimensions: HDC vector dimensions (10k-64k)
            postgres_*: PostgreSQL connection settings
            falkor_*: FalkorDB connection settings
            enable_postgres: Whether to use PostgreSQL (required for persistence)
            enable_falkor: Whether to use FalkorDB (for graph features)
            enable_graph_expansion: Whether to expand recall with graph
        """
        self.enable_postgres = enable_postgres and POSTGRES_AVAILABLE
        self.enable_falkor = enable_falkor and FALKOR_AVAILABLE
        self.enable_graph_expansion = enable_graph_expansion

        # Initialize HDC (always enabled - core recall engine)
        self.hdc = HDCMemory(dim=hdc_dimensions)
        print(f"[IntegratedMemory] HDC initialized ({hdc_dimensions} dimensions)")

        # Initialize PostgreSQL
        self.postgres = None
        if self.enable_postgres:
            try:
                self.postgres = PostgresMemoryStore(
                    host=postgres_host,
                    port=postgres_port,
                    database=postgres_db,
                    user=postgres_user,
                    password=postgres_password,
                )
            except Exception as e:
                print(f"[IntegratedMemory] PostgreSQL disabled: {e}")
                self.enable_postgres = False

        # Initialize FalkorDB
        self.falkor = None
        if self.enable_falkor:
            try:
                self.falkor = FalkorMemoryStore(
                    host=falkor_host,
                    port=falkor_port,
                    graph_name=falkor_graph,
                )
            except Exception as e:
                print(f"[IntegratedMemory] FalkorDB disabled: {e}")
                self.enable_falkor = False

        # Session memory (in-memory, no persistence)
        self.session_memories: List[Memory] = []
        self.max_session_memories = 20

        print(f"[IntegratedMemory] Ready: HDC=✓, PostgreSQL={'✓' if self.enable_postgres else '✗'}, FalkorDB={'✓' if self.enable_falkor else '✗'}")

    def store(
        self,
        content: str,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.SESSION,
        metadata: Dict[str, Any] = None,
        extract_entities: bool = True,
    ) -> Memory:
        """
        Store a new memory across all enabled backends.

        Flow:
        1. Create Memory object with HDC embedding
        2. Store in PostgreSQL (if enabled)
        3. Extract entities and store in FalkorDB (if enabled)
        4. Add to session buffer (if session tier)

        Args:
            content: Memory content text
            importance: Importance score (0-1)
            tier: Memory tier (session, daily, longterm)
            metadata: Additional metadata
            extract_entities: Whether to extract and graph entities

        Returns:
            Created Memory object
        """
        # Create memory with unique ID
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            tier=tier,
            importance=importance,
            timestamp=datetime.now(),
            metadata=metadata or {},
            embedding=None  # Will be set by HDC
        )

        # 1. Encode with HDC and store in HDC index
        memory_id = self.hdc.store(memory)
        # HDC.store() sets the embedding on the memory object

        # 2. Persist to PostgreSQL
        if self.enable_postgres:
            try:
                self.postgres.store(memory)
            except Exception as e:
                print(f"[IntegratedMemory] PostgreSQL store failed: {e}")

        # 3. Extract entities and store in graph
        if self.enable_falkor and extract_entities:
            try:
                stats = self.falkor.store_memory_graph(memory)
                memory.metadata['graph_stats'] = stats
            except Exception as e:
                print(f"[IntegratedMemory] FalkorDB store failed: {e}")

        # 4. Add to session buffer if session tier
        if tier == MemoryTier.SESSION:
            self.session_memories.append(memory)
            # Trim session buffer
            if len(self.session_memories) > self.max_session_memories:
                self.session_memories = self.session_memories[-self.max_session_memories:]

        return memory

    def recall(
        self,
        query: str,
        top_k: int = 5,
        tier_filter: Optional[MemoryTier] = None,
        expand_with_graph: bool = None,
        include_session: bool = True,
    ) -> RecallResult:
        """
        Recall memories relevant to a query.

        Flow:
        1. HDC similarity search for direct matches
        2. FalkorDB graph expansion for related context
        3. PostgreSQL fetch for full memory content
        4. Merge and rank results

        Args:
            query: Query text
            top_k: Number of results to return
            tier_filter: Optional tier filter
            expand_with_graph: Whether to expand with graph (default: self.enable_graph_expansion)
            include_session: Whether to include session memories

        Returns:
            RecallResult with memories, scores, and graph context
        """
        expand_with_graph = expand_with_graph if expand_with_graph is not None else self.enable_graph_expansion

        all_memories = []
        all_scores = []
        sources = {}
        graph_context = {}

        # 1. HDC similarity search
        hdc_results = self.hdc.recall(query, top_k=top_k * 2, tier_filter=tier_filter)
        for memory, score in hdc_results:
            all_memories.append(memory)
            all_scores.append(score)
            sources[memory.id] = "hdc"

        # 2. Graph expansion (if enabled)
        if expand_with_graph and self.enable_falkor:
            try:
                # Extract entities from query
                query_entities = self.falkor.extractor.extract(query)
                entity_names = [e.name for e in query_entities]

                if entity_names:
                    # Get graph context
                    graph_context = self.falkor.graph_rag_query(
                        entity_names,
                        max_hops=2
                    )

                    # Fetch memories linked through graph
                    graph_memory_ids = graph_context.get("memory_ids", [])

                    # Get memories from PostgreSQL that we don't already have
                    existing_ids = {m.id for m in all_memories}
                    new_ids = [mid for mid in graph_memory_ids if mid not in existing_ids]

                    if new_ids and self.enable_postgres:
                        graph_memories = self.postgres.get_batch(new_ids[:top_k])
                        for mem in graph_memories:
                            all_memories.append(mem)
                            all_scores.append(0.5)  # Default score for graph-sourced
                            sources[mem.id] = "graph"

            except Exception as e:
                print(f"[IntegratedMemory] Graph expansion failed: {e}")

        # 3. Include session memories (highest priority)
        if include_session:
            for session_mem in reversed(self.session_memories):
                if session_mem.id not in sources:
                    # Calculate quick similarity
                    if session_mem.embedding is not None:
                        query_vec = self.hdc._text_to_hv(query)
                        score = self.hdc._similarity(query_vec, session_mem.embedding)
                        if score > 0.3:  # Threshold for relevance
                            all_memories.insert(0, session_mem)
                            all_scores.insert(0, score * 1.2)  # Boost session
                            sources[session_mem.id] = "session"

        # 4. Sort by score and limit
        if all_memories:
            paired = list(zip(all_memories, all_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            all_memories, all_scores = zip(*paired[:top_k])
            all_memories = list(all_memories)
            all_scores = list(all_scores)

        return RecallResult(
            memories=all_memories,
            scores=all_scores,
            graph_context=graph_context,
            sources=sources
        )

    def recall_by_entity(
        self,
        entity_name: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Recall all memories mentioning an entity.

        Args:
            entity_name: Entity to search for
            limit: Max results

        Returns:
            List of memories
        """
        if self.enable_falkor and self.enable_postgres:
            # Get memory IDs from graph
            context = self.falkor.get_entity_context(entity_name)
            memory_ids = context.get("memory_ids", [])[:limit]

            if memory_ids:
                return self.postgres.get_batch(memory_ids)

        # Fallback: search in HDC
        return [m for m, s in self.hdc.recall(entity_name, top_k=limit)]

    def get_context_for_prompt(
        self,
        query: str,
        max_memories: int = 5,
        max_tokens: int = 1000,
    ) -> str:
        """
        Get formatted memory context for inclusion in LLM prompt.

        Args:
            query: Current user query
            max_memories: Maximum memories to include
            max_tokens: Approximate token budget (4 chars ≈ 1 token)

        Returns:
            Formatted context string
        """
        result = self.recall(query, top_k=max_memories)

        if not result.memories:
            return ""

        sections = ["## Relevant Memories"]
        char_count = 0
        char_limit = max_tokens * 4

        for memory, score in zip(result.memories, result.scores):
            source = result.sources.get(memory.id, "unknown")
            entry = f"- [{source}] {memory.content}"

            if char_count + len(entry) > char_limit:
                break

            sections.append(entry)
            char_count += len(entry)

        # Add graph context summary if available
        if result.graph_context.get("relationships"):
            sections.append("\n## Related Context")
            for rel in result.graph_context["relationships"][:3]:
                sections.append(f"- {rel.get('entity')} ({rel.get('type')})")

        return "\n".join(sections)

    def consolidate(
        self,
        engine: ConsolidationEngine = None,
        promote_threshold: float = 0.7,
        decay_threshold: float = 0.3,
    ) -> Dict[str, int]:
        """
        Run memory consolidation cycle.

        Promotes important memories, decays less important ones.

        Args:
            engine: Consolidation engine (quantum or classical)
            promote_threshold: Importance threshold for promotion
            decay_threshold: Importance threshold for decay

        Returns:
            Stats dictionary
        """
        stats = {"promoted": 0, "decayed": 0, "deleted": 0}

        if not self.enable_postgres:
            print("[IntegratedMemory] Consolidation requires PostgreSQL")
            return stats

        # Get daily memories
        daily_memories = self.postgres.query(tier=MemoryTier.DAILY, limit=100)

        for memory in daily_memories:
            if memory.importance >= promote_threshold:
                # Promote to long-term
                self.postgres.update_tier(memory.id, MemoryTier.LONGTERM)
                memory.tier = MemoryTier.LONGTERM
                self.hdc.store(memory)  # Re-index with new tier
                stats["promoted"] += 1

            elif memory.importance < decay_threshold:
                # Decay importance
                new_importance = memory.importance * 0.5
                if new_importance < 0.1:
                    # Delete very low importance
                    self.postgres.delete(memory.id)
                    self.hdc.forget(memory.id)
                    stats["deleted"] += 1
                else:
                    self.postgres.update_importance(memory.id, new_importance)
                    stats["decayed"] += 1

        # Clear session memories (they should be promoted or discarded)
        session_count = len(self.session_memories)
        self.session_memories = []
        stats["session_cleared"] = session_count

        return stats

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Manually add a relationship to the graph.

        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Relationship type (e.g., CAUSES, RELATES_TO)
            properties: Optional properties

        Returns:
            Success boolean
        """
        if not self.enable_falkor:
            return False

        return self.falkor.add_relationship(source, target, relation_type, properties)

    def get_session_context(self) -> List[Dict[str, str]]:
        """Get recent session memories as conversation context."""
        return [
            {"role": "memory", "content": m.content}
            for m in self.session_memories[-10:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all backends."""
        stats = {
            "hdc": {
                "dimensions": self.hdc.dim,
                "memories_indexed": len(self.hdc.memories),
            },
            "session": {
                "count": len(self.session_memories),
                "max": self.max_session_memories,
            }
        }

        if self.enable_postgres:
            stats["postgres"] = self.postgres.get_stats()

        if self.enable_falkor:
            stats["falkor"] = self.falkor.get_stats()

        return stats

    def close(self):
        """Close all connections."""
        if self.postgres:
            self.postgres.close()
        if self.falkor:
            self.falkor.close()
        print("[IntegratedMemory] All connections closed")


# Factory function for common configurations
def create_memory_system(
    mode: str = "full",
    **kwargs
) -> IntegratedMemory:
    """
    Create memory system with preset configuration.

    Args:
        mode: Configuration mode
            - "full": HDC + PostgreSQL + FalkorDB
            - "local": HDC + PostgreSQL only
            - "minimal": HDC only (in-memory)

    Returns:
        Configured IntegratedMemory instance
    """
    if mode == "minimal":
        return IntegratedMemory(
            enable_postgres=False,
            enable_falkor=False,
            **kwargs
        )
    elif mode == "local":
        return IntegratedMemory(
            enable_falkor=False,
            **kwargs
        )
    else:  # full
        return IntegratedMemory(**kwargs)
