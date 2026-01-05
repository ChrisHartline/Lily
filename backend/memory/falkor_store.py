"""
FalkorDB Graph Memory Store

Graph-based relationship storage for Clara's memory system.
Stores entities and their relationships for rich context retrieval.

Uses FalkorDB (Redis-compatible graph database).
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

try:
    from falkordb import FalkorDB
    FALKOR_AVAILABLE = True
except ImportError:
    FALKOR_AVAILABLE = False

from .base import Memory


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str  # person, place, topic, emotion, event, etc.
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str  # MENTIONED_WITH, CAUSES, RELATES_TO, etc.
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class EntityExtractor:
    """
    Extracts entities from text using LLM or pattern matching.

    Supports:
    - LLM-based extraction (when model available)
    - Pattern-based fallback (always available)
    """

    # Common patterns for fallback extraction
    PATTERNS = {
        'person': [
            r'\b(?:Chris|Clara|Dr\.\s+\w+|Father\s+\w+)\b',
            r'\b(?:he|she|they|him|her|them)\b',
        ],
        'emotion': [
            r'\b(?:happy|sad|angry|anxious|stressed|overwhelmed|excited|worried|scared|frustrated|tired|exhausted)\b',
        ],
        'topic': [
            r'\b(?:work|project|AI|technology|faith|church|hospital|ER|medicine|health|family)\b',
        ],
        'time': [
            r'\b(?:today|yesterday|tomorrow|last week|next week|this morning|tonight)\b',
        ],
        'place': [
            r'\b(?:hospital|church|home|office|St\.\s+Mary\'s|ER|clinic)\b',
        ],
    }

    def __init__(self, use_llm: bool = True, llm_func: callable = None):
        """
        Initialize entity extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm_func: Function that takes text and returns extracted entities
                      Signature: (text: str) -> List[Dict] with keys: name, type
        """
        self.use_llm = use_llm and llm_func is not None
        self.llm_func = llm_func

    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of Entity objects
        """
        if self.use_llm and self.llm_func:
            try:
                return self._extract_with_llm(text)
            except Exception as e:
                print(f"[EntityExtractor] LLM extraction failed, falling back to patterns: {e}")

        return self._extract_with_patterns(text)

    def _extract_with_llm(self, text: str) -> List[Entity]:
        """Extract entities using LLM."""
        raw_entities = self.llm_func(text)
        return [
            Entity(
                name=e.get('name', ''),
                type=e.get('type', 'unknown'),
                properties=e.get('properties', {})
            )
            for e in raw_entities
            if e.get('name')
        ]

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns (fallback)."""
        entities = []
        seen = set()

        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    normalized = match.lower().strip()
                    if normalized not in seen and len(normalized) > 1:
                        seen.add(normalized)
                        entities.append(Entity(
                            name=match.strip(),
                            type=entity_type
                        ))

        return entities

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities in text.

        Simple heuristic: entities mentioned in same sentence are related.
        """
        relationships = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Find which entities appear in this sentence
            present_entities = [
                e for e in entities
                if e.name.lower() in sentence_lower
            ]

            # Create MENTIONED_WITH relationships
            for i, e1 in enumerate(present_entities):
                for e2 in present_entities[i+1:]:
                    relationships.append(Relationship(
                        source=e1.name,
                        target=e2.name,
                        relation_type="MENTIONED_WITH",
                        properties={"context": sentence.strip()[:100]}
                    ))

        return relationships


class FalkorMemoryStore:
    """
    FalkorDB-backed graph memory storage.

    Handles:
    - Entity storage and relationships
    - Graph traversal for context retrieval
    - Memory-entity linking
    - Graph RAG queries
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "claralilymem",
        password: str = None,
    ):
        """
        Initialize FalkorDB connection.

        Args:
            host: FalkorDB host
            port: FalkorDB port (default 6379)
            graph_name: Name of the graph
            password: Optional password
        """
        if not FALKOR_AVAILABLE:
            raise ImportError("falkordb not installed. Run: pip install falkordb")

        self.host = host
        self.port = port
        self.graph_name = graph_name

        # Connect to FalkorDB - only use password if explicitly provided and non-empty
        connect_args = {"host": host, "port": port}
        if password and str(password).strip():
            connect_args["password"] = password

        self.db = FalkorDB(**connect_args)

        self.graph = self.db.select_graph(graph_name)

        # Entity extractor
        self.extractor = EntityExtractor(use_llm=False)  # Start with patterns

        self._ensure_schema()
        print(f"[FalkorDB] Connected to {host}:{port}, graph: {graph_name}")

    def _ensure_schema(self):
        """Create indexes for common queries."""
        try:
            # Create indexes on entity names and types
            self.graph.query("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            self.graph.query("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            self.graph.query("CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.id)")
            print("[FalkorDB] Schema indexes verified")
        except Exception as e:
            print(f"[FalkorDB] Schema setup note: {e}")

    def set_llm_extractor(self, llm_func: callable):
        """
        Set LLM function for entity extraction.

        Args:
            llm_func: Function with signature (text: str) -> List[Dict]
                      Each dict should have: name, type, (optional) properties
        """
        self.extractor = EntityExtractor(use_llm=True, llm_func=llm_func)
        print("[FalkorDB] LLM entity extractor enabled")

    def store_memory_graph(
        self,
        memory: Memory,
        entities: List[Entity] = None,
        relationships: List[Relationship] = None
    ) -> Dict[str, int]:
        """
        Store a memory and its entities/relationships in the graph.

        Args:
            memory: Memory object
            entities: Pre-extracted entities (or None to extract)
            relationships: Pre-extracted relationships (or None to extract)

        Returns:
            Stats dict with counts
        """
        stats = {"entities": 0, "relationships": 0, "memory_links": 0}

        # Extract entities if not provided
        if entities is None:
            entities = self.extractor.extract(memory.content)

        # Extract relationships if not provided
        if relationships is None:
            relationships = self.extractor.extract_relationships(
                memory.content, entities
            )

        try:
            # Create Memory node
            self.graph.query(
                """
                MERGE (m:Memory {id: $id})
                SET m.timestamp = $timestamp,
                    m.importance = $importance,
                    m.tier = $tier
                """,
                {'id': memory.id, 'timestamp': str(memory.timestamp),
                 'importance': memory.importance, 'tier': memory.tier.value}
            )

            # Create Entity nodes and link to Memory
            for entity in entities:
                self.graph.query(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    WITH e
                    MATCH (m:Memory {id: $memory_id})
                    MERGE (m)-[:MENTIONS]->(e)
                    """,
                    {'name': entity.name, 'type': entity.type,
                     'memory_id': memory.id}
                )
                stats["entities"] += 1
                stats["memory_links"] += 1

            # Create relationships between entities
            for rel in relationships:
                self.graph.query(
                    f"""
                    MATCH (e1:Entity {{name: $source}})
                    MATCH (e2:Entity {{name: $target}})
                    MERGE (e1)-[r:{rel.relation_type}]->(e2)
                    SET r.context = $context
                    """,
                    {'source': rel.source, 'target': rel.target,
                     'context': rel.properties.get('context', '')}
                )
                stats["relationships"] += 1

            return stats

        except Exception as e:
            print(f"[FalkorDB] Store failed: {e}")
            return stats

    def add_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relation_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Add a relationship between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            relation_type: Type of relationship (e.g., CAUSES, RELATES_TO)
            properties: Optional relationship properties

        Returns:
            Success boolean
        """
        try:
            props = properties or {}
            self.graph.query(
                f"""
                MERGE (e1:Entity {{name: $source}})
                MERGE (e2:Entity {{name: $target}})
                MERGE (e1)-[r:{relation_type}]->(e2)
                SET r += $props
                """,
                {'source': source_entity, 'target': target_entity, 'props': props}
            )
            return True
        except Exception as e:
            print(f"[FalkorDB] Add relationship failed: {e}")
            return False

    def get_related_entities(
        self,
        entity_name: str,
        max_hops: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity.

        Args:
            entity_name: Starting entity
            max_hops: Maximum relationship hops
            limit: Max results

        Returns:
            List of related entities with paths
        """
        try:
            result = self.graph.query(
                f"""
                MATCH (e:Entity {{name: $name}})-[*1..{max_hops}]-(related:Entity)
                WHERE e <> related
                RETURN DISTINCT related.name AS name,
                       related.type AS type
                LIMIT $limit
                """,
                {'name': entity_name, 'limit': limit}
            )

            return [
                {'name': row[0], 'type': row[1]}
                for row in result.result_set
            ]
        except Exception as e:
            print(f"[FalkorDB] Get related failed: {e}")
            return []

    def get_entity_context(
        self,
        entity_name: str,
        include_memories: bool = True
    ) -> Dict[str, Any]:
        """
        Get full context for an entity including relationships and memories.

        Args:
            entity_name: Entity to get context for
            include_memories: Whether to include linked memories

        Returns:
            Context dictionary
        """
        context = {
            "entity": entity_name,
            "type": None,
            "related_entities": [],
            "relationships": [],
            "memory_ids": []
        }

        try:
            # Get entity info and direct relationships
            result = self.graph.query(
                """
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (e)-[r]-(other:Entity)
                RETURN e.type AS type,
                       type(r) AS rel_type,
                       other.name AS other_name,
                       other.type AS other_type
                """,
                {'name': entity_name}
            )

            for row in result.result_set:
                if row[0] and not context["type"]:
                    context["type"] = row[0]
                if row[1] and row[2]:
                    context["relationships"].append({
                        "type": row[1],
                        "entity": row[2],
                        "entity_type": row[3]
                    })
                    if row[2] not in context["related_entities"]:
                        context["related_entities"].append(row[2])

            # Get linked memories
            if include_memories:
                mem_result = self.graph.query(
                    """
                    MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $name})
                    RETURN m.id AS memory_id
                    ORDER BY m.timestamp DESC
                    LIMIT 10
                    """,
                    {'name': entity_name}
                )
                context["memory_ids"] = [row[0] for row in mem_result.result_set]

            return context

        except Exception as e:
            print(f"[FalkorDB] Get context failed: {e}")
            return context

    def find_path(
        self,
        source_entity: str,
        target_entity: str,
        max_hops: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Find relationship path between two entities.

        Args:
            source_entity: Starting entity
            target_entity: Target entity
            max_hops: Maximum path length

        Returns:
            Path as list of nodes and relationships
        """
        try:
            result = self.graph.query(
                f"""
                MATCH path = shortestPath(
                    (e1:Entity {{name: $source}})-[*1..{max_hops}]-(e2:Entity {{name: $target}})
                )
                RETURN nodes(path) AS nodes, relationships(path) AS rels
                LIMIT 1
                """,
                {'source': source_entity, 'target': target_entity}
            )

            if result.result_set:
                row = result.result_set[0]
                return {
                    "nodes": row[0],
                    "relationships": row[1],
                    "found": True
                }

            return {"found": False}

        except Exception as e:
            print(f"[FalkorDB] Find path failed: {e}")
            return {"found": False, "error": str(e)}

    def graph_rag_query(
        self,
        query_entities: List[str],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Perform Graph RAG query to get enriched context.

        Given a list of entities from a query, retrieve:
        - Related entities (up to max_hops)
        - Relationship paths between query entities
        - Linked memory IDs

        Args:
            query_entities: Entities mentioned in query
            max_hops: How far to traverse

        Returns:
            Enriched context for RAG
        """
        context = {
            "query_entities": query_entities,
            "expanded_entities": set(),
            "relationships": [],
            "memory_ids": set(),
            "paths": []
        }

        # Expand each query entity
        for entity in query_entities:
            related = self.get_related_entities(entity, max_hops=max_hops)
            for r in related:
                context["expanded_entities"].add(r['name'])

            entity_ctx = self.get_entity_context(entity)
            context["relationships"].extend(entity_ctx["relationships"])
            context["memory_ids"].update(entity_ctx["memory_ids"])

        # Find paths between query entities
        for i, e1 in enumerate(query_entities):
            for e2 in query_entities[i+1:]:
                path = self.find_path(e1, e2, max_hops=max_hops)
                if path.get("found"):
                    context["paths"].append({
                        "from": e1,
                        "to": e2,
                        "path": path
                    })

        # Convert sets to lists for JSON serialization
        context["expanded_entities"] = list(context["expanded_entities"])
        context["memory_ids"] = list(context["memory_ids"])

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            # Count nodes and relationships
            node_result = self.graph.query("MATCH (n) RETURN labels(n) AS label, count(*) AS count")
            rel_result = self.graph.query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")

            return {
                "nodes": {row[0][0] if row[0] else 'unknown': row[1] for row in node_result.result_set},
                "relationships": {row[0]: row[1] for row in rel_result.result_set}
            }
        except Exception as e:
            print(f"[FalkorDB] Get stats failed: {e}")
            return {"error": str(e)}

    def clear_graph(self) -> bool:
        """Clear entire graph (use with caution!)."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("[FalkorDB] Graph cleared")
            return True
        except Exception as e:
            print(f"[FalkorDB] Clear failed: {e}")
            return False

    def close(self):
        """Close connection."""
        if self.db:
            self.db.close()
            print("[FalkorDB] Connection closed")
