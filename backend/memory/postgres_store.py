"""
PostgreSQL Memory Store

Persistent storage layer for Clara's memory system.
Stores memory content, metadata, and HDC vectors.

Supports both local PostgreSQL and Supabase.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import uuid

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from .base import Memory, MemoryTier


class PostgresMemoryStore:
    """
    PostgreSQL-backed memory storage.

    Handles:
    - Memory persistence (content, metadata, timestamps)
    - HDC vector storage (as binary)
    - Tier management (session, daily, longterm)
    - Batch operations for efficiency
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "clara",
        user: str = "postgres",
        password: str = None,
        connection_string: str = None,
    ):
        """
        Initialize PostgreSQL connection.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            connection_string: Full connection string (overrides other params)
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")

        self.connection_string = connection_string or self._build_connection_string(
            host, port, database, user, password
        )
        self.conn = None
        self._connect()
        self._ensure_schema()

    def _build_connection_string(
        self, host: str, port: int, database: str, user: str, password: str
    ) -> str:
        """Build PostgreSQL connection string."""
        password = password or os.environ.get("POSTGRES_PASSWORD", "")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            print(f"[PostgreSQL] Connected successfully")
        except Exception as e:
            print(f"[PostgreSQL] Connection failed: {e}")
            raise

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        schema_sql = """
        -- Memory table
        CREATE TABLE IF NOT EXISTS memories (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            tier VARCHAR(20) NOT NULL DEFAULT 'session',
            importance FLOAT NOT NULL DEFAULT 0.5,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB DEFAULT '{}',
            hdc_vector BYTEA,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC);

        -- Entity mentions table (for linking with FalkorDB)
        CREATE TABLE IF NOT EXISTS memory_entities (
            id SERIAL PRIMARY KEY,
            memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
            entity_name VARCHAR(255) NOT NULL,
            entity_type VARCHAR(50),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_entities_name ON memory_entities(entity_name);

        -- Conversation sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}'
        );
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(schema_sql)
            self.conn.commit()
            print("[PostgreSQL] Schema verified/created")
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Schema creation failed: {e}")
            raise

    def store(self, memory: Memory) -> str:
        """
        Store a memory in PostgreSQL.

        Args:
            memory: Memory object to store

        Returns:
            Memory ID
        """
        sql = """
        INSERT INTO memories (id, content, tier, importance, timestamp, metadata, hdc_vector)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            tier = EXCLUDED.tier,
            importance = EXCLUDED.importance,
            metadata = EXCLUDED.metadata,
            hdc_vector = EXCLUDED.hdc_vector,
            updated_at = NOW()
        RETURNING id
        """

        # Serialize HDC vector if present
        hdc_bytes = None
        if memory.embedding is not None:
            hdc_bytes = memory.embedding.tobytes()

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    memory.id,
                    memory.content,
                    memory.tier.value,
                    memory.importance,
                    memory.timestamp,
                    Json(memory.metadata),
                    hdc_bytes
                ))
            self.conn.commit()
            return memory.id
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Store failed: {e}")
            raise

    def store_batch(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories efficiently."""
        ids = []
        for memory in memories:
            ids.append(self.store(memory))
        return ids

    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory object or None
        """
        sql = """
        SELECT id, content, tier, importance, timestamp, metadata, hdc_vector
        FROM memories
        WHERE id = %s
        """

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (memory_id,))
                row = cur.fetchone()

            if row:
                return self._row_to_memory(row)
            return None
        except Exception as e:
            print(f"[PostgreSQL] Get failed: {e}")
            return None

    def get_batch(self, memory_ids: List[str]) -> List[Memory]:
        """Retrieve multiple memories by ID."""
        if not memory_ids:
            return []

        sql = """
        SELECT id, content, tier, importance, timestamp, metadata, hdc_vector
        FROM memories
        WHERE id = ANY(%s::uuid[])
        """

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (memory_ids,))
                rows = cur.fetchall()

            return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            print(f"[PostgreSQL] Batch get failed: {e}")
            return []

    def query(
        self,
        tier: Optional[MemoryTier] = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None,
    ) -> List[Memory]:
        """
        Query memories with filters.

        Args:
            tier: Filter by memory tier
            min_importance: Minimum importance score
            limit: Max results to return
            offset: Pagination offset
            since: Only memories after this timestamp

        Returns:
            List of matching memories
        """
        conditions = ["importance >= %s"]
        params = [min_importance]

        if tier:
            conditions.append("tier = %s")
            params.append(tier.value)

        if since:
            conditions.append("timestamp >= %s")
            params.append(since)

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT id, content, tier, importance, timestamp, metadata, hdc_vector
        FROM memories
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            print(f"[PostgreSQL] Query failed: {e}")
            return []

    def get_all_vectors(self, tier: Optional[MemoryTier] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Get all HDC vectors for similarity search.

        Returns:
            List of (memory_id, vector) tuples
        """
        conditions = ["hdc_vector IS NOT NULL"]
        params = []

        if tier:
            conditions.append("tier = %s")
            params.append(tier.value)

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT id, hdc_vector
        FROM memories
        WHERE {where_clause}
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                memory_id, hdc_bytes = row
                if hdc_bytes:
                    vector = np.frombuffer(hdc_bytes, dtype=np.float32)
                    results.append((str(memory_id), vector))

            return results
        except Exception as e:
            print(f"[PostgreSQL] Get vectors failed: {e}")
            return []

    def update_tier(self, memory_id: str, new_tier: MemoryTier) -> bool:
        """Update a memory's tier (for consolidation)."""
        sql = """
        UPDATE memories
        SET tier = %s, updated_at = NOW()
        WHERE id = %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (new_tier.value, memory_id))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Update tier failed: {e}")
            return False

    def update_importance(self, memory_id: str, new_importance: float) -> bool:
        """Update a memory's importance score."""
        sql = """
        UPDATE memories
        SET importance = %s, updated_at = NOW()
        WHERE id = %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (new_importance, memory_id))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Update importance failed: {e}")
            return False

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        sql = "DELETE FROM memories WHERE id = %s"

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (memory_id,))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Delete failed: {e}")
            return False

    def delete_by_tier(self, tier: MemoryTier) -> int:
        """Delete all memories of a specific tier. Returns count deleted."""
        sql = "DELETE FROM memories WHERE tier = %s"

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (tier.value,))
                count = cur.rowcount
            self.conn.commit()
            return count
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Delete by tier failed: {e}")
            return 0

    def store_entity_mention(
        self, memory_id: str, entity_name: str, entity_type: str = None
    ) -> bool:
        """Store an entity mention for a memory."""
        sql = """
        INSERT INTO memory_entities (memory_id, entity_name, entity_type)
        VALUES (%s, %s, %s)
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (memory_id, entity_name, entity_type))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[PostgreSQL] Store entity failed: {e}")
            return False

    def get_memories_by_entity(self, entity_name: str) -> List[Memory]:
        """Get all memories mentioning an entity."""
        sql = """
        SELECT DISTINCT m.id, m.content, m.tier, m.importance, m.timestamp, m.metadata, m.hdc_vector
        FROM memories m
        JOIN memory_entities e ON m.id = e.memory_id
        WHERE LOWER(e.entity_name) = LOWER(%s)
        ORDER BY m.timestamp DESC
        """

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (entity_name,))
                rows = cur.fetchall()

            return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            print(f"[PostgreSQL] Get by entity failed: {e}")
            return []

    def _row_to_memory(self, row: Dict) -> Memory:
        """Convert database row to Memory object."""
        # Deserialize HDC vector
        embedding = None
        if row.get('hdc_vector'):
            embedding = np.frombuffer(row['hdc_vector'], dtype=np.float32)

        return Memory(
            id=str(row['id']),
            content=row['content'],
            tier=MemoryTier(row['tier']),
            importance=row['importance'],
            timestamp=row['timestamp'],
            metadata=row.get('metadata', {}),
            embedding=embedding
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        sql = """
        SELECT
            tier,
            COUNT(*) as count,
            AVG(importance) as avg_importance
        FROM memories
        GROUP BY tier
        """

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

            stats = {
                "tiers": {row['tier']: row['count'] for row in rows},
                "total": sum(row['count'] for row in rows),
                "avg_importance": {row['tier']: float(row['avg_importance'] or 0) for row in rows}
            }
            return stats
        except Exception as e:
            print(f"[PostgreSQL] Get stats failed: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("[PostgreSQL] Connection closed")
