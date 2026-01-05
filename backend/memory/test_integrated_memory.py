"""
Test Script for Integrated Memory System

Tests:
1. HDC-only mode (minimal)
2. HDC + PostgreSQL mode (local)
3. Full mode (HDC + PostgreSQL + FalkorDB)
4. Store and recall operations
5. Graph expansion
6. Entity extraction

Run from project root:
    python -m backend.memory.test_integrated_memory

Prerequisites:
    - PostgreSQL running on localhost:5432 (optional)
    - FalkorDB running on localhost:6379 (optional)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[Config] Loaded .env from {env_path}")
except ImportError:
    print("[Config] python-dotenv not installed, using defaults")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.memory import (
    IntegratedMemory,
    create_memory_system,
    Memory,
    MemoryTier,
    POSTGRES_AVAILABLE,
    FALKOR_AVAILABLE,
)


def test_minimal_mode():
    """Test HDC-only mode (no external dependencies)."""
    print("\n" + "="*60)
    print("TEST 1: Minimal Mode (HDC only)")
    print("="*60)

    try:
        memory = create_memory_system(mode="minimal", hdc_dimensions=1000)

        # Store some memories
        m1 = memory.store(
            "Chris mentioned he's been having headaches lately",
            importance=0.7
        )
        print(f"✓ Stored memory: {m1.id[:8]}...")

        m2 = memory.store(
            "We discussed the AI project deadline next week",
            importance=0.8
        )
        print(f"✓ Stored memory: {m2.id[:8]}...")

        m3 = memory.store(
            "Chris is feeling stressed about work",
            importance=0.6
        )
        print(f"✓ Stored memory: {m3.id[:8]}...")

        # Recall
        result = memory.recall("headaches and stress", top_k=3)

        print(f"\nRecall results for 'headaches and stress':")
        for mem, score in zip(result.memories, result.scores):
            print(f"  [{score:.3f}] {mem.content[:50]}...")

        # Stats
        stats = memory.get_stats()
        print(f"\nStats: {stats['hdc']['memories_indexed']} memories indexed")

        memory.close()
        print("\n✓ Minimal mode test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Minimal mode test FAILED: {e}")
        return False


def test_with_postgres():
    """Test HDC + PostgreSQL mode."""
    print("\n" + "="*60)
    print("TEST 2: Local Mode (HDC + PostgreSQL)")
    print("="*60)

    if not POSTGRES_AVAILABLE:
        print("⚠ PostgreSQL not available (psycopg2 not installed)")
        print("  Install with: pip install psycopg2-binary")
        return None

    try:
        memory = IntegratedMemory(
            hdc_dimensions=1000,
            postgres_host="localhost",
            postgres_port=5432,
            postgres_db="clara",
            postgres_user="postgres",
            enable_falkor=False,
        )

        if not memory.enable_postgres:
            print("⚠ PostgreSQL connection failed, skipping test")
            return None

        # Store memories with different tiers
        m1 = memory.store(
            "Clara had a tough shift at the ER today - three traumas back to back",
            importance=0.8,
            tier=MemoryTier.DAILY
        )
        print(f"✓ Stored DAILY memory: {m1.id[:8]}...")

        m2 = memory.store(
            "Chris's favorite coffee is a cortado with oat milk",
            importance=0.9,
            tier=MemoryTier.LONGTERM
        )
        print(f"✓ Stored LONGTERM memory: {m2.id[:8]}...")

        # Recall with persistence
        result = memory.recall("coffee preferences", top_k=3)

        print(f"\nRecall results for 'coffee preferences':")
        for mem, score in zip(result.memories, result.scores):
            source = result.sources.get(mem.id, 'unknown')
            print(f"  [{score:.3f}] ({source}) {mem.content[:50]}...")

        # Stats
        stats = memory.get_stats()
        print(f"\nPostgreSQL stats: {stats.get('postgres', {})}")

        memory.close()
        print("\n✓ Local mode test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Local mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_mode():
    """Test full mode with all backends."""
    print("\n" + "="*60)
    print("TEST 3: Full Mode (HDC + PostgreSQL + FalkorDB)")
    print("="*60)

    if not POSTGRES_AVAILABLE:
        print("⚠ PostgreSQL not available")
        return None

    if not FALKOR_AVAILABLE:
        print("⚠ FalkorDB not available (redis not installed)")
        print("  Install with: pip install redis")
        return None

    try:
        memory = IntegratedMemory(
            hdc_dimensions=1000,
            postgres_host="localhost",
            postgres_port=5432,
            postgres_db=os.environ.get("POSTGRES_DB", "claralily"),
            postgres_user="postgres",
            postgres_password=os.environ.get("POSTGRES_PASSWORD", ""),
            falkor_host="localhost",
            falkor_port=6379,
            falkor_graph=os.environ.get("FALKOR_GRAPH", "claralilymem"),
            falkor_password=None,  # Explicitly no password
        )

        if not memory.enable_postgres or not memory.enable_falkor:
            print("⚠ One or more backends unavailable, skipping test")
            return None

        # Store memories that should create graph relationships
        m1 = memory.store(
            "Chris is working on an AI project at work",
            importance=0.8,
            tier=MemoryTier.DAILY
        )
        print(f"✓ Stored memory with entities: {m1.id[:8]}...")

        m2 = memory.store(
            "The AI project deadline is causing Chris stress",
            importance=0.7,
            tier=MemoryTier.DAILY
        )
        print(f"✓ Stored memory with entities: {m2.id[:8]}...")

        m3 = memory.store(
            "Chris mentioned headaches when stressed about work",
            importance=0.6,
            tier=MemoryTier.SESSION
        )
        print(f"✓ Stored memory with entities: {m3.id[:8]}...")

        # Recall with graph expansion
        result = memory.recall("Chris feeling stressed", top_k=5, expand_with_graph=True)

        print(f"\nRecall results for 'Chris feeling stressed':")
        for mem, score in zip(result.memories, result.scores):
            source = result.sources.get(mem.id, 'unknown')
            print(f"  [{score:.3f}] ({source}) {mem.content[:50]}...")

        # Check graph context
        if result.graph_context:
            print(f"\nGraph context:")
            print(f"  Expanded entities: {result.graph_context.get('expanded_entities', [])[:5]}")
            print(f"  Relationships: {len(result.graph_context.get('relationships', []))}")
            print(f"  Memory IDs from graph: {len(result.graph_context.get('memory_ids', []))}")

        # Test entity-based recall
        print("\nRecall by entity 'Chris':")
        chris_memories = memory.recall_by_entity("Chris", limit=3)
        for mem in chris_memories:
            print(f"  - {mem.content[:50]}...")

        # Stats
        stats = memory.get_stats()
        print(f"\nFull stats:")
        print(f"  HDC: {stats['hdc']}")
        print(f"  Session: {stats['session']}")
        if 'postgres' in stats:
            print(f"  PostgreSQL: {stats['postgres']}")
        if 'falkor' in stats:
            print(f"  FalkorDB: {stats['falkor']}")

        memory.close()
        print("\n✓ Full mode test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Full mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_for_prompt():
    """Test generating context for LLM prompts."""
    print("\n" + "="*60)
    print("TEST 4: Context Generation for Prompts")
    print("="*60)

    try:
        memory = create_memory_system(mode="minimal", hdc_dimensions=1000)

        # Store test memories
        memory.store("Chris loves hiking in the mountains on weekends", importance=0.7)
        memory.store("Clara and Chris met at St. Mary's church", importance=0.9)
        memory.store("Chris has been stressed about the AI project deadline", importance=0.8)
        memory.store("We talked about faith and doubt last Sunday", importance=0.6)

        # Generate context
        context = memory.get_context_for_prompt(
            "How are you handling stress?",
            max_memories=3,
            max_tokens=500
        )

        print("Generated context for prompt:")
        print("-" * 40)
        print(context)
        print("-" * 40)

        memory.close()
        print("\n✓ Context generation test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Context generation test FAILED: {e}")
        return False


def test_session_memory():
    """Test session memory buffer."""
    print("\n" + "="*60)
    print("TEST 5: Session Memory Buffer")
    print("="*60)

    try:
        memory = create_memory_system(mode="minimal", hdc_dimensions=1000)

        # Store session memories
        for i in range(5):
            memory.store(
                f"Turn {i+1}: User said something interesting about topic {i}",
                tier=MemoryTier.SESSION
            )

        print(f"Session buffer size: {len(memory.session_memories)}")

        # Get session context
        session_ctx = memory.get_session_context()
        print(f"Session context entries: {len(session_ctx)}")

        for entry in session_ctx[-3:]:
            print(f"  - {entry['content'][:50]}...")

        memory.close()
        print("\n✓ Session memory test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Session memory test FAILED: {e}")
        return False


def run_all_tests():
    """Run all integrated memory tests."""
    print("\n" + "="*60)
    print("INTEGRATED MEMORY SYSTEM - TEST SUITE")
    print("="*60)

    print(f"\nDependencies:")
    print(f"  PostgreSQL (psycopg2): {'✓ Available' if POSTGRES_AVAILABLE else '✗ Not installed'}")
    print(f"  FalkorDB (redis): {'✓ Available' if FALKOR_AVAILABLE else '✗ Not installed'}")

    results = {
        "Minimal Mode (HDC)": test_minimal_mode(),
        "Session Memory": test_session_memory(),
        "Context Generation": test_context_for_prompt(),
        "Local Mode (PostgreSQL)": test_with_postgres(),
        "Full Mode (Graph)": test_full_mode(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {test_name}: {status}")

    passed_count = sum(1 for p in results.values() if p is True)
    skipped_count = sum(1 for p in results.values() if p is None)
    failed_count = sum(1 for p in results.values() if p is False)

    print(f"\nResults: {passed_count} passed, {skipped_count} skipped, {failed_count} failed")
    print("="*60 + "\n")

    return failed_count == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
