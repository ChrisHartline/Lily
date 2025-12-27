"""
Clara Memory System

Modular, swappable memory architecture combining:
- HDC (Hyperdimensional Computing) for associative recall
- Quantum circuits (TFQ/Cirq) for consolidation
- PostgreSQL for long-term structured storage
- FalkorDB for relationship graphs

Architecture:
    Session → Daily → Long-term (with quantum consolidation)
"""

from .base import MemoryStore, Memory, MemoryTier, ConsolidationEngine
from .hdc_memory import HDCMemory, ClassicalConsolidation
from .quantum_consolidation import (
    QuantumConsolidation,
    QuantumHDCBridge,
    ClassicalQuantumFallback,
    get_consolidation_engine,
    QUANTUM_AVAILABLE
)
from .postgres_store import PostgresMemoryStore, POSTGRES_AVAILABLE
from .falkor_store import FalkorMemoryStore, Entity, Relationship, FALKOR_AVAILABLE
from .integrated_memory import IntegratedMemory, RecallResult, create_memory_system

__all__ = [
    # Base interfaces
    'MemoryStore',
    'Memory',
    'MemoryTier',
    'ConsolidationEngine',
    # HDC implementation
    'HDCMemory',
    'ClassicalConsolidation',
    # Quantum consolidation
    'QuantumConsolidation',
    'QuantumHDCBridge',
    'ClassicalQuantumFallback',
    'get_consolidation_engine',
    'QUANTUM_AVAILABLE',
    # PostgreSQL store
    'PostgresMemoryStore',
    'POSTGRES_AVAILABLE',
    # FalkorDB store
    'FalkorMemoryStore',
    'Entity',
    'Relationship',
    'FALKOR_AVAILABLE',
    # Integrated memory
    'IntegratedMemory',
    'RecallResult',
    'create_memory_system',
]
