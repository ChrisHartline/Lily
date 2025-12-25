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
]
