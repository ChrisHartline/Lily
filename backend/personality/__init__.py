"""
Clara Personality System

Modular personality loading with:
- Core module (always loaded)
- Contextual modules (loaded by trigger keywords)
- Silent context switching (no acknowledgment)
- System prompt building from active modules
"""

from .module_loader import PersonalityModule, ModuleLoader
from .context_builder import ContextBuilder, SystemPromptBuilder

__all__ = [
    'PersonalityModule',
    'ModuleLoader',
    'ContextBuilder',
    'SystemPromptBuilder',
]
