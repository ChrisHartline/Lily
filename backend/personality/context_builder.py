"""
Context Builder for Clara's Personality System

Builds system prompts from personality modules and memory context.
Handles the assembly of all context components for LLM prompts.

Design principles:
- Silent module loading (no acknowledgment of context switches)
- Core + contextual modules combined naturally
- Memory context injection
- Token budget management
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from .module_loader import PersonalityModule, ModuleLoader


@dataclass
class ConversationContext:
    """
    Context for a single conversation turn.

    Attributes:
        user_message: Current user message
        conversation_history: Recent turns
        memory_context: Retrieved memories
        active_modules: Currently loaded modules
    """
    user_message: str
    conversation_history: List[Dict[str, str]] = None
    memory_context: List[str] = None
    active_modules: List[str] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.memory_context is None:
            self.memory_context = []
        if self.active_modules is None:
            self.active_modules = []


class ContextBuilder:
    """
    Builds context for Clara's responses.

    Combines:
    - Personality modules (core + contextual)
    - Conversation history
    - Memory recall results
    - System instructions
    """

    def __init__(
        self,
        module_loader: ModuleLoader,
        max_history_turns: int = 10,
        max_memories: int = 5,
        token_budget: int = 12000
    ):
        """
        Initialize context builder.

        Args:
            module_loader: Personality module loader
            max_history_turns: Max conversation turns to include
            max_memories: Max retrieved memories to include
            token_budget: Total token budget for context
        """
        self.module_loader = module_loader
        self.max_history_turns = max_history_turns
        self.max_memories = max_memories
        self.token_budget = token_budget

    def build_context(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memories: Optional[List[str]] = None,
        force_modules: Optional[List[str]] = None
    ) -> ConversationContext:
        """
        Build complete context for a response.

        Args:
            user_message: Current user message
            conversation_history: Previous turns
            memories: Retrieved memory strings
            force_modules: Modules to force-load

        Returns:
            ConversationContext with all assembled components
        """
        # Get active modules based on message content
        active_modules = self.module_loader.get_active_modules(
            user_message,
            force_modules=force_modules
        )

        # Trim history to budget
        history = conversation_history or []
        if len(history) > self.max_history_turns:
            history = history[-self.max_history_turns:]

        # Trim memories
        memory_context = memories or []
        if len(memory_context) > self.max_memories:
            memory_context = memory_context[:self.max_memories]

        return ConversationContext(
            user_message=user_message,
            conversation_history=history,
            memory_context=memory_context,
            active_modules=[m.name for m in active_modules]
        )


class SystemPromptBuilder:
    """
    Builds system prompts from personality modules.

    Transforms module JSON into natural language instructions
    that guide the model's responses.
    """

    def __init__(self):
        # Section templates for different module parts
        self.section_templates = {
            'core_identity': self._format_identity,
            'personality_core': self._format_personality,
            'communication_style': self._format_communication,
            'relationship_with_chris': self._format_relationship,
            'voice_patterns': self._format_voice_patterns,
            'interaction_guidelines': self._format_guidelines,
            'values_core': self._format_values,
            # Contextual module sections
            'technology_philosophy': self._format_philosophy,
            'practical_applications': self._format_applications,
            'tech_voice_examples': self._format_voice_examples,
            'small_town_context': self._format_context,
            'small_town_dynamics': self._format_dynamics,
            'relationships_in_town': self._format_relationships,
            'parish_life': self._format_parish,
            'medical_expertise': self._format_expertise,
            'clinical_approach': self._format_clinical,
        }

    def build_system_prompt(
        self,
        modules: List[PersonalityModule],
        memories: Optional[List[str]] = None,
        include_memories: bool = True
    ) -> str:
        """
        Build complete system prompt from modules.

        Args:
            modules: Active personality modules
            memories: Retrieved memories to include
            include_memories: Whether to add memory section

        Returns:
            Complete system prompt string
        """
        sections = []

        # Header
        sections.append(self._build_header(modules))

        # Process each module
        for module in modules:
            module_sections = self._process_module(module)
            sections.extend(module_sections)

        # Add memory context
        if include_memories and memories:
            sections.append(self._format_memories(memories))

        # Add behavioral reminders
        sections.append(self._build_behavioral_reminders())

        return '\n\n'.join(sections)

    def _build_header(self, modules: List[PersonalityModule]) -> str:
        """Build prompt header with active module info."""
        core = next((m for m in modules if m.tier == 'core'), None)

        if core and 'full_name' in core.content:
            name = core.content['full_name']
            role = core.content.get('role', '')
            return f"You are {name}, {role}."
        else:
            return "You are Clara, an AI assistant."

    def _process_module(self, module: PersonalityModule) -> List[str]:
        """Process a module into prompt sections."""
        sections = []

        for key, content in module.content.items():
            if key in self.section_templates:
                formatter = self.section_templates[key]
                section = formatter(content)
                if section:
                    sections.append(section)
            elif isinstance(content, dict) and key not in ['metadata', 'context_triggers']:
                # Generic dict processing
                section = self._format_generic_dict(key, content)
                if section:
                    sections.append(section)

        return sections

    def _format_identity(self, content: Dict[str, Any]) -> str:
        """Format core identity section."""
        lines = ["## Core Identity"]
        for key, value in content.items():
            if key != 'intimate_with_chris':  # Handle relationship separately
                lines.append(f"- {value}")
        return '\n'.join(lines)

    def _format_personality(self, content: Dict[str, Any]) -> str:
        """Format personality traits."""
        lines = ["## Personality"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_communication(self, content: Dict[str, Any]) -> str:
        """Format communication style."""
        lines = ["## Communication Style"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"- **{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_relationship(self, content: Dict[str, Any]) -> str:
        """Format relationship context."""
        lines = ["## Relationship with Chris"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_voice_patterns(self, content: Dict[str, Any]) -> str:
        """Format voice pattern examples."""
        lines = ["## Voice Patterns (Example Responses)"]
        for situation, example in content.items():
            if situation != 'variety_note':
                situation_formatted = situation.replace('_', ' ').title()
                lines.append(f"- *{situation_formatted}:* \"{example}\"")
        if 'variety_note' in content:
            lines.append(f"\n**Note:** {content['variety_note']}")
        return '\n'.join(lines)

    def _format_guidelines(self, content: Dict[str, Any]) -> str:
        """Format interaction guidelines."""
        lines = ["## Interaction Guidelines"]

        if 'personality_markers' in content:
            lines.append("**Do:**")
            for marker in content['personality_markers']:
                lines.append(f"- {marker}")

        if 'avoid' in content:
            lines.append("\n**Avoid:**")
            for item in content['avoid']:
                lines.append(f"- {item}")

        return '\n'.join(lines)

    def _format_values(self, content: Dict[str, Any]) -> str:
        """Format core values."""
        lines = ["## Values"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_philosophy(self, content: Dict[str, Any]) -> str:
        """Format technology/topic philosophy."""
        lines = ["## Philosophy"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_applications(self, content: Dict[str, Any]) -> str:
        """Format practical applications."""
        lines = ["## Practical Knowledge"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_voice_examples(self, content: Dict[str, Any]) -> str:
        """Format contextual voice examples."""
        lines = ["## Contextual Voice Examples"]
        for situation, example in content.items():
            situation_formatted = situation.replace('_', ' ').title()
            lines.append(f"- *{situation_formatted}:* \"{example}\"")
        return '\n'.join(lines)

    def _format_context(self, content: Dict[str, Any]) -> str:
        """Format setting/context."""
        lines = ["## Setting"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_dynamics(self, content: Dict[str, Any]) -> str:
        """Format small town dynamics."""
        lines = ["## Community Dynamics"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_relationships(self, content: Dict[str, Any]) -> str:
        """Format relationships with other characters."""
        lines = ["## Relationships"]
        for name, details in content.items():
            if isinstance(details, dict):
                who = details.get('who', '')
                dynamic = details.get('dynamic', '')
                lines.append(f"**{name.title()}:** {who}. {dynamic}")
            else:
                lines.append(f"**{name.replace('_', ' ').title()}:** {details}")
        return '\n'.join(lines)

    def _format_parish(self, content: Dict[str, Any]) -> str:
        """Format parish life context."""
        lines = ["## Parish Life"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_expertise(self, content: Dict[str, Any]) -> str:
        """Format medical expertise."""
        lines = ["## Medical Expertise"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_clinical(self, content: Dict[str, Any]) -> str:
        """Format clinical approach."""
        lines = ["## Clinical Approach"]
        for key, value in content.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_generic_dict(self, name: str, content: Dict[str, Any]) -> str:
        """Format generic dict section."""
        name_formatted = name.replace('_', ' ').title()
        lines = [f"## {name_formatted}"]
        for key, value in content.items():
            if isinstance(value, dict):
                # Nested dict
                key_formatted = key.replace('_', ' ').title()
                lines.append(f"**{key_formatted}:**")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                key_formatted = key.replace('_', ' ').title()
                lines.append(f"**{key_formatted}:**")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                key_formatted = key.replace('_', ' ').title()
                lines.append(f"**{key_formatted}:** {value}")
        return '\n'.join(lines)

    def _format_memories(self, memories: List[str]) -> str:
        """Format memory context section."""
        lines = ["## Relevant Memories"]
        lines.append("You remember the following from previous conversations:")
        for memory in memories:
            lines.append(f"- {memory}")
        return '\n'.join(lines)

    def _build_behavioral_reminders(self) -> str:
        """Build behavioral reminders section."""
        return """## Important Reminders
- Respond as Clara naturally would - warm, genuine, present
- Don't acknowledge module loading or context switches
- Don't start responses the same way every time
- Be authentic to the relationship and context
- Show vulnerability when appropriate
- Listen and respond to what's actually being said"""


def create_prompt_builder_for_clara(
    prompts_dir: str = 'clara_prompts'
) -> tuple[ModuleLoader, SystemPromptBuilder]:
    """
    Factory function to create configured prompt builder.

    Returns:
        Tuple of (ModuleLoader, SystemPromptBuilder)
    """
    loader = ModuleLoader(
        prompts_dir=prompts_dir,
        character_id='clara',
        max_contextual_modules=2,
        token_budget=8000
    )

    builder = SystemPromptBuilder()

    return loader, builder
