"""
Personality Module Loader

Loads and manages Clara's personality modules from JSON files.
Handles trigger detection for contextual module loading.

Design principles:
- Core module always loaded (identity, values, relationship)
- Contextual modules loaded silently by keyword triggers
- Fixed personality (not evolving)
- One relationship focus (Chris)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field


@dataclass
class PersonalityModule:
    """
    Represents a single personality module.

    Attributes:
        name: Module identifier
        tier: 'core' or 'contextual'
        always_loaded: Whether to always include in prompts
        triggers: Keywords that trigger loading
        content: Full module content as dict
        token_estimate: Approximate token count
    """
    name: str
    tier: str
    always_loaded: bool
    triggers: List[str]
    content: Dict[str, Any]
    token_estimate: int = 0

    @classmethod
    def from_json(cls, path: Path) -> 'PersonalityModule':
        """Load module from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        name = data.get('module_name', path.stem)
        tier = data.get('tier', 'contextual')
        always_loaded = data.get('always_loaded', False)

        # Get triggers from multiple possible fields
        triggers = data.get('load_triggers', [])
        if not triggers and 'context_triggers' in data:
            # Core module has nested triggers
            for trigger_list in data['context_triggers'].values():
                triggers.extend(trigger_list)

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        content_str = json.dumps(data)
        token_estimate = len(content_str) // 4

        return cls(
            name=name,
            tier=tier,
            always_loaded=always_loaded,
            triggers=triggers,
            content=data,
            token_estimate=token_estimate
        )


class ModuleLoader:
    """
    Loads and manages personality modules.

    Handles:
    - Loading all modules from directory
    - Detecting triggers in user messages
    - Selecting active modules for context
    - Caching for performance
    """

    def __init__(
        self,
        prompts_dir: str = 'clara_prompts',
        character_id: str = 'clara',
        max_contextual_modules: int = 2,
        token_budget: int = 8000
    ):
        """
        Initialize module loader.

        Args:
            prompts_dir: Directory containing JSON modules
            character_id: Character to load (filters by prefix)
            max_contextual_modules: Max contextual modules to load
            token_budget: Max tokens for personality context
        """
        self.prompts_dir = Path(prompts_dir)
        self.character_id = character_id
        self.max_contextual_modules = max_contextual_modules
        self.token_budget = token_budget

        # Module storage
        self.core_module: Optional[PersonalityModule] = None
        self.contextual_modules: Dict[str, PersonalityModule] = {}

        # Trigger index for fast lookup
        self._trigger_index: Dict[str, str] = {}  # trigger -> module_name

        # Currently active modules
        self.active_modules: Set[str] = set()

        # Load modules on init
        self._load_all_modules()

    def _load_all_modules(self):
        """Load all modules from prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        pattern = f"{self.character_id}_*.json"
        for path in self.prompts_dir.glob(pattern):
            try:
                module = PersonalityModule.from_json(path)

                if module.tier == 'core' or module.always_loaded:
                    self.core_module = module
                else:
                    self.contextual_modules[module.name] = module

                    # Build trigger index
                    for trigger in module.triggers:
                        self._trigger_index[trigger.lower()] = module.name

            except Exception as e:
                print(f"Warning: Failed to load module {path}: {e}")

        if self.core_module:
            self.active_modules.add(self.core_module.name)

    def detect_triggers(self, text: str) -> List[str]:
        """
        Detect which modules should be triggered by text.

        Args:
            text: User message to scan

        Returns:
            List of triggered module names
        """
        text_lower = text.lower()
        triggered = set()

        for trigger, module_name in self._trigger_index.items():
            # Word boundary matching for accuracy
            pattern = r'\b' + re.escape(trigger) + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                triggered.add(module_name)

        return list(triggered)

    def get_active_modules(
        self,
        user_message: str,
        force_modules: Optional[List[str]] = None
    ) -> List[PersonalityModule]:
        """
        Get modules that should be active for this message.

        Args:
            user_message: Current user message
            force_modules: Modules to force-load (optional)

        Returns:
            List of active modules (core + triggered contextual)
        """
        modules = []

        # Always include core
        if self.core_module:
            modules.append(self.core_module)

        # Detect triggered modules
        triggered = set(self.detect_triggers(user_message))

        # Add forced modules
        if force_modules:
            triggered.update(force_modules)

        # Add contextual modules (respecting budget)
        token_count = self.core_module.token_estimate if self.core_module else 0
        contextual_added = 0

        for module_name in triggered:
            if module_name in self.contextual_modules:
                module = self.contextual_modules[module_name]

                # Check token budget
                if token_count + module.token_estimate > self.token_budget:
                    continue

                # Check max modules
                if contextual_added >= self.max_contextual_modules:
                    continue

                modules.append(module)
                token_count += module.token_estimate
                contextual_added += 1

        # Update active set
        self.active_modules = {m.name for m in modules}

        return modules

    def get_all_triggers(self) -> Dict[str, List[str]]:
        """Get all triggers organized by module."""
        result = {}
        for module in self.contextual_modules.values():
            result[module.name] = module.triggers
        return result

    def get_module_by_name(self, name: str) -> Optional[PersonalityModule]:
        """Get a specific module by name."""
        if self.core_module and self.core_module.name == name:
            return self.core_module
        return self.contextual_modules.get(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'core_module': self.core_module.name if self.core_module else None,
            'contextual_modules': list(self.contextual_modules.keys()),
            'total_triggers': len(self._trigger_index),
            'active_modules': list(self.active_modules),
            'token_estimates': {
                name: m.token_estimate
                for name, m in self.contextual_modules.items()
            }
        }
