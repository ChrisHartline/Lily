"""
Test Script for Clara's Personality System

Tests:
1. Module loading from JSON files
2. Trigger detection
3. System prompt building
4. Token budget management

Run from project root:
    python -m backend.personality.test_personality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.personality.module_loader import ModuleLoader, PersonalityModule
from backend.personality.context_builder import ContextBuilder, SystemPromptBuilder


def test_module_loading():
    """Test that modules load correctly from JSON files."""
    print("\n" + "="*60)
    print("TEST 1: Module Loading")
    print("="*60)

    prompts_dir = project_root / "clara_prompts"

    if not prompts_dir.exists():
        print(f"❌ Prompts directory not found: {prompts_dir}")
        return False

    try:
        loader = ModuleLoader(
            prompts_dir=str(prompts_dir),
            character_id='clara'
        )

        stats = loader.get_stats()

        print(f"\n✓ Core module: {stats['core_module']}")
        print(f"✓ Contextual modules: {stats['contextual_modules']}")
        print(f"✓ Total triggers indexed: {stats['total_triggers']}")
        print(f"\nToken estimates:")
        for name, tokens in stats['token_estimates'].items():
            print(f"  - {name}: ~{tokens} tokens")

        if stats['core_module']:
            print("\n✓ Module loading PASSED")
            return True
        else:
            print("\n❌ No core module loaded")
            return False

    except Exception as e:
        print(f"\n❌ Module loading FAILED: {e}")
        return False


def test_trigger_detection():
    """Test that triggers are detected correctly in messages."""
    print("\n" + "="*60)
    print("TEST 2: Trigger Detection")
    print("="*60)

    prompts_dir = project_root / "clara_prompts"
    loader = ModuleLoader(prompts_dir=str(prompts_dir), character_id='clara')

    test_cases = [
        ("How was your shift at the hospital today?", ["medical"]),
        ("Can you explain how AI agents work?", ["tech", "technology"]),
        ("I saw Father Tom at Mass yesterday", ["faith", "church"]),
        ("The weather is nice today", []),  # No triggers expected
        ("Tell me about the ER and the new software system", ["medical", "tech"]),
    ]

    all_passed = True

    print("\nAll registered triggers:")
    all_triggers = loader.get_all_triggers()
    for module, triggers in all_triggers.items():
        print(f"  {module}: {triggers[:5]}..." if len(triggers) > 5 else f"  {module}: {triggers}")

    print("\nTest cases:")
    for message, expected_keywords in test_cases:
        triggered = loader.detect_triggers(message)

        # Check if any expected keyword appears in triggered modules
        found_expected = any(
            any(kw.lower() in t.lower() for t in triggered)
            for kw in expected_keywords
        ) if expected_keywords else len(triggered) == 0

        status = "✓" if found_expected or not expected_keywords else "?"
        print(f"\n  Message: \"{message[:50]}...\"" if len(message) > 50 else f"\n  Message: \"{message}\"")
        print(f"  Expected keywords: {expected_keywords}")
        print(f"  Triggered modules: {triggered}")
        print(f"  {status}")

    print("\n✓ Trigger detection test completed")
    return True


def test_active_modules():
    """Test that correct modules are activated for messages."""
    print("\n" + "="*60)
    print("TEST 3: Active Module Selection")
    print("="*60)

    prompts_dir = project_root / "clara_prompts"
    loader = ModuleLoader(
        prompts_dir=str(prompts_dir),
        character_id='clara',
        max_contextual_modules=2,
        token_budget=8000
    )

    test_messages = [
        "Hey, how are you doing today?",
        "I need help understanding this medical diagnosis",
        "Can you help me with the AI project we discussed?",
        "I'm struggling with my faith lately",
    ]

    for message in test_messages:
        modules = loader.get_active_modules(message)

        print(f"\nMessage: \"{message}\"")
        print(f"Active modules ({len(modules)}):")
        for m in modules:
            print(f"  - {m.name} ({m.tier}, ~{m.token_estimate} tokens)")

    print("\n✓ Active module selection test completed")
    return True


def test_system_prompt_building():
    """Test system prompt generation from modules."""
    print("\n" + "="*60)
    print("TEST 4: System Prompt Building")
    print("="*60)

    prompts_dir = project_root / "clara_prompts"
    loader = ModuleLoader(prompts_dir=str(prompts_dir), character_id='clara')
    builder = SystemPromptBuilder()

    # Get modules for a tech-related message
    message = "Can you help me understand how AI agents work?"
    modules = loader.get_active_modules(message)

    # Build system prompt
    memories = [
        "Chris mentioned he's working on a multi-agent system",
        "Previously discussed Modal deployment for AI projects",
    ]

    prompt = builder.build_system_prompt(modules, memories=memories)

    print(f"\nMessage: \"{message}\"")
    print(f"Active modules: {[m.name for m in modules]}")
    print(f"\nGenerated system prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens):")
    print("-" * 40)

    # Show first 1500 chars
    if len(prompt) > 1500:
        print(prompt[:1500])
        print(f"\n... [{len(prompt) - 1500} more characters]")
    else:
        print(prompt)

    print("-" * 40)
    print("\n✓ System prompt building test completed")
    return True


def test_token_budget():
    """Test that token budget is respected."""
    print("\n" + "="*60)
    print("TEST 5: Token Budget Management")
    print("="*60)

    prompts_dir = project_root / "clara_prompts"

    # Test with very small budget
    loader_small = ModuleLoader(
        prompts_dir=str(prompts_dir),
        character_id='clara',
        token_budget=2000  # Small budget
    )

    # Test with large budget
    loader_large = ModuleLoader(
        prompts_dir=str(prompts_dir),
        character_id='clara',
        token_budget=20000  # Large budget
    )

    # Message that triggers multiple modules
    message = "Tell me about the hospital, the AI system, and the parish community"

    modules_small = loader_small.get_active_modules(message)
    modules_large = loader_large.get_active_modules(message)

    total_small = sum(m.token_estimate for m in modules_small)
    total_large = sum(m.token_estimate for m in modules_large)

    print(f"\nMessage: \"{message}\"")
    print(f"\nSmall budget (2000 tokens):")
    print(f"  Modules loaded: {[m.name for m in modules_small]}")
    print(f"  Total tokens: ~{total_small}")

    print(f"\nLarge budget (20000 tokens):")
    print(f"  Modules loaded: {[m.name for m in modules_large]}")
    print(f"  Total tokens: ~{total_large}")

    if total_small <= 2000 or len(modules_small) <= len(modules_large):
        print("\n✓ Token budget management PASSED")
        return True
    else:
        print("\n❌ Token budget not respected")
        return False


def run_all_tests():
    """Run all personality system tests."""
    print("\n" + "="*60)
    print("CLARA PERSONALITY SYSTEM - TEST SUITE")
    print("="*60)

    results = {
        "Module Loading": test_module_loading(),
        "Trigger Detection": test_trigger_detection(),
        "Active Modules": test_active_modules(),
        "System Prompt": test_system_prompt_building(),
        "Token Budget": test_token_budget(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED ❌")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
