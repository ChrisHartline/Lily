"""
Auto-fix Clara voice in flagged dataset entries
Adds endearments like "love", "darling", "Chris" to make responses warmer
"""

import json
import random
from pathlib import Path

# Clara's endearments to add
ENDEARMENTS = ["love", "darling", "my love", "dear", "sweetheart"]
CHRIS_REFS = ["Chris", "my love"]

# Sentence starters that feel like Clara
CLARA_STARTERS = [
    "Oh, ",
    "Well, ",
    "Hmm, ",
    "",  # Sometimes no starter
]

def add_clara_voice(response: str, user_msg: str) -> str:
    """Add Clara's warm voice to a response"""
    
    # Don't modify if already has voice
    response_lower = response.lower()
    if any(term in response_lower for term in ["love", "darling", "chris", "dear", "sweetheart"]):
        return response
    
    # Choose an endearment
    endearment = random.choice(ENDEARMENTS)
    
    # Strategy based on response type
    if response.endswith("?"):
        # Questions: add endearment at end or beginning
        if random.random() > 0.5:
            # "What do you think, love?"
            response = response[:-1] + f", {endearment}?"
        else:
            # "Love, what do you think?"
            response = f"{endearment.capitalize()}, " + response[0].lower() + response[1:]
    
    elif response.endswith("!"):
        # Exclamations: add at beginning or end
        if random.random() > 0.5:
            response = response[:-1] + f", {endearment}!"
        else:
            response = f"{endearment.capitalize()}, " + response[0].lower() + response[1:]
    
    else:
        # Statements: various positions
        choice = random.random()
        if choice < 0.33:
            # Beginning: "Love, here's what I think..."
            response = f"{endearment.capitalize()}, " + response[0].lower() + response[1:]
        elif choice < 0.66:
            # End: "...that's my thought, love."
            if response.endswith("."):
                response = response[:-1] + f", {endearment}."
            else:
                response = response + f", {endearment}."
        else:
            # After first sentence
            if ". " in response:
                parts = response.split(". ", 1)
                response = f"{parts[0]}, {endearment}. {parts[1]}"
            else:
                response = f"{endearment.capitalize()}, " + response[0].lower() + response[1:]
    
    return response


def fix_dataset():
    base_path = Path(__file__).parent
    flagged_path = base_path / "flagged_for_review.jsonl"
    validated_path = base_path / "validated_dataset.jsonl"
    output_path = base_path / "clara_training_data.jsonl"
    
    fixed_count = 0
    skipped_count = 0
    all_examples = []
    
    # Load validated examples first
    with open(validated_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(all_examples)} validated examples")
    
    # Process flagged examples
    with open(flagged_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            item = json.loads(line.strip())
            issue = item.get("issue", "")
            
            # Only auto-fix missing_clara_voice
            if issue == "missing_clara_voice":
                user_msg = item["user"]
                old_response = item["response"]
                new_response = add_clara_voice(old_response, user_msg)
                
                # Reconstruct the training example
                system_prompt = "You are Clara (nickname Lily), the user's romantic partner and assistant. You are warm, genuine, caring, and supportive, responding naturally like a close companion. Use emojis sparingly, and offer assistance with daily tasks when appropriate."
                
                fixed_example = {
                    "text": f"<|system|>{system_prompt}</s><|user|>{user_msg}</s><|assistant|>{new_response}</s>"
                }
                all_examples.append(fixed_example)
                fixed_count += 1
                
                print(f"Fixed: '{user_msg[:30]}...' → '{new_response[:40]}...'")
            else:
                # Skip other issues (need manual review)
                skipped_count += 1
    
    # Write final dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print()
    print("=" * 60)
    print(f"✅ FINAL DATASET: {output_path}")
    print(f"   Total examples: {len(all_examples)}")
    print(f"   - From validated: {len(all_examples) - fixed_count}")
    print(f"   - Auto-fixed: {fixed_count}")
    print(f"   - Skipped (need manual fix): {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    fix_dataset()
