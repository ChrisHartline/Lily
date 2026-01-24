"""
Validate Clara Fine-tuning Dataset
===================================

This script validates the JSONL fine-tuning dataset for:
1. Valid JSON format
2. Correct TinyLlama chat template
3. Response relevance to prompt
4. Response quality (not truncated, not gibberish)
5. Clara personality consistency

Usage:
    python validate_dataset.py check_this.jsonl

Output:
    - validated_dataset.jsonl (clean data)
    - flagged_for_review.jsonl (problematic lines)
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Keywords that should trigger related responses
TOPIC_KEYWORDS = {
    "greeting": ["morning", "evening", "hello", "hi ", "hey", "what's up", "how are you"],
    "emotional_negative": ["sad", "stressed", "overwhelmed", "lonely", "down", "rough", "anxious", "grief"],
    "emotional_positive": ["happy", "excited", "great", "promotion", "good mood", "amazing"],
    "romantic": ["love you", "miss you", "kiss", "hug", "beautiful", "anniversary", "date"],
    "question_about_clara": ["your favorite", "about yourself", "you like", "you thinking", "your hobby"],
    "task_request": ["remind", "schedule", "calendar", "timer", "plan", "recipe", "weather", "grocery"],
    "advice": ["advice", "should i", "help me", "what should", "can't decide"],
    "farewell": ["bye", "goodnight", "should go", "thanks", "thank you"],
    "fact": ["fact", "interesting", "tell me something"],
    "translate": ["translate", "french", "spanish", "german"],
}

# Common gibberish/truncated patterns
GIBBERISH_PATTERNS = [
    r"^\w{1,3}\.\s*$",  # Very short like "Hi."
    r"\[placeholder\]",  # Placeholder text
    r"^\w+\.\s+\w+\?\s*$",  # Too terse like "Fall. Cozy. Yours?"
    r"^.{1,15}$",  # Less than 15 chars is likely truncated
]

# Clara's expected voice patterns
CLARA_TERMS = ["love", "darling", "chris", "dear", "sweet", "my", "you", "we"]


def detect_topic(user_message: str) -> str:
    """Detect the topic category of a user message"""
    msg_lower = user_message.lower()
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                return topic
    
    return "general"


def check_response_quality(response: str) -> tuple[bool, str]:
    """Check if a response is valid"""
    issues = []
    
    # Too short
    if len(response) < 10:
        issues.append("too_short")
    
    # Too long (unusual for Clara)
    if len(response) > 300:
        issues.append("too_long")
    
    # Gibberish patterns
    for pattern in GIBBERISH_PATTERNS:
        if re.match(pattern, response):
            issues.append("gibberish_pattern")
            break
    
    # Missing Clara personality markers
    response_lower = response.lower()
    has_clara_voice = any(term in response_lower for term in CLARA_TERMS)
    if not has_clara_voice and len(response) > 30:
        issues.append("missing_clara_voice")
    
    return (len(issues) == 0, ", ".join(issues) if issues else "ok")


def check_topic_match(user_message: str, response: str) -> tuple[bool, str]:
    """Check if response matches the topic"""
    topic = detect_topic(user_message)
    response_lower = response.lower()
    user_lower = user_message.lower()
    
    # Specific checks for common mismatches
    if "translate" in user_lower and "french" not in response_lower and "spanish" not in response_lower:
        if "traffic" in response_lower or "drive" in response_lower:
            return False, "translate_mismatch"
    
    if "pet peeve" in user_lower and "invisibility" in response_lower:
        return False, "topic_mismatch"
    
    if "holiday" in user_lower and "invisibility" in response_lower:
        return False, "topic_mismatch"
    
    if "dream" in user_lower and "vacation" in user_lower:
        if len(response) < 20:
            return False, "truncated_response"
    
    if "appointment" in user_lower and "placeholder" in response_lower:
        return False, "placeholder_found"
    
    if "fact" in user_lower:
        # Facts should be informative, not just a few words
        if len(response.split()) < 5:
            return False, "fact_too_short"
    
    return True, "ok"


def parse_line(line: str) -> dict | None:
    """Parse a JSONL line and extract components"""
    try:
        data = json.loads(line.strip())
        text = data.get("text", "")
        
        # Extract user message
        user_match = re.search(r'<\|user\|>(.*?)</s>', text)
        assistant_match = re.search(r'<\|assistant\|>(.*?)</s>', text)
        
        if user_match and assistant_match:
            return {
                "raw": data,
                "user": user_match.group(1).strip(),
                "assistant": assistant_match.group(1).strip(),
            }
        return None
    except json.JSONDecodeError:
        return None


def validate_dataset(input_path: Path):
    """Main validation function"""
    print(f"Validating: {input_path}")
    print("=" * 60)
    
    valid_lines = []
    flagged_lines = []
    stats = defaultdict(int)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip wrapper tags
        if line.startswith("<DOCUMENT") or line.startswith("</DOCUMENT"):
            stats["skipped_wrapper"] += 1
            continue
        
        # Skip empty lines
        if not line:
            stats["skipped_empty"] += 1
            continue
        
        # Parse JSON
        parsed = parse_line(line)
        if not parsed:
            flagged_lines.append({
                "line": i,
                "issue": "invalid_json",
                "content": line[:100] + "..." if len(line) > 100 else line
            })
            stats["invalid_json"] += 1
            continue
        
        user_msg = parsed["user"]
        response = parsed["assistant"]
        
        # Check response quality
        quality_ok, quality_issue = check_response_quality(response)
        if not quality_ok:
            flagged_lines.append({
                "line": i,
                "issue": quality_issue,
                "user": user_msg,
                "response": response
            })
            stats[f"quality_{quality_issue}"] += 1
            continue
        
        # Check topic match
        match_ok, match_issue = check_topic_match(user_msg, response)
        if not match_ok:
            flagged_lines.append({
                "line": i,
                "issue": match_issue,
                "user": user_msg,
                "response": response
            })
            stats[f"match_{match_issue}"] += 1
            continue
        
        # Line is valid
        valid_lines.append(parsed["raw"])
        stats["valid"] += 1
    
    # Write valid lines
    output_valid = input_path.parent / "validated_dataset.jsonl"
    with open(output_valid, 'w', encoding='utf-8') as f:
        for item in valid_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Write flagged lines
    output_flagged = input_path.parent / "flagged_for_review.jsonl"
    with open(output_flagged, 'w', encoding='utf-8') as f:
        for item in flagged_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total lines processed: {len(lines)}")
    print(f"✅ Valid examples: {stats['valid']}")
    print(f"❌ Flagged for review: {len(flagged_lines)}")
    print()
    
    print("Issue breakdown:")
    for key, count in sorted(stats.items()):
        if key != "valid":
            print(f"  - {key}: {count}")
    
    print()
    print(f"📁 Output files:")
    print(f"  ✅ {output_valid} ({stats['valid']} lines)")
    print(f"  ⚠️  {output_flagged} ({len(flagged_lines)} lines)")
    
    # Show sample of flagged items
    if flagged_lines:
        print(f"\n⚠️  SAMPLE FLAGGED ITEMS (first 10):")
        print("-" * 60)
        for item in flagged_lines[:10]:
            print(f"Line {item['line']}: {item['issue']}")
            if 'user' in item:
                print(f"  User: {item['user'][:50]}...")
                print(f"  Response: {item['response'][:50]}...")
            print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path(__file__).parent / "check_this.jsonl"
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    validate_dataset(input_file)
