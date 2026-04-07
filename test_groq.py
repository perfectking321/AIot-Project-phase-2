"""
Test Groq API integration.
Run: python test_groq.py
"""

from config import config
from brain.llm import get_llm_client, GroqClient

def test_groq():
    print(f"Current provider: {config.llm.provider}")
    print(f"Current model: {config.llm.model_name}")
    print("-" * 50)

    # Get the configured client
    client = get_llm_client()
    print(f"Client type: {type(client).__name__}")

    # Check availability
    print(f"\nChecking availability...")
    if client.is_available():
        print("[OK] Groq API is available!")
    else:
        print("[X] Groq API not available. Check API key.")
        return

    # Test a simple planning prompt
    print("\n" + "=" * 50)
    print("Testing planning prompt...")
    print("=" * 50)

    test_prompt = """You are a Windows automation assistant. Convert this voice command into executable steps.

User command: "Open YouTube and search for funny cat videos"

Respond with ONLY a JSON array:
[
  {"step": 1, "action": "Description", "tool": "tool_name", "params": {"key": "value"}}
]

Available tools: open_application, click_text, type_text, press_key, hotkey, wait"""

    print("\nSending request to Groq...")
    response = client.generate(test_prompt)

    print("\n--- Response ---")
    print(response.content)
    print("\n--- Stats ---")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.eval_count}")

    # Verify it's valid JSON
    import json
    try:
        content = response.content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            steps = json.loads(content[start:end])
            print(f"\n[OK] Parsed {len(steps)} steps successfully!")
            for step in steps:
                print(f"  Step {step.get('step')}: {step.get('tool')} - {step.get('action')}")
    except json.JSONDecodeError as e:
        print(f"\n[X] JSON parse error: {e}")


if __name__ == "__main__":
    test_groq()
