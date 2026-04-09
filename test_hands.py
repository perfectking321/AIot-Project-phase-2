"""
Test Hands module - Qwen 7B local inference for actions.
Run: python test_hands.py

Prerequisites:
1. Ollama running: ollama serve
2. Model pulled: ollama pull qwen2.5:7b-instruct-q4_K_M
   Or any qwen model: ollama pull qwen2.5:3b
"""
import sys
import time


def test_hands():
    print("=" * 60)
    print("VOXCODE Hands Test (Qwen Local Inference)")
    print("=" * 60)

    # Test 1: Import
    print("\n[1] Testing Hands import...")
    try:
        from agent.hands import Hands, get_hands, QWEN_MODEL
        print(f"    OK - Hands module imported")
        print(f"    Default model: {QWEN_MODEL}")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Check Ollama availability
    print("\n[2] Checking Ollama connection...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            print(f"    OK - Ollama reachable")
            print(f"    Available models: {models}")
        else:
            print(f"    FAIL - Ollama returned status {resp.status_code}")
            print("    Run: ollama serve")
            return False
    except Exception as e:
        print(f"    FAIL - Cannot connect to Ollama: {e}")
        print("    Make sure Ollama is running: ollama serve")
        return False

    # Test 3: Create Hands instance
    print("\n[3] Creating Hands instance...")
    try:
        hands = Hands()
        print(f"    OK - Using model: {hands.model}")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 4: Test inference with simple prompt
    print("\n[4] Testing inference (simple click task)...")
    sample_elements = """[0] button 'Search' at (500, 100) (conf:0.95)
[1] input 'Search Google or type URL' at (640, 80) (conf:0.90)
[2] text 'Gmail' at (1200, 50) (conf:0.85)
[3] text 'Images' at (1260, 50) (conf:0.85)
[4] icon 'Settings' at (1400, 50) (conf:0.75)"""

    subgoal = "click on the search bar"

    t0 = time.time()
    try:
        decision = hands.decide(subgoal, sample_elements)
        t1 = time.time()
        print(f"    Inference time: {(t1-t0)*1000:.0f}ms")
        print(f"    Decision: {decision}")

        # Validate decision
        if decision.get("action") == "click":
            x, y = decision.get("x"), decision.get("y")
            if x and y:
                print(f"    OK - Click at ({x}, {y})")
            else:
                print("    WARN - Click missing coordinates")
        elif decision.get("action") == "wait":
            print("    WARN - Model returned wait (may not have understood)")
        else:
            print(f"    OK - Action: {decision.get('action')}")

    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Test type action
    print("\n[5] Testing inference (type task)...")
    subgoal2 = "type 'hello world' in the search bar"

    t0 = time.time()
    try:
        decision2 = hands.decide(subgoal2, sample_elements)
        t1 = time.time()
        print(f"    Inference time: {(t1-t0)*1000:.0f}ms")
        print(f"    Decision: {decision2}")
    except Exception as e:
        print(f"    FAIL - {e}")

    # Test 6: Test done detection
    print("\n[6] Testing 'done' detection...")
    done_elements = """[0] text 'Search results for: cats' at (400, 200) (conf:0.95)
[1] text 'cats - Google Search' at (200, 50) (conf:0.90)
[2] link 'Cats Wikipedia' at (300, 300) (conf:0.85)"""

    subgoal3 = "task is complete - we already searched for cats"

    t0 = time.time()
    try:
        decision3 = hands.decide(subgoal3, done_elements)
        t1 = time.time()
        print(f"    Inference time: {(t1-t0)*1000:.0f}ms")
        print(f"    Decision: {decision3}")
    except Exception as e:
        print(f"    FAIL - {e}")

    # Test 7: JSON parsing edge cases
    print("\n[7] Testing JSON parsing...")
    from agent.hands import Hands
    h = Hands.__new__(Hands)  # Create without __init__

    test_cases = [
        '{"action":"click","x":100,"y":200}',
        '```json\n{"action":"type","text":"hello"}\n```',
        'Here is the action: {"action":"press","key":"enter"}',
        'I will click on the button.\n{"action":"click","x":50,"y":50}\nDone.',
    ]

    for i, test in enumerate(test_cases):
        result = h._parse_action_json(test)
        if result:
            print(f"    Case {i+1}: OK - {result}")
        else:
            print(f"    Case {i+1}: FAIL - Could not parse")

    print("\n" + "=" * 60)
    print("Hands test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_hands()
    sys.exit(0 if success else 1)
