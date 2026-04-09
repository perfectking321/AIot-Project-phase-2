"""
Test plan_subgoals method in brain/llm.py
Run: python test_plan_subgoals.py
"""
import sys


def test_plan_subgoals():
    print("=" * 60)
    print("VOXCODE plan_subgoals Test (Groq)")
    print("=" * 60)

    # Test 1: Import
    print("\n[1] Testing import...")
    try:
        from brain.llm import GroqClient, get_llm_client
        from config import config
        print(f"    OK - Imported")
        print(f"    Provider: {config.llm.provider}")
        print(f"    Model: {config.llm.groq_model}")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Create Groq client
    print("\n[2] Creating Groq client...")
    try:
        client = GroqClient()
        print(f"    OK - Client created")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 3: Check availability
    print("\n[3] Checking Groq API availability...")
    try:
        available = client.is_available()
        if available:
            print("    OK - Groq API reachable")
        else:
            print("    FAIL - Groq API not reachable")
            print("    Check GROQ_API_KEY environment variable")
            return False
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 4: Test plan_subgoals with simple command
    print("\n[4] Testing plan_subgoals (simple command)...")
    command1 = "open notepad"
    print(f"    Command: '{command1}'")

    try:
        subgoals = client.plan_subgoals(command1)
        print(f"    Subgoals ({len(subgoals)}):")
        for i, sg in enumerate(subgoals, 1):
            print(f"      {i}. {sg}")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Test plan_subgoals with complex command
    print("\n[5] Testing plan_subgoals (complex command)...")
    command2 = "open youtube and play me arijit singh songs"
    print(f"    Command: '{command2}'")

    try:
        subgoals = client.plan_subgoals(command2)
        print(f"    Subgoals ({len(subgoals)}):")
        for i, sg in enumerate(subgoals, 1):
            print(f"      {i}. {sg}")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Test plan_subgoals with screen context
    print("\n[6] Testing plan_subgoals (with screen context)...")
    command3 = "search for cats"
    context = "Chrome browser is open showing Google homepage"
    print(f"    Command: '{command3}'")
    print(f"    Context: '{context}'")

    try:
        subgoals = client.plan_subgoals(command3, screen_context=context)
        print(f"    Subgoals ({len(subgoals)}):")
        for i, sg in enumerate(subgoals, 1):
            print(f"      {i}. {sg}")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("plan_subgoals test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_plan_subgoals()
    sys.exit(0 if success else 1)
