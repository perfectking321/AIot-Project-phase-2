"""
Test Reactive Agent
Run this to test the perceive-think-act-verify loop.
"""

import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_reactive_agent():
    """Test the reactive agent with a sample task."""
    print("=" * 60)
    print("REACTIVE AGENT TEST")
    print("=" * 60)
    print()

    # Check dependencies
    print("[1] Checking dependencies...")

    try:
        import pyautogui
        print(f"    [OK] PyAutoGUI available")
    except ImportError:
        print("    [ERROR] PyAutoGUI not installed")
        return

    try:
        from agent.omniparser import get_omniparser
        print("    [OK] OmniParser available")
    except ImportError as e:
        print(f"    [ERROR] OmniParser not available: {e}")
        return

    try:
        from agent.reactive_loop import ReactiveAgent, ScreenState
        print("    [OK] ReactiveAgent available")
    except ImportError as e:
        print(f"    [ERROR] ReactiveAgent not available: {e}")
        return

    print()

    # Test screen state capture
    print("[2] Testing screen state capture...")

    try:
        from agent.reactive_loop import ReactiveAgent

        agent = ReactiveAgent(
            on_message=lambda m: print(f"    > {m}"),
            on_state_change=lambda s: print(f"    [STATE] {s}"),
            max_iterations=10,  # More iterations for complex tasks
            verify_actions=False  # Disable verification for simple test
        )

        state = agent.capture_screen_state()
        print(f"    [OK] Captured screen state:")
        print(f"         Window: {state.active_window}")
        print(f"         Elements: {len(state.visible_elements)} found")
        print(f"         Apps detected: {state.visible_apps}")

        if state.visible_elements:
            # Filter out just "icon"
            meaningful = [e for e in state.visible_elements if e.lower() != "icon"][:10]
            print(f"         Sample elements: {meaningful}")

    except Exception as e:
        print(f"    [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Test action decision
    print("[3] Testing action decision (requires Ollama)...")

    try:
        goal = "Open notepad"
        print(f"    Goal: '{goal}'")

        decision = agent.decide_next_action(goal, state)
        print(f"    [OK] Decision: {decision}")

    except Exception as e:
        print(f"    [WARN] Decision failed (Ollama may not be running): {e}")

    print()

    # Interactive test
    print("[4] Interactive Test")
    print("-" * 60)
    print("Enter a simple goal to test (or 'quit' to exit):")
    print("Examples:")
    print("  - Open notepad")
    print("  - Click on the Start button")
    print("  - Type hello world")
    print()

    while True:
        try:
            goal = input("Goal> ").strip()

            if not goal or goal.lower() in ['quit', 'exit', 'q']:
                break

            print()
            print(f"Processing: {goal}")
            print("-" * 40)

            result = agent.process_goal(goal)

            print("-" * 40)
            print(f"Result: {result}")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_reactive_agent()
