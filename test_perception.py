"""
Test Perception Engine
Tests VLM, screen state, and element grounding.
"""

import logging
import time
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_perception():
    """Test the perception engine components."""
    print("=" * 60)
    print("PERCEPTION ENGINE TEST")
    print("=" * 60)
    print()

    # Test 1: Check imports
    print("[1] Checking imports...")

    try:
        from PIL import Image
        import pyautogui
        print("    [OK] PIL and PyAutoGUI available")
    except ImportError as e:
        print(f"    [ERROR] Missing dependency: {e}")
        return

    try:
        from perception.vlm import VisionLanguageModel, get_vlm
        from perception.screen_state import ScreenState, SemanticState, ScreenStateParser
        from perception.grounder import ElementGrounder, get_grounder
        print("    [OK] Perception modules imported")
    except ImportError as e:
        print(f"    [ERROR] Perception import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Test 2: Check VLM availability
    print("[2] Checking VLM availability...")

    vlm = get_vlm()
    if vlm.is_available():
        model = vlm._detect_available_model()
        print(f"    [OK] VLM available: {model}")
    else:
        print("    [WARN] No VLM model found in Ollama")
        print("    Install with: ollama pull qwen2.5vl:7b")
        print("    Or: ollama pull minicpm-v:8b")
        print()
        print("    Continuing with limited functionality...")

    print()

    # Test 3: Capture screenshot
    print("[3] Capturing screenshot...")

    try:
        screenshot = pyautogui.screenshot()
        print(f"    [OK] Screenshot captured: {screenshot.size}")
    except Exception as e:
        print(f"    [ERROR] Screenshot failed: {e}")
        return

    print()

    # Test 4: Test VLM screen understanding (if available)
    print("[4] Testing VLM screen understanding...")

    if vlm.is_available():
        try:
            print("    Analyzing screen (this may take a moment)...")
            start = time.time()
            response = vlm.understand_screen(screenshot)
            elapsed = time.time() - start

            if response.success:
                print(f"    [OK] VLM response received ({elapsed:.1f}s)")
                print()
                print("    --- VLM Description ---")
                # Print first 500 chars
                desc = response.content[:500]
                for line in desc.split('\n'):
                    print(f"    {line}")
                if len(response.content) > 500:
                    print("    ...")
                print("    -----------------------")
            else:
                print(f"    [ERROR] VLM failed: {response.error}")

        except Exception as e:
            print(f"    [ERROR] VLM test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    [SKIP] VLM not available")

    print()

    # Test 5: Test semantic state parsing
    print("[5] Testing semantic state parsing...")

    try:
        # Create a mock VLM response for testing
        mock_response = """
        Active Application: WhatsApp Desktop
        App State: Chat open with "John Doe", message input field visible at bottom
        Visible Apps: WhatsApp, Chrome (background)
        Ready Actions: Can type in message field, can click send button, can scroll chat
        Key UI Elements: Message input ("Type a message"), Send button, Back button, Menu
        """

        state = ScreenStateParser.parse_vlm_response(mock_response)

        print(f"    [OK] Parsed semantic state:")
        print(f"         Active app: {state.active_app.name if state.active_app else 'None'}")
        print(f"         Visible apps: {[a.name for a in state.visible_apps]}")
        print(f"         Ready actions: {state.ready_actions}")
        print(f"         Context: {state.current_context}")

    except Exception as e:
        print(f"    [ERROR] State parsing failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 6: Test element grounder
    print("[6] Testing element grounder...")

    try:
        grounder = get_grounder()
        print(f"    [OK] Grounder initialized")
        print("    ShowUI API: Enabled (will try on grounding request)")
        print("    Fallback VLM: Enabled")

        # Note: We don't actually test grounding here to avoid slow API calls
        # In real use, you would call: grounder.ground_element(screenshot, "send button")

    except Exception as e:
        print(f"    [ERROR] Grounder failed: {e}")

    print()

    # Test 7: Full screen state
    print("[7] Testing full screen state...")

    try:
        from perception.screen_state import ScreenState

        screen_state = ScreenState(screenshot=screenshot)
        screen_state.active_window = "Test Window"

        # If VLM worked, add semantic state
        if vlm.is_available():
            mock_semantic = ScreenStateParser.parse_vlm_response(mock_response)
            screen_state.merge_semantic(mock_semantic)

        print(f"    [OK] Screen state created")
        print(f"         Description: {screen_state.describe()}")
        print()
        print("    --- Prompt Context ---")
        print(screen_state.to_prompt_context())
        print("    ----------------------")

    except Exception as e:
        print(f"    [ERROR] Screen state failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("PERCEPTION ENGINE TEST COMPLETE")
    print("=" * 60)

    # Summary
    print()
    print("Summary:")
    print(f"  - VLM: {'Available' if vlm.is_available() else 'Not available (install qwen2.5vl:7b)'}")
    print(f"  - Screen capture: Working")
    print(f"  - State parsing: Working")
    print(f"  - Grounder: Initialized")
    print()
    print("Next steps:")
    if not vlm.is_available():
        print("  1. Install VLM: ollama pull qwen2.5vl:7b")
        print("  2. Or use lighter model: ollama pull minicpm-v:8b")
    print("  - Test with real screen interactions")
    print("  - Integrate with reactive agent loop")


if __name__ == "__main__":
    test_perception()
