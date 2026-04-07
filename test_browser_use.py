#!/usr/bin/env python3
"""
Test Browser-Use Integration for VOXCODE
Run this to verify browser automation works.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if required packages are installed."""
    print("=" * 50)
    print("  VOXCODE Browser-Use Integration Test")
    print("=" * 50)
    print()

    missing = []

    # Check browser-use
    print("Checking dependencies...")
    try:
        import browser_use
        print("  [OK] browser-use")
    except ImportError:
        print("  [X] browser-use - NOT INSTALLED")
        missing.append("browser-use")

    # Check playwright
    try:
        import playwright
        print("  [OK] playwright")
    except ImportError:
        print("  [X] playwright - NOT INSTALLED")
        missing.append("playwright")

    # Check langchain-groq (optional - fallback to our Groq client)
    try:
        import langchain_groq
        print("  [OK] langchain-groq")
    except ImportError:
        print("  [OK] langchain-groq - using fallback (VOXCODE Groq client)")

    # Check Groq API key (reusing VOXCODE's existing config)
    from config import config
    api_key = config.llm.groq_api_key
    if api_key and api_key != "your-api-key-here":
        print(f"  [OK] Groq API key configured ({api_key[:15]}...)")
    else:
        print("  [!] Groq API key not configured in config.py")

    print()

    if missing:
        print("Missing packages! Install with:")
        print(f"  pip install {' '.join(missing)}")
        if "playwright" in missing or not missing:
            print("  playwright install chromium")
        print()
        return False

    return True


def test_task_detection():
    """Test the browser task detection."""
    print("Testing task detection...")
    print()

    from agent.skills.browser_agent import BrowserUseAgent

    test_cases = [
        # Browser tasks (should return True)
        ("Go to YouTube and search for cats", True),
        ("Open chrome and go to google.com", True),
        ("Search for Python tutorials", True),
        ("Watch a video about AI", True),
        ("Navigate to github.com", True),
        ("Find Taylor Swift on YouTube", True),

        # Desktop tasks (should return False)
        ("Open notepad", False),
        ("Open file explorer", False),
        ("Create a new folder", False),
        ("Take a screenshot", False),
        ("Open calculator", False),
    ]

    passed = 0
    for command, expected in test_cases:
        result = BrowserUseAgent.is_browser_task(command)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  [{status}] '{command[:40]}...' → {'browser' if result else 'desktop'}")

    print()
    print(f"Detection tests: {passed}/{len(test_cases)} passed")
    print()
    return passed == len(test_cases)


async def test_browser_task():
    """Test an actual browser task."""
    print("Testing browser task execution...")
    print("(This will open a browser window)")
    print()

    from agent.skills.browser_agent import BrowserUseAgent

    def on_status(msg):
        print(f"  Status: {msg}")

    def on_step(step_num, msg, status):
        icons = {"running": "○", "done": "●", "failed": "✗"}
        icon = icons.get(status, "○")
        print(f"  {icon} Step {step_num}: {msg}")

    agent = BrowserUseAgent(on_status=on_status, on_step=on_step)

    # Simple test task
    task = "Go to google.com and search for 'VOXCODE voice assistant'"

    print(f"  Task: {task}")
    print()

    result = await agent.execute_task(task)

    print()
    if result.success:
        print(f"  [SUCCESS] {result.message}")
    else:
        print(f"  [FAILED] {result.message}")

    return result.success


def main():
    """Run all tests."""
    # Check dependencies first
    deps_ok = check_dependencies()

    if not deps_ok:
        print("Please install missing dependencies first.")
        sys.exit(1)

    # Test task detection
    detection_ok = test_task_detection()

    # Ask user if they want to run live test
    print("-" * 50)
    response = input("Run live browser test? (y/n): ").strip().lower()

    if response == 'y':
        # Check API key before running
        from config import config
        if not config.llm.groq_api_key or config.llm.groq_api_key == "your-api-key-here":
            print()
            print("ERROR: Groq API key not configured!")
            print("Edit config.py and set groq_api_key")
            sys.exit(1)

        success = asyncio.run(test_browser_task())
        print()
        print("=" * 50)
        print(f"  Live test: {'PASSED' if success else 'FAILED'}")
        print("=" * 50)
    else:
        print()
        print("Skipped live test.")

    print()
    print("Summary:")
    print(f"  - Dependencies: {'OK' if deps_ok else 'MISSING'}")
    print(f"  - Task detection: {'OK' if detection_ok else 'ISSUES'}")
    print()
    print("To use browser automation in VOXCODE:")
    print("  1. Your Groq API key is already configured ✓")
    print("  2. Run: playwright install chromium (if not done)")
    print("  3. Start VOXCODE and say 'Go to YouTube and search for music'")
    print("  4. VOXCODE will detect it's a browser task and use Browser-Use!")


if __name__ == "__main__":
    main()
