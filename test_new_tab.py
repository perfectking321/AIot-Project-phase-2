#!/usr/bin/env python3
"""
Quick Browser Test for VOXCODE
Test the new tab functionality.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_new_tab():
    """Test opening a new tab and going to YouTube."""
    print("=" * 60)
    print("  VOXCODE Browser-Use Test: New Tab + YouTube")
    print("=" * 60)
    print()

    from agent.skills.browser_agent import BrowserUseAgent

    def on_status(msg):
        print(f"   Status: {msg}")

    def on_step(step_num, msg, status):
        icons = {"running": "○", "done": "●", "failed": "✗"}
        icon = icons.get(status, "○")
        print(f"   {icon} Step {step_num}: {msg}")

    agent = BrowserUseAgent(
        on_status=on_status,
        on_step=on_step,
        headless=False  # Show browser window
    )

    # Test the new tab request
    task = "Open a new tab and go to YouTube"

    print(f"Task: {task}")
    print("(This will open a new Chrome window - this is normal!)")
    print()

    try:
        result = await asyncio.wait_for(agent.execute_task(task), timeout=60)

        print()
        if result.success:
            print("✅ SUCCESS!")
            print(f"   {result.message}")
            print()
            print("🎉 The browser automation is working!")
            print("📝 Note: Browser-Use opens its own Chrome window for reliability")
            print("   This is separate from your existing Chrome windows")
        else:
            print("❌ FAILED")
            print(f"   {result.message}")

        return result.success

    except asyncio.TimeoutError:
        print("⏰ TIMEOUT: Task took too long (but browser might still be working)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_new_tab())

        print()
        print("=" * 60)
        if success:
            print("🎉 TEST PASSED - VOXCODE Browser automation is working!")
            print()
            print("Now you can use voice commands like:")
            print('   • "Go to YouTube and search for cats"')
            print('   • "Open a new tab and go to Google"')
            print('   • "Search for Python tutorials"')
        else:
            print("❌ TEST FAILED - Check the errors above")
        print("=" * 60)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        sys.exit(1)