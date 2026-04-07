#!/usr/bin/env python3
"""
Test Normal Chrome Integration for VOXCODE
This will work with your regular Chrome browser - no special setup required!
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_normal_chrome():
    """Test controlling normal Chrome browser."""
    print("=" * 60)
    print("  VOXCODE Normal Chrome Test")
    print("=" * 60)
    print()

    from agent.skills.normal_chrome_agent import NormalChromeAgent
    from agent.tools import WindowsTools
    from agent.vision import ScreenVision

    def on_status(msg):
        print(f"   Status: {msg}")

    def on_step(step_num, msg, status):
        icons = {"running": "○", "done": "●", "failed": "✗"}
        icon = icons.get(status, "○")
        print(f"   {icon} Step {step_num}: {msg}")

    # Initialize vision and tools (like VOXCODE does)
    print("Initializing tools...")
    try:
        vision = ScreenVision(preload=True)
        tools = WindowsTools(vision_instance=vision)
        print("✓ Tools ready")
    except Exception as e:
        print(f"✗ Tools failed: {e}")
        return False

    agent = NormalChromeAgent(
        on_status=on_status,
        on_step=on_step,
        tools=tools
    )

    # Test opening a new tab and going to YouTube
    task = "Open a new tab and go to YouTube"

    print(f"Task: {task}")
    print()
    print("This will work with your normal Chrome browser!")
    print("No special setup required - just make sure Chrome is running.")
    print()

    try:
        result = await asyncio.wait_for(agent.execute_task(task), timeout=45)

        print()
        if result.success:
            print("✅ SUCCESS!")
            print(f"   {result.message}")
            print()
            print("🎉 Your normal Chrome browser should now have YouTube open!")
        else:
            print("❌ FAILED")
            print(f"   {result.message}")

        return result.success

    except asyncio.TimeoutError:
        print("⏰ TIMEOUT: Task took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_normal_chrome())

        print()
        print("=" * 60)
        if success:
            print("🎉 TEST PASSED - VOXCODE can control your normal Chrome!")
            print()
            print("Now you can use voice commands like:")
            print('   • "Open a new tab and go to YouTube"')
            print('   • "Search for Python tutorials"')
            print('   • "Go to YouTube and search for cats"')
            print()
            print("These will work with your normal Chrome browser!")
            print("No special Chrome setup required!")
        else:
            print("❌ TEST FAILED - Check the errors above")
            print()
            print("Make sure Chrome is running and try again.")
        print("=" * 60)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        sys.exit(1)