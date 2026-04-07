#!/usr/bin/env python3
"""
Test Existing Browser Integration for VOXCODE
This will connect to your existing Chrome browser.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_existing_browser():
    """Test connecting to existing Chrome browser."""
    print("=" * 60)
    print("  VOXCODE Existing Browser Test")
    print("=" * 60)
    print()

    from agent.skills.existing_browser_agent import ExistingBrowserAgent

    def on_status(msg):
        print(f"   Status: {msg}")

    def on_step(step_num, msg, status):
        icons = {"running": "○", "done": "●", "failed": "✗"}
        icon = icons.get(status, "○")
        print(f"   {icon} Step {step_num}: {msg}")

    agent = ExistingBrowserAgent(
        on_status=on_status,
        on_step=on_step
    )

    # Test opening a new tab in existing Chrome
    task = "Open a new tab and go to Google"

    print(f"Task: {task}")
    print()
    print("Prerequisites:")
    print("1. Make sure Chrome is running")
    print("2. Chrome should be started with: chrome.exe --remote-debugging-port=9222")
    print()

    try:
        result = await asyncio.wait_for(agent.execute_task(task), timeout=30)

        print()
        if result.success:
            print("✅ SUCCESS!")
            print(f"   {result.message}")
            print()
            print("🎉 Your existing Chrome browser should now have a new tab with Google!")
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


def check_chrome_debugging():
    """Check if Chrome is running with remote debugging."""
    print("Checking if Chrome is running with remote debugging...")

    import requests

    for port in [9222, 9223, 9224, 9225]:
        try:
            response = requests.get(f"http://localhost:{port}/json/version", timeout=2)
            if response.status_code == 200:
                print(f"✅ Found Chrome with remote debugging on port {port}")
                return True
        except:
            continue

    print("❌ Chrome not found with remote debugging")
    print()
    print("To start Chrome with remote debugging:")
    print("1. Close all Chrome windows")
    print("2. Run this command:")
    print("   chrome.exe --remote-debugging-port=9222")
    print("3. Or on Windows:")
    print('   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222')
    print()
    return False


if __name__ == "__main__":
    print("Checking Chrome status...")
    chrome_ok = check_chrome_debugging()

    if not chrome_ok:
        print("Please start Chrome with remote debugging and try again.")
        sys.exit(1)

    try:
        success = asyncio.run(test_existing_browser())

        print()
        print("=" * 60)
        if success:
            print("🎉 TEST PASSED - VOXCODE can control your existing Chrome!")
            print()
            print("Now you can use voice commands like:")
            print('   • "Open a new tab and go to YouTube"')
            print('   • "Search for Python tutorials"')
            print('   • "Go to YouTube and search for cats"')
            print()
            print("These will work in your existing Chrome browser with your profile!")
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