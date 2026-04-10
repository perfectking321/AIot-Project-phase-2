"""
VOXCODE 2.0 Architecture Test Suite
Tests: SystemContextProvider, BrowserSessionManager, SkillRouter, DOM Skills

The browser tests launch a REAL Playwright Chromium browser (no CDP needed).
"""

import sys
import os
import time
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
SKIP = 0

def test(name, func):
    global PASS, FAIL, SKIP
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        result = func()
        if result == "SKIP":
            print(f"  ⚠️  SKIPPED")
            SKIP += 1
        else:
            print(f"  ✅  PASSED")
            PASS += 1
    except Exception as e:
        print(f"  ❌  FAILED: {e}")
        traceback.print_exc()
        FAIL += 1


# ═══════════════════════════════════════════════════════════════
# TEST 1: SystemContextProvider
# ═══════════════════════════════════════════════════════════════

def test_system_context():
    from agent.system_context import SystemContextProvider, get_system_context

    provider = SystemContextProvider()
    provider.start()
    time.sleep(1)  # Let it do one cycle

    ctx = get_system_context()
    print(f"  Context keys: {list(ctx.keys())}")
    print(f"  Active window: {ctx.get('active_window', 'N/A')}")
    print(f"  Battery: {ctx.get('battery', 'N/A')}")
    print(f"  Clipboard length: {len(ctx.get('clipboard', ''))}")

    assert "active_window" in ctx, "active_window key missing"
    assert "battery" in ctx, "battery key missing"
    assert "clipboard" in ctx, "clipboard key missing"

    provider.stop()
    print("  SystemContextProvider works correctly.")


# ═══════════════════════════════════════════════════════════════
# TEST 2: SkillRouter
# ═══════════════════════════════════════════════════════════════

def test_skill_router():
    from agent.dispatcher import SkillRouter

    test_cases = [
        ("search for python tutorials on youtube", "BROWSER_DOM"),
        ("open youtube and play a the first video you see", "BROWSER_DOM"),
        ("google latest AI news", "BROWSER_DOM"),
        ("set volume to 50", "SYSTEM_API"),
        ("turn off bluetooth", "SYSTEM_API"),
        ("show my ip address", "SYSTEM_API"),
        ("mute the sound", "SYSTEM_API"),
        ("list files in documents", "TERMINAL"),
        ("create file called test.txt", "TERMINAL"),
        ("pip install requests", "TERMINAL"),
        ("open notepad and type hello", "NATIVE_UIA"),
        ("do something weird with custom app", "OMNIPARSER"),
    ]

    all_ok = True
    for command, expected in test_cases:
        actual = SkillRouter.route(command)
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_ok = False
        print(f"  {status} '{command}' → {actual} (expected: {expected})")

    assert all_ok, "Some SkillRouter routes did not match expected channels"
    print("  SkillRouter all routes correct.")


# ═══════════════════════════════════════════════════════════════
# TEST 3: BrowserSessionManager — Launch browser
# ═══════════════════════════════════════════════════════════════

def test_browser_session_manager():
    from agent.browser_session import BrowserSessionManager, PLAYWRIGHT_AVAILABLE, run_async

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    print("  Launching Playwright Chromium browser...")

    async def _test_launch():
        session = BrowserSessionManager.get_instance()
        browser, page = await session.get_active_page()
        # Navigate to a real page to verify the browser works
        await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
        title = await page.title()
        url = page.url
        return title, url

    title, url = run_async(_test_launch())
    print(f"  Browser launched! Page: '{title}' at {url}")
    assert title is not None, "Page title is None"
    assert "google" in url.lower(), f"Expected Google URL, got {url}"
    print("  BrowserSessionManager launch + navigation works.")


# ═══════════════════════════════════════════════════════════════
# TEST 4: DOM Skills — Observe (list interactive elements)
# ═══════════════════════════════════════════════════════════════

def test_dom_observe():
    from agent.skills.dom_browser_skills import DOMObserveSkill, PLAYWRIGHT_AVAILABLE

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    # First navigate to Google so there ARE interactive elements
    from agent.skills.dom_browser_skills import DOMNavigateSkill
    nav_skill = DOMNavigateSkill()
    nav_result = nav_skill.execute(url="https://www.google.com")
    print(f"  Navigate: {nav_result.message}")

    time.sleep(1)

    skill = DOMObserveSkill()
    result = skill.execute()

    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")

    assert result.success, f"Observe failed: {result.message}"

    count = result.data.get("count", 0)
    print(f"  Interactive elements found: {count}")

    # Show first 10 elements
    for el in result.data.get("elements", [])[:10]:
        name = el.get("name", "")[:40]
        print(f"    [{el['ref_id']}] <{el['tag']}> {name}")

    assert count > 0, "No elements found on Google"
    print("  DOMObserveSkill works correctly.")


# ═══════════════════════════════════════════════════════════════
# TEST 5: DOM Skills — Read Page
# ═══════════════════════════════════════════════════════════════

def test_dom_read_page():
    from agent.skills.dom_browser_skills import DOMReadPageSkill, PLAYWRIGHT_AVAILABLE

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    skill = DOMReadPageSkill()
    result = skill.execute(max_chars=500)

    print(f"  Status: {result.status}")
    assert result.success, f"ReadPage failed: {result.message}"

    print(f"  Title: {result.data.get('title', '')}")
    print(f"  URL: {result.data.get('url', '')}")
    print(f"  Body chars: {result.data.get('char_count', 0)}")

    assert result.data.get("title"), "Page title is empty"
    print("  DOMReadPageSkill works correctly.")


# ═══════════════════════════════════════════════════════════════
# TEST 6: DOM Skills — Accessibility Tree
# ═══════════════════════════════════════════════════════════════

def test_dom_accessibility():
    from agent.skills.dom_browser_skills import DOMAccessibilitySkill, PLAYWRIGHT_AVAILABLE

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    skill = DOMAccessibilitySkill()
    result = skill.execute()

    print(f"  Status: {result.status}")
    assert result.success, f"Accessibility failed: {result.message}"

    tree = result.data.get("tree", {})
    role = tree.get("role", "N/A")
    name = tree.get("name", "N/A")
    children_count = len(tree.get("children", []))
    print(f"  Root role: {role}")
    print(f"  Root name: {name}")
    print(f"  Children count: {children_count}")

    assert role is not None, "Accessibility tree role is None"
    print("  DOMAccessibilitySkill works correctly.")


# ═══════════════════════════════════════════════════════════════
# TEST 7: DOM Skills — Type + Click (Google Search)
# ═══════════════════════════════════════════════════════════════

def test_dom_search_workflow():
    """Full workflow: navigate to Google → observe → type → click search."""
    from agent.skills.dom_browser_skills import (
        DOMNavigateSkill, DOMObserveSkill, DOMTypeRefSkill,
        DOMClickRefSkill, DOMReadPageSkill, PLAYWRIGHT_AVAILABLE
    )

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    # Step 1: Navigate to Google
    print("  Step 1: Navigate to Google...")
    nav = DOMNavigateSkill()
    nav_result = nav.execute(url="https://www.google.com")
    assert nav_result.success, f"Navigate failed: {nav_result.message}"
    print(f"    → {nav_result.message}")
    time.sleep(1)

    # Step 2: Observe elements
    print("  Step 2: Observe interactive elements...")
    observe = DOMObserveSkill()
    obs_result = observe.execute()
    assert obs_result.success, f"Observe failed: {obs_result.message}"
    elements = obs_result.data.get("elements", [])
    print(f"    → Found {len(elements)} elements")

    # Step 3: Find the search input
    search_input_ref = None
    for el in elements:
        tag = el.get("tag", "")
        el_type = el.get("type", "")
        name = el.get("name", "").lower()
        aria = el.get("aria_label", "").lower()
        if tag == "textarea" and ("search" in name or "search" in aria):
            search_input_ref = el["ref_id"]
            print(f"    → Found search textarea: ref_id={search_input_ref}")
            break
        if tag == "input" and el_type == "text" and ("search" in name or "search" in aria):
            search_input_ref = el["ref_id"]
            print(f"    → Found search input: ref_id={search_input_ref}")
            break

    if search_input_ref is None:
        # Try by name/aria
        for el in elements:
            if "search" in el.get("aria_label", "").lower():
                search_input_ref = el["ref_id"]
                print(f"    → Found search element by aria-label: ref_id={search_input_ref}")
                break

    assert search_input_ref is not None, f"Could not find search input. Elements: {[e.get('tag') + ':' + e.get('name','')[:20] for e in elements[:15]]}"

    # Step 4: Type a search query
    print("  Step 4: Type 'VOXCODE AI agent test'...")
    type_skill = DOMTypeRefSkill()
    type_result = type_skill.execute(ref_id=search_input_ref, text="VOXCODE AI agent test", press_enter=True)
    assert type_result.success, f"Type failed: {type_result.message}"
    print(f"    → {type_result.message}")

    # Wait for results
    time.sleep(3)

    # Step 5: Read the results page
    print("  Step 5: Read search results page...")
    read = DOMReadPageSkill()
    read_result = read.execute(max_chars=1000)
    assert read_result.success, f"ReadPage failed: {read_result.message}"
    title = read_result.data.get("title", "")
    url = read_result.data.get("url", "")
    body = read_result.data.get("body_text", "")[:200]
    print(f"    → Title: {title}")
    print(f"    → URL: {url}")
    print(f"    → Body preview: {body}...")

    assert "VOXCODE" in title or "voxcode" in title.lower() or "search" in url.lower(), \
        f"Expected search results page, got title='{title}', url='{url}'"

    print("  Full Google search workflow completed successfully!")


# ═══════════════════════════════════════════════════════════════
# TEST 8: DOM Skills — Scroll
# ═══════════════════════════════════════════════════════════════

def test_dom_scroll():
    from agent.skills.dom_browser_skills import DOMScrollSkill, PLAYWRIGHT_AVAILABLE

    if not PLAYWRIGHT_AVAILABLE:
        print("  Playwright not installed, skipping.")
        return "SKIP"

    skill = DOMScrollSkill()
    result = skill.execute(direction="down", amount=300)

    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    assert result.success, f"Scroll failed: {result.message}"
    print("  DOMScrollSkill works correctly.")


# ═══════════════════════════════════════════════════════════════
# TEST 9: Dispatcher integration
# ═══════════════════════════════════════════════════════════════

def test_dispatcher_import():
    from agent.dispatcher import CommandDispatcher, SkillRouter, get_dispatcher

    dispatcher = get_dispatcher(
        on_message=lambda m: print(f"    MSG: {m}"),
        on_state_change=lambda s: None,
    )

    print(f"  Dispatcher type: {type(dispatcher).__name__}")
    print(f"  SkillRouter available: {SkillRouter is not None}")

    channel = SkillRouter.route("set volume to 50")
    print(f"  Route 'set volume to 50' → {channel}")
    assert channel == "SYSTEM_API"

    print("  Dispatcher imports and routing work correctly.")


# ═══════════════════════════════════════════════════════════════
# CLEANUP: Close the browser after all tests
# ═══════════════════════════════════════════════════════════════

def cleanup_browser():
    """Close the Playwright browser after all tests."""
    try:
        from agent.browser_session import BrowserSessionManager, PLAYWRIGHT_AVAILABLE, run_async
        if PLAYWRIGHT_AVAILABLE:
            session = BrowserSessionManager.get_instance()
            run_async(session.disconnect())
            print("  Browser closed.")
    except Exception as e:
        print(f"  Browser cleanup: {e}")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VOXCODE 2.0 ARCHITECTURE TEST SUITE")
    print("  (Launches its own browser — no CDP needed)")
    print("=" * 60)

    test("1. SystemContextProvider", test_system_context)
    test("2. SkillRouter", test_skill_router)
    test("3. BrowserSessionManager (launch browser)", test_browser_session_manager)
    test("4. DOMObserveSkill (list elements)", test_dom_observe)
    test("5. DOMReadPageSkill", test_dom_read_page)
    test("6. DOMAccessibilitySkill", test_dom_accessibility)
    test("7. Full Search Workflow (type + click)", test_dom_search_workflow)
    test("8. DOMScrollSkill", test_dom_scroll)
    test("9. Dispatcher Integration", test_dispatcher_import)

    # Cleanup
    print(f"\n{'='*60}")
    print("CLEANUP: Closing browser...")
    cleanup_browser()

    print("\n" + "=" * 60)
    print(f"  RESULTS: ✅ {PASS} passed, ❌ {FAIL} failed, ⚠️ {SKIP} skipped")
    print("=" * 60 + "\n")

    if FAIL > 0:
        sys.exit(1)
    else:
        sys.exit(0)
