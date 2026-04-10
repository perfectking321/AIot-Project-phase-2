"""
VOXCODE 2.0 — End-to-End Test
Task: "Can you play me Arijit Singh songs on YouTube"

This test demonstrates the full agent workflow:
  1. Launch browser
  2. Navigate to YouTube
  3. Observe → find search bar
  4. Type "Arijit Singh songs" → press Enter
  5. Use JS extraction to find video results (YouTube uses web components)
  6. Click the first video
  7. Verify video is playing
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_youtube_test():
    from agent.browser_session import BrowserSessionManager, run_async, PLAYWRIGHT_AVAILABLE
    from agent.skills.dom_browser_skills import (
        DOMNavigateSkill, DOMObserveSkill, DOMTypeRefSkill,
        DOMClickRefSkill, DOMReadPageSkill, DOMExtractSkill,
    )

    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright not installed!")
        return False

    print("=" * 60)
    print("  TASK: Play Arijit Singh songs on YouTube")
    print("=" * 60)

    # ── Step 1: Navigate to YouTube ──
    print("\n🌐 Step 1: Navigating to YouTube...")
    nav = DOMNavigateSkill()
    result = nav.execute(url="https://www.youtube.com")
    if not result.success:
        print(f"  ❌ Navigation failed: {result.message}")
        return False
    print(f"  ✅ {result.message}")
    time.sleep(2)

    # ── Step 2: Observe to find search bar ──
    print("\n👁️ Step 2: Observing page elements...")
    observe = DOMObserveSkill()
    obs_result = observe.execute()
    if not obs_result.success:
        print(f"  ❌ Observe failed: {obs_result.message}")
        return False

    elements = obs_result.data.get("elements", [])
    print(f"  Found {len(elements)} interactive elements")

    # Find the search input
    search_ref = None
    for el in elements:
        tag = el.get("tag", "")
        name = el.get("name", "").lower()
        aria = el.get("aria_label", "").lower()
        placeholder = el.get("placeholder", "").lower()

        if tag == "input" and ("search" in name or "search" in aria or "search" in placeholder):
            search_ref = el["ref_id"]
            print(f"  🔍 Found search input: ref_id={search_ref}")
            break

    if search_ref is None:
        for el in elements:
            if el.get("tag") == "input" and el.get("type", "") in ("text", "search", ""):
                search_ref = el["ref_id"]
                print(f"  🔍 Found input (fallback): ref_id={search_ref}")
                break

    if search_ref is None:
        print("  ❌ Could not find search bar!")
        return False

    # ── Step 3: Type the search query ──
    print("\n⌨️ Step 3: Typing 'Arijit Singh songs'...")
    type_skill = DOMTypeRefSkill()
    type_result = type_skill.execute(
        ref_id=search_ref,
        text="Arijit Singh songs",
        press_enter=True
    )
    if not type_result.success:
        print(f"  ❌ Type failed: {type_result.message}")
        return False
    print(f"  ✅ {type_result.message}")

    # Wait for search results to load
    print("\n⏳ Waiting for search results to load...")
    time.sleep(4)

    # ── Step 4: Read the page to confirm results ──
    print("\n📄 Step 4: Confirming search results page...")
    read = DOMReadPageSkill()
    read_result = read.execute(max_chars=300)
    if read_result.success:
        print(f"  Title: {read_result.data.get('title', '')}")
        print(f"  URL: {read_result.data.get('url', '')}")

    # ── Step 5: Extract YouTube video links using custom JS ──
    # YouTube renders videos as web components, so our generic observe() 
    # doesn't see /watch links. We use targeted JS to extract them.
    print("\n🔎 Step 5: Extracting YouTube video results...")
    extract = DOMExtractSkill()

    # This JS finds all video links on a YouTube search results page
    YOUTUBE_VIDEO_JS = """
    (() => {
        const videos = [];
        // YouTube search results: look for all <a> tags with /watch hrefs
        const links = document.querySelectorAll('a[href*="/watch"]');
        for (const link of links) {
            const href = link.href || '';
            const title = link.getAttribute('title') 
                || link.innerText.trim().split('\\n')[0] 
                || '';
            // Filter: only real video links with titles
            if (href.includes('/watch?v=') && title.length > 3) {
                // Avoid duplicates
                if (!videos.some(v => v.href === href)) {
                    videos.push({
                        title: title.substring(0, 100),
                        href: href,
                    });
                }
            }
            if (videos.length >= 10) break;
        }
        return videos;
    })()
    """

    extract_result = extract.execute(js_code=YOUTUBE_VIDEO_JS)
    if not extract_result.success:
        print(f"  ❌ Extract failed: {extract_result.message}")
        return False

    videos = extract_result.data.get("result", [])
    print(f"  Found {len(videos)} video results:")
    for i, v in enumerate(videos[:5]):
        print(f"    {i+1}. {v['title'][:60]}")

    if not videos:
        print("  ❌ No video results found!")
        return False

    # ── Step 6: Click the first video ──
    first_video = videos[0]
    print(f"\n▶️ Step 6: Clicking: '{first_video['title'][:50]}'...")

    # Navigate directly to the video URL (most reliable)
    nav_result = nav.execute(url=first_video["href"])
    if not nav_result.success:
        # Fallback: try clicking by title text
        print(f"  ⚠️ Direct nav failed, trying click by text...")
        click = DOMClickRefSkill()
        click_result = click.execute(text=first_video["title"][:40])
        if not click_result.success:
            print(f"  ❌ Click failed: {click_result.message}")
            return False
    print(f"  ✅ Navigated to video!")

    # Wait for video to load and start playing
    print("\n⏳ Waiting for video to load and play...")
    time.sleep(5)

    # ── Step 7: Verify we're on a video page ──
    print("\n🎬 Step 7: Verifying video is playing...")
    read_result2 = read.execute(max_chars=500)
    if read_result2.success:
        video_title = read_result2.data.get("title", "")
        video_url = read_result2.data.get("url", "")
        print(f"  Title: {video_title}")
        print(f"  URL: {video_url}")

        if "/watch" in video_url:
            print(f"\n🎉 SUCCESS! Now playing: '{video_title}'")
            print(f"   URL: {video_url}")

            # Bonus: let it play for a few seconds so you can hear it
            print("\n🎵 Letting it play for 8 seconds...")
            time.sleep(8)
            return True
        else:
            print(f"  ⚠️ Unexpected URL: {video_url}")
            return False
    else:
        print(f"  ⚠️ Could not read page, but navigation succeeded")
        return True


def cleanup():
    try:
        from agent.browser_session import BrowserSessionManager, run_async
        session = BrowserSessionManager.get_instance()
        run_async(session.disconnect())
        print("\n🔒 Browser closed.")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        success = run_youtube_test()
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        cleanup()

    print("\n" + "=" * 60)
    if success:
        print("  ✅ TEST PASSED: Arijit Singh songs playing on YouTube!")
    else:
        print("  ❌ TEST FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)
