"""
Test Eyes module - OmniParser wrapper with token compression.
Run: python test_eyes.py
"""
import sys
import time

def test_eyes():
    print("=" * 60)
    print("VOXCODE Eyes Test")
    print("=" * 60)

    # Test 1: Import and initialization
    print("\n[1] Testing Eyes import and initialization...")
    try:
        from agent.eyes import Eyes, get_eyes, ScreenElement
        print("    OK - Eyes module imported")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Create Eyes instance (may take time to load models)
    print("\n[2] Creating Eyes instance (loading OmniParser)...")
    t0 = time.time()
    try:
        eyes = Eyes(use_caption_model=False, preload=True)  # Disable caption for faster test
        t1 = time.time()
        print(f"    OK - Eyes created in {(t1-t0):.1f}s")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 3: Scan screen
    print("\n[3] Scanning screen (OmniParser + OCR)...")
    t0 = time.time()
    try:
        elements = eyes.scan()
        t1 = time.time()
        print(f"    OK - Scan time: {(t1-t0)*1000:.0f}ms")
        print(f"    Total elements found: {len(elements)}")

        # Show sample elements
        if elements:
            print("\n    Sample elements:")
            for e in elements[:5]:
                print(f"      [{e.id}] {e.element_type}: '{e.label[:30]}' at {e.center}")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Cache test (second scan should be fast)
    print("\n[4] Testing cache (scan within 100ms)...")
    t0 = time.time()
    elements2 = eyes.scan()  # Should return cached
    t1 = time.time()
    if (t1 - t0) < 0.05:  # Less than 50ms = cached
        print(f"    OK - Cache hit: {(t1-t0)*1000:.1f}ms (cached)")
    else:
        print(f"    WARN - Cache may not be working: {(t1-t0)*1000:.1f}ms")

    # Test 5: Token compression (filter_near)
    print("\n[5] Testing token compression (filter_near)...")
    if len(elements) > 0:
        # Filter around screen center
        cx, cy = 960, 540
        filtered = eyes.filter_near(elements, cx, cy, radius=350)

        full_str = eyes.elements_to_prompt_str(elements)
        filtered_str = eyes.elements_to_prompt_str(filtered)

        full_tokens_est = len(full_str.split()) * 1.3
        filtered_tokens_est = len(filtered_str.split()) * 1.3

        print(f"    Full elements: {len(elements)} -> ~{full_tokens_est:.0f} tokens")
        print(f"    Filtered (r=350): {len(filtered)} -> ~{filtered_tokens_est:.0f} tokens")
        print(f"    Token reduction: {(1 - filtered_tokens_est/full_tokens_est)*100:.0f}%")
    else:
        print("    SKIP - No elements to filter")

    # Test 6: Prompt formatting
    print("\n[6] Testing prompt formatting...")
    if elements:
        sample = elements[:3]
        prompt_str = eyes.elements_to_prompt_str(sample)
        print("    Sample output format:")
        for line in prompt_str.split("\n"):
            print(f"      {line}")
    else:
        print("    SKIP - No elements")

    # Test 7: Force scan (bypass cache)
    print("\n[7] Testing force scan (bypass cache)...")
    t0 = time.time()
    elements3 = eyes.scan(force=True)
    t1 = time.time()
    print(f"    Forced scan time: {(t1-t0)*1000:.0f}ms")
    print(f"    Elements: {len(elements3)}")

    # Test 8: Screen summary
    print("\n[8] Testing screen summary...")
    if elements3:
        summary = eyes.get_screen_summary(elements3)
        print(f"    {summary}")

    print("\n" + "=" * 60)
    print("Eyes test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_eyes()
    sys.exit(0 if success else 1)
