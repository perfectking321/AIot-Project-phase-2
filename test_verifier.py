"""
Test Verifier module - Pixel diff verification.
Run: python test_verifier.py
"""
import sys
import time
import numpy as np


def test_verifier():
    print("=" * 60)
    print("VOXCODE Verifier Test (Pixel Diff)")
    print("=" * 60)

    # Test 1: Import
    print("\n[1] Testing Verifier import...")
    try:
        from agent.verifier import Verifier, get_verifier
        print("    OK - Verifier module imported")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Create instance
    print("\n[2] Creating Verifier instance...")
    try:
        verifier = Verifier(change_threshold=5.0, check_radius=100)
        print(f"    OK - threshold={verifier.threshold}, radius={verifier.radius}")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 3: Capture region
    print("\n[3] Testing region capture...")
    t0 = time.time()
    try:
        region = verifier.capture_region(960, 540)  # Screen center
        t1 = time.time()
        if region is not None:
            print(f"    OK - Capture time: {(t1-t0)*1000:.1f}ms")
            print(f"    Region shape: {region.shape}")
        else:
            print("    FAIL - Region is None")
            return False
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 4: Pixel diff with identical images
    print("\n[4] Testing pixel diff (identical images)...")
    region_copy = region.copy()
    changed = verifier.did_screen_change(region, region_copy)
    print(f"    Changed: {changed} (expected: False)")
    if changed:
        print("    WARN - Identical images reported as changed")

    # Test 5: Pixel diff with modified images
    print("\n[5] Testing pixel diff (modified image)...")
    region_modified = region.copy()
    # Add noise to simulate screen change
    region_modified[50:150, 50:150] = 255  # White square
    changed2 = verifier.did_screen_change(region, region_modified)
    print(f"    Changed: {changed2} (expected: True)")
    if not changed2:
        print("    WARN - Modified image not detected as changed")

    # Test 6: Change percentage
    print("\n[6] Testing change percentage...")
    pct = verifier.get_change_percentage(region, region_modified)
    print(f"    Change percentage: {pct:.2f}%")

    # Test 7: Capture two screenshots and compare (should be similar)
    print("\n[7] Testing live screen comparison...")
    shot1 = verifier.capture_region(960, 540)
    time.sleep(0.1)
    shot2 = verifier.capture_region(960, 540)
    live_changed = verifier.did_screen_change(shot1, shot2)
    live_pct = verifier.get_change_percentage(shot1, shot2)
    print(f"    0.1s gap - Changed: {live_changed}, Diff: {live_pct:.2f}%")

    # Test 8: Audit logging
    print("\n[8] Testing audit logging...")
    try:
        verifier.verify_action(
            subgoal="test click",
            decision={"action": "click", "x": 100, "y": 200},
            elements_seen=[{"id": 0, "label": "test"}],
            before_region=region,
            after_region=region_modified,
            retry_count=0
        )
        print("    OK - Audit log written")
    except Exception as e:
        print(f"    FAIL - {e}")

    # Test 9: Statistics
    print("\n[9] Verifier statistics...")
    stats = verifier.get_stats()
    print(f"    Total checks: {stats['total_checks']}")
    print(f"    Changes detected: {stats['changes_detected']}")
    print(f"    No changes: {stats['no_changes_detected']}")

    print("\n" + "=" * 60)
    print("Verifier test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_verifier()
    sys.exit(0 if success else 1)
