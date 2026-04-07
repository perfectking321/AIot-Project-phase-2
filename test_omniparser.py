"""
Test OmniParser Integration
Run this to verify OmniParser is working correctly.
"""

import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_omniparser():
    """Test OmniParser screen parsing."""
    print("=" * 60)
    print("OMNIPARSER TEST")
    print("=" * 60)
    print()

    # Check dependencies
    print("[1] Checking dependencies...")

    try:
        import torch
        print(f"    [OK] PyTorch {torch.__version__}")
        print(f"         CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("    [WARN] PyTorch not installed")

    try:
        import easyocr
        print(f"    [OK] EasyOCR available")
    except ImportError:
        print("    [ERROR] EasyOCR not installed - run: pip install easyocr")
        return

    try:
        from ultralytics import YOLO
        print(f"    [OK] Ultralytics (YOLO) available")
    except ImportError:
        print("    [WARN] Ultralytics not installed - icon detection disabled")
        print("           run: pip install ultralytics")

    try:
        import pyautogui
        screen_size = pyautogui.size()
        print(f"    [OK] PyAutoGUI - Screen: {screen_size.width}x{screen_size.height}")
    except ImportError:
        print("    [ERROR] PyAutoGUI not installed")
        return

    print()

    # Test OmniParser
    print("[2] Initializing OmniParser...")

    try:
        from agent.omniparser import OmniParser, get_omniparser

        start = time.time()
        parser = OmniParser(preload=False)
        init_time = time.time() - start
        print(f"    [OK] OmniParser initialized in {init_time:.2f}s")
        print(f"         Device: {parser.device}")
    except Exception as e:
        print(f"    [ERROR] Failed to initialize: {e}")
        return

    print()

    # Parse screen
    print("[3] Parsing current screen...")
    print("    (This may take a few seconds on first run)")
    print()

    try:
        start = time.time()
        result = parser.parse_screen(save_labeled=True)
        parse_time = time.time() - start

        print(f"    [OK] Parsed in {parse_time:.2f}s")
        print(f"         Found {len(result.elements)} UI elements")
        print()

        if result.labeled_image_path:
            print(f"    Labeled image saved: {result.labeled_image_path}")
            print()

        # Show detected elements
        print("[4] Detected Elements:")
        print("-" * 60)

        for elem in result.elements[:15]:  # Show first 15
            interactable = "Y" if elem.is_interactable else "N"
            print(f"    [{elem.id:2d}] {elem.element_type:8s} | {elem.label[:40]:40s} | Int:{interactable}")

        if len(result.elements) > 15:
            print(f"    ... and {len(result.elements) - 15} more elements")

        print("-" * 60)
        print()

        # Show prompt format
        print("[5] LLM Prompt Format:")
        print("-" * 60)
        print(result.to_prompt_format()[:500])
        if len(result.to_prompt_format()) > 500:
            print("...")
        print("-" * 60)

        print()
        print("[SUCCESS] OmniParser is working correctly!")
        print()

        # Test search
        print("[6] Testing element search...")
        search_terms = ["search", "type", "send", "close", "settings"]

        for term in search_terms:
            matches = result.find_by_label(term)
            if matches:
                print(f"    '{term}' -> Found: [{matches[0].id}] \"{matches[0].label}\"")
            else:
                print(f"    '{term}' -> Not found")

    except Exception as e:
        print(f"    [ERROR] Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_omniparser()
