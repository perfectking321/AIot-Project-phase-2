"""
VOXCODE Full Integration Test - YouTube Arijit Singh Workflow

This test demonstrates the complete Plan1.md architecture:
1. Groq 70B plans subgoals (ONE API call)
2. OmniParser + Qwen/Groq executes each subgoal
3. Pixel diff verifies each action

The LLM must autonomously:
- Handle Chrome profile selection panel (no hardcoded instructions)
- Navigate to YouTube
- Search for Arijit Singh songs
- Click the first video result

Run: python test_youtube_integration.py
"""
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("integration_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voxcode.integration")


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(num: int, msg: str, status: str = ""):
    """Print a step update."""
    icons = {"running": "◌", "done": "●", "failed": "✗", "": ""}
    icon = icons.get(status, "○")
    print(f"  {icon} Step {num}: {msg}")


def test_full_workflow():
    """Test the complete YouTube workflow."""
    print_header("VOXCODE Integration Test - YouTube Arijit Singh")

    # The command to test
    command = "open youtube and play me arijit singh songs"
    print(f"\n  Command: '{command}'")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Test component imports
    # ══════════════════════════════════════════════════════════════════════
    print_header("Phase 1: Component Verification")

    components = {}

    # Test Eyes
    print("\n  [1/4] Loading Eyes (OmniParser)...")
    try:
        from agent.eyes import get_eyes, Eyes
        eyes = get_eyes(use_caption_model=False, preload=True)
        components['eyes'] = eyes
        print("        ✓ Eyes loaded successfully")
    except Exception as e:
        print(f"        ✗ Eyes failed: {e}")
        return False

    # Test Hands
    print("\n  [2/4] Loading Hands (Qwen/Groq)...")
    try:
        from agent.hands import get_hands, Hands
        hands = get_hands()
        components['hands'] = hands
        if hands._use_groq:
            print("        ✓ Hands loaded (using Groq fallback)")
        else:
            print(f"        ✓ Hands loaded (Ollama model: {hands.model})")
    except Exception as e:
        print(f"        ✗ Hands failed: {e}")
        return False

    # Test Verifier
    print("\n  [3/4] Loading Verifier (Pixel Diff)...")
    try:
        from agent.verifier import get_verifier, Verifier
        verifier = get_verifier()
        components['verifier'] = verifier
        print("        ✓ Verifier loaded successfully")
    except Exception as e:
        print(f"        ✗ Verifier failed: {e}")
        return False

    # Test Groq Client
    print("\n  [4/4] Testing Groq API...")
    try:
        from brain.llm import GroqClient
        groq = GroqClient()
        if groq.is_available():
            print("        ✓ Groq API available")
        else:
            print("        ✗ Groq API not available")
            return False
    except Exception as e:
        print(f"        ✗ Groq client failed: {e}")
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Planning with Groq
    # ══════════════════════════════════════════════════════════════════════
    print_header("Phase 2: Groq Planning (ONE API call)")

    print(f"\n  Planning subgoals for: '{command}'")

    # Get current screen context
    try:
        import pygetwindow as gw
        active_win = gw.getActiveWindow()
        screen_context = active_win.title if active_win else "Unknown"
    except:
        screen_context = "Unknown"

    print(f"  Current screen: {screen_context}")

    t0 = time.time()
    try:
        subgoals = groq.plan_subgoals(command, screen_context)
        t1 = time.time()
        print(f"\n  ✓ Groq planned {len(subgoals)} subgoals in {(t1-t0):.1f}s:")
        for i, sg in enumerate(subgoals, 1):
            print(f"     {i}. {sg}")
    except Exception as e:
        print(f"  ✗ Planning failed: {e}")
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Initial Screen Analysis
    # ══════════════════════════════════════════════════════════════════════
    print_header("Phase 3: Initial Screen Analysis")

    print("\n  Scanning current screen with OmniParser...")
    t0 = time.time()
    try:
        elements = eyes.scan(force=True)
        t1 = time.time()
        print(f"  ✓ Found {len(elements)} UI elements in {(t1-t0)*1000:.0f}ms")

        # Show summary
        type_counts = {}
        for e in elements:
            type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1

        print(f"  Element types: {dict(type_counts)}")

        # Show some sample text elements
        text_elements = [e for e in elements if e.label and len(e.label) > 2][:10]
        if text_elements:
            print(f"  Sample labels: {[e.label for e in text_elements[:5]]}")
    except Exception as e:
        print(f"  ✗ Screen scan failed: {e}")
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Execute Pipeline
    # ══════════════════════════════════════════════════════════════════════
    print_header("Phase 4: Pipeline Execution")

    print("\n  Starting pipeline execution...")
    print("  Press Ctrl+C to abort\n")

    # Import pipeline
    from agent.pipeline import Pipeline

    # Create status trackers
    executed_steps = []
    start_time = time.time()

    def on_status(msg: str):
        print(f"  Status: {msg}")

    def on_step(step_num: int, msg: str, status: str):
        executed_steps.append((step_num, msg, status))
        print_step(step_num, msg, status)

    # Create and run pipeline
    try:
        pipeline = Pipeline(
            on_status=on_status,
            on_step=on_step,
            use_caption_model=False,
            preload_models=False  # Already loaded
        )

        # Inject pre-loaded components
        pipeline._eyes = eyes
        pipeline._hands = hands
        pipeline._verifier = verifier

        print(f"\n  Executing {len(subgoals)} subgoals...")
        print("-" * 60)

        result = pipeline.run_task(subgoals)

        print("-" * 60)
        elapsed = time.time() - start_time
        print(f"\n  {result}")
        print(f"  Total execution time: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n\n  ⚠ Execution aborted by user")
        return False
    except Exception as e:
        print(f"\n  ✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Results Summary
    # ══════════════════════════════════════════════════════════════════════
    print_header("Phase 5: Results Summary")

    # Pipeline stats
    stats = pipeline.get_stats()
    print(f"\n  Pipeline Statistics:")
    print(f"    Total subgoals:  {stats['total_subgoals']}")
    print(f"    Successful:      {stats['successful']}")
    print(f"    Failed:          {stats['failed']}")
    print(f"    Retries:         {stats['total_retries']}")
    print(f"    Success rate:    {stats['success_rate']:.1f}%")

    # Verifier stats
    v_stats = verifier.get_stats()
    print(f"\n  Verifier Statistics:")
    print(f"    Total checks:    {v_stats['total_checks']}")
    print(f"    Changes found:   {v_stats['changes_detected']}")
    print(f"    No changes:      {v_stats['no_changes_detected']}")

    # Check audit log
    from pathlib import Path
    audit_path = Path("audit_log.jsonl")
    if audit_path.exists():
        with open(audit_path, 'r') as f:
            audit_lines = f.readlines()
        print(f"\n  Audit log: {len(audit_lines)} entries in {audit_path}")

    # Final verdict
    print("\n" + "=" * 70)
    if stats['successful'] == stats['total_subgoals']:
        print("  ✓ TEST PASSED - All subgoals completed successfully!")
    elif stats['successful'] > 0:
        print(f"  ⚠ TEST PARTIAL - {stats['successful']}/{stats['total_subgoals']} subgoals completed")
    else:
        print("  ✗ TEST FAILED - No subgoals completed")
    print("=" * 70 + "\n")

    return stats['successful'] > 0


def test_single_subgoal_demo():
    """Quick demo: test a single subgoal to verify pipeline works."""
    print_header("Quick Demo: Single Subgoal Test")

    print("\n  This tests a single simple action to verify the pipeline...")

    from agent.pipeline import Pipeline

    def on_status(msg):
        print(f"    Status: {msg}")

    def on_step(num, msg, status):
        icons = {"running": "◌", "done": "●", "failed": "✗"}
        print(f"    {icons.get(status, '○')} {msg}")

    pipeline = Pipeline(
        on_status=on_status,
        on_step=on_step,
        preload_models=True
    )

    # Test simple subgoal
    test_subgoals = ["wait for 2 seconds"]
    print(f"\n  Subgoals: {test_subgoals}")

    result = pipeline.run_task(test_subgoals)
    print(f"\n  Result: {result}")

    return "Successfully" in result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" VOXCODE - Plan1.md Architecture Integration Test ".center(70))
    print("=" * 70)

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick demo mode
        success = test_single_subgoal_demo()
    else:
        # Full workflow test
        print("\n  This test will:")
        print("  1. Use Groq to plan subgoals for 'open youtube and play arijit singh songs'")
        print("  2. Execute each subgoal using OmniParser + Qwen/Groq")
        print("  3. Verify each action with pixel diff")
        print("\n  The LLM will autonomously handle:")
        print("  - Chrome profile selection (if it appears)")
        print("  - YouTube navigation")
        print("  - Search and video selection")
        print("\n  Press Enter to start, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            sys.exit(0)

        success = test_full_workflow()

    sys.exit(0 if success else 1)
