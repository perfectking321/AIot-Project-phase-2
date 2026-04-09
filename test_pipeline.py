"""
Test Pipeline module - Full 3-model orchestration.
Run: python test_pipeline.py

Prerequisites:
1. Ollama running: ollama serve
2. Model available: ollama pull qwen2.5:7b-instruct-q4_K_M (or any qwen)
"""
import sys
import time


def test_pipeline():
    print("=" * 60)
    print("VOXCODE Pipeline Test")
    print("=" * 60)

    # Test 1: Import
    print("\n[1] Testing Pipeline import...")
    try:
        from agent.pipeline import Pipeline, get_pipeline
        print("    OK - Pipeline module imported")
    except ImportError as e:
        print(f"    FAIL - Import error: {e}")
        return False

    # Test 2: Create Pipeline (this loads all components)
    print("\n[2] Creating Pipeline (loading Eyes, Hands, Verifier)...")
    t0 = time.time()
    try:
        # Use callbacks to track progress
        status_msgs = []
        step_msgs = []

        def on_status(msg):
            status_msgs.append(msg)
            print(f"    Status: {msg}")

        def on_step(step_num, msg, status):
            step_msgs.append((step_num, msg, status))
            icon = {"running": "◌", "done": "●", "failed": "✗"}.get(status, "○")
            print(f"    {icon} Step {step_num}: {msg}")

        pipeline = Pipeline(
            on_status=on_status,
            on_step=on_step,
            use_caption_model=False,
            preload_models=True
        )
        t1 = time.time()
        print(f"    OK - Pipeline created in {(t1-t0):.1f}s")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Check if Ollama is available
    print("\n[3] Checking Ollama connection...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            print(f"    OK - Ollama available, models: {models[:3]}...")
            ollama_ok = True
        else:
            print(f"    WARN - Ollama status {resp.status_code}")
            ollama_ok = False
    except Exception as e:
        print(f"    WARN - Ollama not reachable: {e}")
        print("    Skipping inference tests (Ollama required)")
        ollama_ok = False

    # Test 4: Test Eyes component directly
    print("\n[4] Testing Eyes scan...")
    t0 = time.time()
    try:
        eyes = pipeline._get_eyes()
        elements = eyes.scan(force=True)
        t1 = time.time()
        print(f"    OK - Scan: {len(elements)} elements in {(t1-t0)*1000:.0f}ms")
    except Exception as e:
        print(f"    FAIL - {e}")
        return False

    # Test 5: Test token compression
    print("\n[5] Testing token compression...")
    filtered = eyes.filter_near(elements, 960, 540, radius=400)
    full_str = eyes.elements_to_prompt_str(elements)
    filtered_str = eyes.elements_to_prompt_str(filtered)
    print(f"    Full: {len(elements)} elements (~{len(full_str.split())*1.3:.0f} tokens)")
    print(f"    Filtered: {len(filtered)} elements (~{len(filtered_str.split())*1.3:.0f} tokens)")

    # Test 6: Test Verifier component
    print("\n[6] Testing Verifier capture...")
    try:
        verifier = pipeline._get_verifier()
        region = verifier.capture_region(960, 540)
        print(f"    OK - Captured region shape: {region.shape}")
    except Exception as e:
        print(f"    FAIL - {e}")

    if not ollama_ok:
        print("\n[SKIP] Skipping full pipeline test (Ollama not available)")
        print("       Start Ollama with: ollama serve")
        print("       Then run: python test_pipeline.py")
        print("\n" + "=" * 60)
        print("Partial test complete (Eyes + Verifier OK, Hands requires Ollama)")
        print("=" * 60)
        return True

    # Test 7: Test single subgoal execution
    print("\n[7] Testing single subgoal execution...")
    print("    Subgoal: 'move mouse to center of screen'")
    t0 = time.time()
    try:
        success, _ = pipeline.run_subgoal(
            subgoal="move mouse to center of screen",
            step_num=1,
            total_steps=1,
            prefetched_elements=None
        )
        t1 = time.time()
        print(f"    Result: {'OK' if success else 'FAIL'} in {(t1-t0):.1f}s")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()

    # Test 8: Test multi-subgoal pipeline
    print("\n[8] Testing multi-subgoal pipeline...")
    test_subgoals = [
        "wait for 1 second",
        "task is complete"
    ]
    print(f"    Subgoals: {test_subgoals}")

    t0 = time.time()
    try:
        result = pipeline.run_task(test_subgoals)
        t1 = time.time()
        print(f"    Result: {result}")
        print(f"    Total time: {(t1-t0):.1f}s")
    except Exception as e:
        print(f"    FAIL - {e}")
        import traceback
        traceback.print_exc()

    # Test 9: Pipeline statistics
    print("\n[9] Pipeline statistics...")
    stats = pipeline.get_stats()
    print(f"    Total subgoals: {stats['total_subgoals']}")
    print(f"    Successful: {stats['successful']}")
    print(f"    Failed: {stats['failed']}")
    print(f"    Retries: {stats['total_retries']}")
    print(f"    Success rate: {stats['success_rate']:.1f}%")

    # Cleanup
    print("\n[10] Shutting down pipeline...")
    pipeline.shutdown()
    print("    OK - Pipeline shutdown")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
