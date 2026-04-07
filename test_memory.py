"""
Test Memory System
Tests episodic and working memory.
"""

import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_memory():
    """Test the memory system."""
    print("=" * 60)
    print("MEMORY SYSTEM TEST")
    print("=" * 60)
    print()

    # Test 1: Check imports
    print("[1] Checking imports...")

    try:
        from memory.episodic import EpisodicMemory, EventType, Episode
        from memory.working import WorkingMemory, AppStatus, AppMemory
        from memory.manager import MemoryManager, get_memory
        print("    [OK] All memory modules imported")
    except ImportError as e:
        print(f"    [ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Test 2: Test episodic memory
    print("[2] Testing episodic memory...")

    try:
        episodic = EpisodicMemory()

        # Add some events
        episodic.add(EventType.GOAL_STARTED, "Open WhatsApp and send hello")
        episodic.add(EventType.APP_OPENED, "Opened WhatsApp", data={"app": "WhatsApp"})
        episodic.add(EventType.ACTION_SUCCEEDED, "Clicked on chat", success=True)
        episodic.add(EventType.ACTION_FAILED, "Could not find send button", success=False)
        episodic.add(EventType.GOAL_COMPLETED, "Task completed", success=True)

        print(f"    [OK] Added {len(episodic)} episodes")

        # Test retrieval
        recent = episodic.get_recent(3)
        print(f"    [OK] Retrieved {len(recent)} recent episodes")

        failures = episodic.get_failures()
        print(f"    [OK] Found {len(failures)} failures")

        # Test search
        results = episodic.search("WhatsApp")
        print(f"    [OK] Search 'WhatsApp' found {len(results)} results")

        # Test context summary
        summary = episodic.get_context_summary()
        print(f"    [OK] Context summary:\n{summary}")

    except Exception as e:
        print(f"    [ERROR] Episodic memory failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 3: Test working memory
    print("[3] Testing working memory...")

    try:
        working = WorkingMemory()

        # Start a task
        task = working.start_task("Send message to John", total_subtasks=3)
        print(f"    [OK] Started task: {task.goal}")

        # Update progress
        working.update_subtask("Opening WhatsApp", index=1)
        working.record_action("Opened WhatsApp")

        # Track app state
        working.update_app("WhatsApp", status=AppStatus.FOCUSED, details="Chat list visible")
        working.update_app("Chrome", status=AppStatus.BACKGROUND)

        print(f"    [OK] Task progress: {task.progress_text()}")
        print(f"    [OK] Actions taken: {task.actions_taken}")

        # Check app states
        open_apps = working.get_open_apps()
        print(f"    [OK] Open apps: {[a.name for a in open_apps]}")

        focused = working.get_focused_app()
        print(f"    [OK] Focused app: {focused.name if focused else 'None'}")

        # Get context
        context = working.get_context()
        print(f"    [OK] Working memory context:\n{context}")

    except Exception as e:
        print(f"    [ERROR] Working memory failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 4: Test memory manager
    print("[4] Testing memory manager...")

    try:
        memory = get_memory()

        # Start a goal
        memory.start_goal("Test automation task", subtasks=2)

        # Record actions
        memory.record_action("Step 1 completed", success=True)
        memory.app_opened("Notepad", details="Empty document")
        memory.record_action("Typed hello world", success=True)

        # Add observation
        memory.observe("Notepad window is now visible with text")

        # Get full context
        context = memory.get_full_context()
        print(f"    [OK] Full context:\n{context}")

        # Complete goal
        memory.complete_goal(success=True, message="Task completed successfully")

        # Check history
        history = memory.get_relevant_history("Notepad")
        print(f"    [OK] Relevant history:\n{history}")

    except Exception as e:
        print(f"    [ERROR] Memory manager failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 5: Test persistence (optional)
    print("[5] Testing persistence...")

    try:
        import tempfile
        import os

        # Create temporary file for testing
        temp_file = os.path.join(tempfile.gettempdir(), "voxcode_memory_test.json")

        # Create memory with persistence
        mem = EpisodicMemory(persist_path=temp_file)
        mem.add(EventType.GOAL_STARTED, "Persistent test")
        mem.add(EventType.GOAL_COMPLETED, "Test done", success=True)
        mem.save()

        # Load in new instance
        mem2 = EpisodicMemory(persist_path=temp_file)
        print(f"    [OK] Saved and loaded {len(mem2)} episodes")

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        print(f"    [WARN] Persistence test: {e}")

    print()
    print("=" * 60)
    print("MEMORY SYSTEM TEST COMPLETE")
    print("=" * 60)
    print()
    print("Memory system capabilities:")
    print("  - Episodic: Long-term event history with search")
    print("  - Working: Current task and app state tracking")
    print("  - Manager: Unified interface for both")
    print("  - Persistence: Optional save/load to JSON")


if __name__ == "__main__":
    test_memory()
