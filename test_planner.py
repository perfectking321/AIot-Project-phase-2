"""
Test Hierarchical Planner
Tests task decomposition and planning.
"""

import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_planner():
    """Test the hierarchical planner."""
    print("=" * 60)
    print("HIERARCHICAL PLANNER TEST")
    print("=" * 60)
    print()

    # Test 1: Check imports
    print("[1] Checking imports...")

    try:
        from brain.planner import (
            HierarchicalPlanner, TaskPlan, Subtask, TaskStatus, get_planner
        )
        print("    [OK] Planner modules imported")
    except ImportError as e:
        print(f"    [ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Test 2: Check LLM availability
    print("[2] Checking LLM availability...")

    try:
        from brain.llm import OllamaClient
        llm = OllamaClient()
        if llm.is_available():
            print(f"    [OK] LLM available: {llm.model}")
        else:
            print("    [WARN] LLM not available - tests will use fallback")
    except Exception as e:
        print(f"    [WARN] LLM check failed: {e}")

    print()

    # Test 3: Create a simple plan
    print("[3] Testing plan creation...")

    try:
        planner = get_planner()

        # Test with a simple goal
        goal = "Open Notepad and type 'Hello World'"

        screen_context = """
        Active App: Windows Desktop
        Visible: Taskbar, Desktop icons
        No applications currently open
        """

        print(f"    Goal: {goal}")
        print("    Creating plan...")

        plan = planner.create_plan(goal, screen_context)

        print(f"    [OK] Plan created with {len(plan.subtasks)} subtasks")
        print()
        print("    --- Plan Details ---")
        print(planner.describe_plan(plan))
        print("    --------------------")

    except Exception as e:
        print(f"    [ERROR] Plan creation failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 4: Create a complex plan
    print("[4] Testing complex plan...")

    try:
        complex_goal = "Open WhatsApp and send 'Hello' to John"

        screen_context = """
        Active App: WhatsApp Desktop
        App State: Chat list visible
        Visible chats: John, Mary, Work Group
        No chat currently open
        """

        print(f"    Goal: {complex_goal}")
        print(f"    Context: WhatsApp is already open")
        print("    Creating plan...")

        plan = planner.create_plan(complex_goal, screen_context)

        print(f"    [OK] Plan created with {len(plan.subtasks)} subtasks")
        print()
        print("    --- Plan Details ---")
        print(planner.describe_plan(plan))
        print("    --------------------")

    except Exception as e:
        print(f"    [ERROR] Complex plan failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 5: Test plan operations
    print("[5] Testing plan operations...")

    try:
        # Create a manual plan
        plan = TaskPlan(goal="Test plan operations")

        plan.add_subtask(Subtask(
            id=1,
            description="First task",
            action_type="test",
            preconditions=["nothing"],
            postconditions=["first_done"]
        ))

        plan.add_subtask(Subtask(
            id=2,
            description="Second task",
            action_type="test",
            preconditions=["first_done"],
            postconditions=["second_done"]
        ))

        print(f"    Plan has {len(plan.subtasks)} tasks")

        # Test current task
        current = plan.get_current_task()
        print(f"    Current task: {current.description}")

        # Test advancing
        current.status = TaskStatus.COMPLETED
        plan.advance()
        current = plan.get_current_task()
        print(f"    After advance: {current.description}")

        # Test completion check
        current.status = TaskStatus.COMPLETED
        print(f"    Is complete: {plan.is_complete()}")

        print("    [OK] Plan operations working")

    except Exception as e:
        print(f"    [ERROR] Plan operations failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 6: Test replanning
    print("[6] Testing replanning...")

    try:
        original = TaskPlan(goal="Open Chrome and search for 'cats'")
        original.add_subtask(Subtask(
            id=1,
            description="Open Chrome",
            action_type="open_app",
            status=TaskStatus.COMPLETED
        ))
        original.add_subtask(Subtask(
            id=2,
            description="Click search box",
            action_type="click_element",
            status=TaskStatus.FAILED
        ))

        failed_task = original.subtasks[1]

        new_plan = planner.replan(
            original_plan=original,
            failed_task=failed_task,
            failure_reason="Search box not found on screen",
            screen_context="Chrome is open but showing a different page"
        )

        print(f"    [OK] Replanned with {len(new_plan.subtasks)} new subtasks")

    except Exception as e:
        print(f"    [WARN] Replanning test: {e}")

    print()
    print("=" * 60)
    print("PLANNER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_planner()
