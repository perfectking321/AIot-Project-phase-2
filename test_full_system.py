"""
VOXCODE v2.0 Full System Test
Tests all components: Perception, Planning, Skills, Memory, Agent
"""

import logging
import time
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_full_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def print_header(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    print()
    print(f"[{text}]")
    print("-" * 50)

def test_full_system():
    """Test the complete VOXCODE v2.0 system."""
    print_header("VOXCODE v2.0 FULL SYSTEM TEST")

    results = {
        "perception": False,
        "planning": False,
        "skills": False,
        "memory": False,
        "agent": False
    }

    # ==================== Test 1: Perception ====================
    print_section("1. PERCEPTION ENGINE")

    try:
        from perception.vlm import VisionLanguageModel, get_vlm
        from perception.screen_state import ScreenState, ScreenStateParser
        from perception.grounder import get_grounder
        print("    [OK] Perception modules imported")

        vlm = get_vlm()
        if vlm.is_available():
            print(f"    [OK] VLM available")
        else:
            print("    [WARN] VLM not available (install qwen2.5vl:7b)")

        grounder = get_grounder()
        print("    [OK] Grounder initialized")

        results["perception"] = True

    except Exception as e:
        print(f"    [ERROR] Perception failed: {e}")

    # ==================== Test 2: Planning ====================
    print_section("2. HIERARCHICAL PLANNER")

    try:
        from brain.planner import HierarchicalPlanner, TaskPlan, get_planner
        print("    [OK] Planner modules imported")

        planner = get_planner()
        print("    [OK] Planner initialized")

        # Quick test
        plan = planner.create_plan(
            "Test task",
            "Desktop visible, no apps open"
        )
        print(f"    [OK] Created test plan with {len(plan.subtasks)} subtasks")

        results["planning"] = True

    except Exception as e:
        print(f"    [ERROR] Planning failed: {e}")

    # ==================== Test 3: Skills ====================
    print_section("3. SKILL SYSTEM")

    try:
        from agent.skills.base import get_registry
        print("    [OK] Skills modules imported")

        registry = get_registry()
        skills = registry.list_skills()
        print(f"    [OK] Registry has {len(skills)} skills:")

        for skill in skills[:5]:
            print(f"        - {skill['name']}")
        if len(skills) > 5:
            print(f"        ... and {len(skills) - 5} more")

        results["skills"] = True

    except Exception as e:
        print(f"    [ERROR] Skills failed: {e}")

    # ==================== Test 4: Memory ====================
    print_section("4. MEMORY SYSTEM")

    try:
        from memory.manager import get_memory
        print("    [OK] Memory modules imported")

        memory = get_memory()
        print("    [OK] Memory manager initialized")

        # Test operations
        memory.start_goal("Test goal")
        memory.record_action("Test action", success=True)
        memory.app_opened("TestApp")
        memory.complete_goal(success=True)

        print(f"    [OK] Memory operations working")
        print(f"    [OK] Episodic memory: {len(memory.episodic)} events")

        results["memory"] = True

    except Exception as e:
        print(f"    [ERROR] Memory failed: {e}")

    # ==================== Test 5: Integrated Agent ====================
    print_section("5. INTELLIGENT AGENT")

    try:
        from agent.intelligent_agent import IntelligentAgent, create_agent
        print("    [OK] Agent modules imported")

        def on_msg(msg):
            print(f"        > {msg}")

        def on_state(state):
            print(f"        [STATE: {state}]")

        agent = create_agent(
            on_message=on_msg,
            on_state_change=on_state,
            max_iterations=5
        )
        print("    [OK] Agent created")

        # Test perception
        print("    Testing perception...")
        import pyautogui
        screen_state = agent.perceive()
        print(f"    [OK] Perceived screen: {screen_state.describe()}")

        results["agent"] = True

    except Exception as e:
        print(f"    [ERROR] Agent failed: {e}")
        import traceback
        traceback.print_exc()

    # ==================== Test 6: Dependencies Check ====================
    print_section("6. DEPENDENCIES CHECK")

    deps = {
        "pyautogui": False,
        "PIL": False,
        "requests": False,
        "textual": False,
        "whisper": False,
    }

    for dep in deps:
        try:
            __import__(dep if dep != "PIL" else "PIL.Image")
            deps[dep] = True
            print(f"    [OK] {dep}")
        except ImportError:
            print(f"    [MISSING] {dep}")

    # ==================== Test 7: Ollama Models ====================
    print_section("7. OLLAMA MODELS")

    try:
        from brain.llm import OllamaClient
        llm = OllamaClient()

        if llm.is_available():
            models = llm.list_models()
            print(f"    [OK] Ollama available with {len(models)} models:")

            for model in models[:5]:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)
                print(f"        - {name} ({size:.1f}GB)")

            # Check for required models
            model_names = [m.get("name", "").lower() for m in models]

            required = ["qwen2.5", "qwen2.5vl", "minicpm"]
            for req in required:
                found = any(req in name for name in model_names)
                status = "[OK]" if found else "[MISSING]"
                print(f"    {status} {req}")

        else:
            print("    [ERROR] Ollama not running!")
            print("    Start with: ollama serve")

    except Exception as e:
        print(f"    [ERROR] Ollama check failed: {e}")

    # ==================== Summary ====================
    print_header("TEST SUMMARY")

    all_passed = all(results.values())

    for component, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {component.capitalize()}")

    print()
    if all_passed:
        print("    ALL TESTS PASSED! System is ready.")
    else:
        print("    Some tests failed. Check the errors above.")

    print()
    print("=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    if not results["perception"]:
        print("    - Check perception module imports")

    # Always show VLM recommendation
    print()
    print("    For best performance, install VLM model:")
    print("      ollama pull qwen2.5vl:7b")
    print()
    print("    Or lighter alternative:")
    print("      ollama pull minicpm-v:8b")

    print()
    print("=" * 70)


def test_interactive():
    """Interactive test with real screen automation."""
    print_header("INTERACTIVE TEST")
    print()
    print("This will test the agent with a real task.")
    print("Make sure you have:")
    print("  - Ollama running (ollama serve)")
    print("  - A model installed (qwen2.5:7b or better)")
    print()

    try:
        from agent.intelligent_agent import create_agent

        def on_msg(msg):
            print(f"  {msg}")

        def on_state(state):
            pass  # Silent state changes

        agent = create_agent(
            on_message=on_msg,
            on_state_change=on_state,
            max_iterations=10
        )

        print("Agent ready. Enter a goal to test (or 'quit' to exit):")
        print("Examples:")
        print("  - Open notepad")
        print("  - Open chrome and go to google.com")
        print("  - Open calculator")
        print()

        while True:
            try:
                goal = input("Goal> ").strip()

                if not goal or goal.lower() in ['quit', 'exit', 'q']:
                    break

                print()
                print(f"Processing: {goal}")
                print("-" * 50)

                result = agent.process_goal(goal)

                print("-" * 50)
                print(f"Result: {result}")
                print()

            except KeyboardInterrupt:
                print("\nInterrupted")
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        test_interactive()
    else:
        test_full_system()
