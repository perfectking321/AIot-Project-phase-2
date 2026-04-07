"""
Test Skill System
Tests the skill registry and individual skills.
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def test_skills():
    """Test the skill system."""
    print("=" * 60)
    print("SKILL SYSTEM TEST")
    print("=" * 60)
    print()

    # Test 1: Check imports
    print("[1] Checking imports...")

    try:
        from agent.skills.base import (
            Skill, SkillResult, SkillStatus, SkillRegistry, get_registry
        )
        from agent.skills.app_skills import OpenAppSkill, CloseAppSkill
        from agent.skills.messaging_skills import SendMessageSkill, OpenChatSkill
        from agent.skills.browser_skills import NavigateToUrlSkill, SearchWebSkill
        from agent.skills.input_skills import TypeTextSkill, ClickElementSkill, ScrollSkill
        print("    [OK] All skill modules imported")
    except ImportError as e:
        print(f"    [ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Test 2: Test skill registry
    print("[2] Testing skill registry...")

    try:
        registry = get_registry()
        skills = registry.list_skills()

        print(f"    [OK] Registry has {len(skills)} skills:")
        for skill in skills:
            print(f"         - {skill['name']}: {skill['description']}")

    except Exception as e:
        print(f"    [ERROR] Registry failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 3: Test skill instantiation
    print("[3] Testing skill instantiation...")

    try:
        # Get a skill by name
        open_app = registry.get("open_app")
        if open_app:
            print(f"    [OK] Got skill: {open_app.name}")
            print(f"         Params: {open_app.params}")
            print(f"         Preconditions: {open_app.preconditions}")
            print(f"         Postconditions: {open_app.postconditions}")
        else:
            print("    [ERROR] Could not get open_app skill")

    except Exception as e:
        print(f"    [ERROR] Instantiation failed: {e}")

    print()

    # Test 4: Test skill result creation
    print("[4] Testing SkillResult...")

    try:
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            message="Test completed",
            steps_completed=3,
            steps_total=3
        )
        print(f"    [OK] Created result: success={result.success}")

        result2 = SkillResult(
            status=SkillStatus.FAILED,
            message="Test failed"
        )
        print(f"    [OK] Created failed result: success={result2.success}")

    except Exception as e:
        print(f"    [ERROR] SkillResult failed: {e}")

    print()

    # Test 5: Test action type mapping
    print("[5] Testing action type mapping...")

    try:
        action_types = [
            "open_app", "send_message", "navigate_to", "click_element", "type_text"
        ]

        for action in action_types:
            skill_name = registry.find_skill_for_action(action)
            if skill_name:
                print(f"    [OK] {action} -> {skill_name}")
            else:
                print(f"    [WARN] No skill for: {action}")

    except Exception as e:
        print(f"    [ERROR] Mapping failed: {e}")

    print()

    # Test 6: Test skill execution (dry run - no actual execution)
    print("[6] Testing skill structure...")

    try:
        # Test browser skill
        nav_skill = registry.get("navigate_url")
        if nav_skill:
            print(f"    [OK] navigate_url skill:")
            print(f"         Name: {nav_skill.name}")
            print(f"         Params: {nav_skill.params}")

        # Test messaging skill
        msg_skill = registry.get("send_message")
        if msg_skill:
            print(f"    [OK] send_message skill:")
            print(f"         Name: {msg_skill.name}")
            print(f"         Params: {msg_skill.params}")
            print(f"         Preconditions: {msg_skill.preconditions}")

        # Test input skill
        type_skill = registry.get("type_text")
        if type_skill:
            print(f"    [OK] type_text skill:")
            print(f"         Name: {type_skill.name}")

    except Exception as e:
        print(f"    [ERROR] Skill structure test failed: {e}")

    print()
    print("=" * 60)
    print("SKILL SYSTEM TEST COMPLETE")
    print("=" * 60)
    print()
    print("Available skills for automation:")
    for skill in registry.list_skills():
        print(f"  - {skill['name']}: {skill['description']}")


if __name__ == "__main__":
    test_skills()
