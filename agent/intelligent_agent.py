"""
VOXCODE Intelligent Agent v2.0
Integrated agent with perception, planning, skills, and memory.
"""

import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("voxcode.agent.v2")

try:
    from PIL import Image
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from config import config

# Import new components
from brain.llm import get_llm_client
from brain.planner import HierarchicalPlanner, TaskPlan, Subtask, TaskStatus, get_planner
from memory.manager import MemoryManager, get_memory
from memory.working import AppStatus
from agent.tools import WindowsTools

# Perception components (optional - may not be fully set up)
try:
    from perception.vlm import VisionLanguageModel, get_vlm
    from perception.screen_state import ScreenState, SemanticState, ScreenStateParser
    from perception.grounder import ElementGrounder, get_grounder
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    logger.warning("Perception module not fully available")

# Skills
try:
    from agent.skills.base import SkillRegistry, get_registry, SkillResult, SkillStatus as SkillExecStatus
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    logger.warning("Skills module not available")


@dataclass
class AgentState:
    """Current state of the agent."""
    phase: str = "idle"  # idle, perceiving, planning, executing, verifying
    current_goal: str = ""
    current_subtask: str = ""
    iteration: int = 0
    max_iterations: int = 20


class IntelligentAgent:
    """
    VOXCODE Intelligent Agent v2.0

    Integrates:
    - Perception (VLM for semantic understanding)
    - Planning (Hierarchical task decomposition)
    - Skills (Pre-defined action sequences)
    - Memory (Episodic + Working memory)
    - Tools (Low-level Windows automation)
    """

    def __init__(
        self,
        on_message: Callable[[str], None] = None,
        on_state_change: Callable[[str], None] = None,
        max_iterations: int = None
    ):
        """
        Initialize the intelligent agent.

        Args:
            on_message: Callback for status messages
            on_state_change: Callback for state changes
            max_iterations: Maximum iterations per goal
        """
        self.on_message = on_message or (lambda m: None)
        self.on_state_change = on_state_change or (lambda s: None)

        # Core components
        self.llm = get_llm_client()
        self.tools = WindowsTools(use_omniparser=True)
        self.planner = get_planner()
        self.memory = get_memory()

        # Perception (optional)
        self.vlm = get_vlm() if PERCEPTION_AVAILABLE else None
        self.grounder = get_grounder() if PERCEPTION_AVAILABLE else None

        # Skills
        self.skills = get_registry() if SKILLS_AVAILABLE else None

        # State
        self.state = AgentState(
            max_iterations=max_iterations or config.agent.reactive_max_iterations
        )
        self.is_running = False

        logger.info("IntelligentAgent v2.0 initialized")
        logger.info(f"  - Perception: {'Available' if PERCEPTION_AVAILABLE else 'Not available'}")
        logger.info(f"  - Skills: {'Available' if SKILLS_AVAILABLE else 'Not available'}")
        logger.info(f"  - VLM: {'Available' if self.vlm and self.vlm.is_available() else 'Not available'}")

    def _set_phase(self, phase: str):
        """Update agent phase."""
        self.state.phase = phase
        self.on_state_change(phase)

    def _msg(self, msg: str):
        """Send a message."""
        self.on_message(msg)
        logger.info(msg)

    # ==================== Perception ====================

    def perceive(self) -> ScreenState:
        """
        Perceive the current screen state.

        Returns:
            ScreenState with element detection and semantic understanding
        """
        self._set_phase("perceiving")

        screen_state = ScreenState(timestamp=time.time())

        try:
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            screen_state.screenshot = screenshot

            # Get active window
            try:
                win = pyautogui.getActiveWindow()
                if win:
                    screen_state.active_window = win.title
            except:
                pass

            # OmniParser element detection
            try:
                result = self.tools.parse_screen()
                if result.success:
                    screen_state.elements = result.data.get("elements", [])
                    screen_state.element_count = result.data.get("count", 0)
            except Exception as e:
                logger.warning(f"Element detection failed: {e}")

            # VLM semantic understanding (if available)
            if self.vlm and self.vlm.is_available() and config.perception.use_vlm:
                try:
                    vlm_response = self.vlm.understand_screen(screenshot)
                    if vlm_response.success:
                        semantic = ScreenStateParser.parse_vlm_response(vlm_response.content)
                        screen_state.merge_semantic(semantic)

                        # Update memory with app states
                        if semantic.visible_apps:
                            for app in semantic.visible_apps:
                                status = AppStatus.FOCUSED if app == semantic.active_app else AppStatus.OPEN
                                self.memory.update_app_state(app.name, status=status, details=app.details)
                except Exception as e:
                    logger.warning(f"VLM perception failed: {e}")

            # Record observation
            self.memory.observe(screen_state.describe())

        except Exception as e:
            logger.error(f"Perception failed: {e}")

        return screen_state

    # ==================== Planning ====================

    def plan(self, goal: str, screen_state: ScreenState) -> TaskPlan:
        """
        Create a plan for the goal.

        Args:
            goal: The goal to achieve
            screen_state: Current screen state

        Returns:
            TaskPlan with subtasks
        """
        self._set_phase("planning")
        self._msg("Creating plan...")

        # Get full context for planning
        context = self._build_context(screen_state)

        # Create plan
        plan = self.planner.create_plan(goal, context)

        # Log plan
        self._msg(f"Plan: {len(plan.subtasks)} subtasks")
        for task in plan.subtasks:
            self._msg(f"  [{task.id}] {task.description}")

        # Start goal in memory
        self.memory.start_goal(goal, subtasks=len(plan.subtasks))

        return plan

    def _build_context(self, screen_state: ScreenState) -> str:
        """Build context string for planning/reasoning."""
        parts = []

        # Screen state
        parts.append(screen_state.to_prompt_context())

        # Memory context
        parts.append("")
        parts.append(self.memory.get_full_context())

        return "\n".join(parts)

    # ==================== Execution ====================

    def execute_subtask(self, subtask: Subtask, screen_state: ScreenState) -> bool:
        """
        Execute a single subtask.

        Args:
            subtask: The subtask to execute
            screen_state: Current screen state

        Returns:
            True if successful
        """
        self._set_phase("executing")
        self._msg(f"Executing: {subtask.description}")

        # Update memory
        self.memory.update_progress(subtask.description, subtask.id)

        success = False

        # Try to use a skill if available
        if self.skills:
            skill_name = self.skills.find_skill_for_action(subtask.action_type)
            if skill_name:
                success = self._execute_with_skill(skill_name, subtask)

        # Fall back to direct tool execution
        if not success:
            success = self._execute_with_tools(subtask, screen_state)

        # Record result
        self.memory.record_action(
            subtask.description,
            success=success,
            data=subtask.params
        )

        return success

    def _execute_with_skill(self, skill_name: str, subtask: Subtask) -> bool:
        """Execute subtask using a skill."""
        skill = self.skills.get(skill_name, tools=self.tools)
        if not skill:
            return False

        try:
            self._msg(f"  Using skill: {skill_name}")
            result = skill.execute(**subtask.params)

            if result.success:
                self._msg(f"  [OK] {result.message}")
                return True
            else:
                self._msg(f"  [FAIL] {result.message}")
                return False

        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            return False

    def _execute_with_tools(self, subtask: Subtask, screen_state: ScreenState) -> bool:
        """Execute subtask using direct tool calls."""
        action_type = subtask.action_type
        params = subtask.params

        try:
            # Map action types to tool methods
            if action_type == "open_app" or action_type == "ensure_app_open":
                result = self.tools.open_application(params.get("app_name", ""))
            elif action_type == "click_element":
                result = self.tools.click_text(params.get("element_description", ""))
            elif action_type == "type_text":
                result = self.tools.type_text(params.get("text", ""))
            elif action_type == "send_message":
                # Click input, type, send
                self.tools.click_text("Type a message")
                time.sleep(0.3)
                self.tools.type_text(params.get("message", ""))
                time.sleep(0.2)
                result = self.tools.press_key("enter")
            elif action_type == "navigate_to":
                # Use hotkey to focus address bar, then type
                self.tools.hotkey("ctrl", "l")
                time.sleep(0.3)
                self.tools.type_text(params.get("destination", ""))
                result = self.tools.press_key("enter")
            elif action_type == "search":
                # Type in search and enter
                self.tools.type_text(params.get("query", ""))
                result = self.tools.press_key("enter")
            elif action_type == "wait":
                result = self.tools.wait(params.get("seconds", 1))
            elif action_type == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 3)
                scroll_val = amount if direction == "up" else -amount
                result = self.tools.scroll(scroll_val)
            elif action_type == "verify":
                # Verification is handled separately
                return True
            else:
                # Try to interpret the description
                self._msg(f"  Unknown action type: {action_type}, attempting interpretation...")
                return self._interpret_and_execute(subtask.description)

            return result.success if hasattr(result, 'success') else True

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return False

    def _interpret_and_execute(self, description: str) -> bool:
        """Use LLM to interpret and execute an unclear action."""
        prompt = f"""Convert this action description to a single tool call.

ACTION: {description}

AVAILABLE TOOLS:
- click_text(text): Click on visible text
- type_text(text): Type text
- press_key(key): Press a key (enter, tab, escape, etc.)
- hotkey(key1, key2, ...): Press key combo (ctrl+c, alt+tab, etc.)
- open_application(name): Open an app
- scroll(amount): Scroll (positive=up, negative=down)
- wait(seconds): Wait

Respond with ONLY JSON:
{{"tool": "tool_name", "params": {{"key": "value"}}}}"""

        try:
            response = self.llm.generate(prompt)

            import json
            content = response.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                tool_name = data.get("tool")
                params = data.get("params", {})

                tool_method = getattr(self.tools, tool_name, None)
                if tool_method:
                    result = tool_method(**params)
                    return result.success if hasattr(result, 'success') else True

        except Exception as e:
            logger.error(f"Interpretation failed: {e}")

        return False

    # ==================== Verification ====================

    def verify(self, subtask: Subtask, before: ScreenState, after: ScreenState) -> bool:
        """
        Verify if a subtask succeeded.

        Args:
            subtask: The executed subtask
            before: Screen state before execution
            after: Screen state after execution

        Returns:
            True if verification passes
        """
        self._set_phase("verifying")

        # Simple verification: check if screen changed
        if before.active_window != after.active_window:
            return True

        if before.element_count != after.element_count:
            return True

        # Check postconditions using planner
        if subtask.postconditions:
            context = after.to_prompt_context()
            success, reason = self.planner.check_postconditions(subtask, context)
            if not success:
                self._msg(f"  Verification failed: {reason}")
            return success

        # Default: assume success
        return True

    # ==================== Main Loop ====================

    def process_goal(self, goal: str) -> str:
        """
        Process a goal using the full pipeline.

        Args:
            goal: The goal to achieve

        Returns:
            Summary of what happened
        """
        self.state.current_goal = goal
        self.state.iteration = 0
        self.is_running = True

        self._msg(f"Goal: {goal}")

        try:
            # Initial perception
            screen_state = self.perceive()
            self._msg(f"Screen: {screen_state.describe()}")

            # Create plan
            plan = self.plan(goal, screen_state)

            # Execute plan
            while self.is_running and self.state.iteration < self.state.max_iterations:
                self.state.iteration += 1

                # Get current subtask
                subtask = plan.get_current_task()
                if not subtask:
                    break

                self._msg(f"")
                self._msg(f"[{self.state.iteration}] Subtask {subtask.id}: {subtask.description}")

                # Perceive current state
                before_state = self.perceive()

                # Check if preconditions already satisfied
                pre_met, unmet = self.planner.check_preconditions(subtask, before_state.to_prompt_context())
                if not pre_met:
                    self._msg(f"  Preconditions not met: {unmet}")
                    # Could trigger replanning here

                # Execute
                subtask.status = TaskStatus.IN_PROGRESS
                success = self.execute_subtask(subtask, before_state)

                # Small delay for UI
                time.sleep(0.3)

                # Verify
                after_state = self.perceive()
                if config.agent.verify_actions:
                    verified = self.verify(subtask, before_state, after_state)
                    success = success and verified

                # Update status
                if success:
                    subtask.status = TaskStatus.COMPLETED
                    self._msg(f"  [OK] Subtask completed")
                else:
                    subtask.attempts += 1
                    if subtask.attempts >= subtask.max_attempts:
                        subtask.status = TaskStatus.FAILED
                        self._msg(f"  [FAIL] Subtask failed after {subtask.attempts} attempts")
                    else:
                        self._msg(f"  [RETRY] Attempt {subtask.attempts}/{subtask.max_attempts}")
                        continue  # Retry same subtask

                # Move to next subtask
                if not plan.advance():
                    break

            # Complete
            self.is_running = False

            if plan.is_complete():
                self.memory.complete_goal(success=True, message=f"Completed: {goal}")
                return f"Goal completed: {goal}"
            else:
                failed = [t for t in plan.subtasks if t.status == TaskStatus.FAILED]
                self.memory.complete_goal(success=False, message=f"Failed subtasks: {len(failed)}")
                return f"Goal partially completed. {len(plan.get_completed_tasks())}/{len(plan.subtasks)} subtasks done."

        except Exception as e:
            logger.error(f"Goal processing failed: {e}")
            self.memory.record_error(str(e))
            return f"Error: {e}"

    def stop(self):
        """Stop the agent."""
        self.is_running = False
        self._msg("Agent stopped")


# Factory function
def create_agent(**kwargs) -> IntelligentAgent:
    """Create a new intelligent agent."""
    return IntelligentAgent(**kwargs)
