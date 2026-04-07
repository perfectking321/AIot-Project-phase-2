"""
VOXCODE Skill Base Classes
Foundation for the skill system.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger("voxcode.skills")

if TYPE_CHECKING:
    from perception.screen_state import ScreenState


class SkillStatus(Enum):
    """Status of skill execution."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some steps succeeded
    PRECONDITION_NOT_MET = "precondition_not_met"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Result of skill execution."""
    status: SkillStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    steps_completed: int = 0
    steps_total: int = 0

    @property
    def success(self) -> bool:
        return self.status == SkillStatus.SUCCESS


@dataclass
class SkillStep:
    """A single step within a skill."""
    name: str
    action: Callable
    description: str = ""
    verify: Optional[Callable] = None  # Optional verification function
    retry_on_fail: bool = True
    max_retries: int = 2


class Skill(ABC):
    """
    Base class for all skills.

    A skill is a pre-defined sequence of actions for a specific task,
    with built-in verification and error handling.
    """

    # Skill metadata
    name: str = "base_skill"
    description: str = "Base skill class"

    # What parameters this skill accepts
    params: List[str] = []

    # Preconditions that must be true
    preconditions: List[str] = []

    # What should be true after execution
    postconditions: List[str] = []

    def __init__(self, tools=None, perception=None):
        """
        Initialize skill.

        Args:
            tools: WindowsTools instance
            perception: Perception engine for screen understanding
        """
        self._tools = tools
        self._perception = perception
        self._steps: List[SkillStep] = []

    def _get_tools(self):
        """Lazy load tools."""
        if self._tools is None:
            from agent.tools import WindowsTools
            self._tools = WindowsTools()
        return self._tools

    def _get_perception(self):
        """Lazy load perception."""
        if self._perception is None:
            try:
                from perception.vlm import get_vlm
                self._perception = get_vlm()
            except ImportError:
                pass
        return self._perception

    @abstractmethod
    def execute(self, **params) -> SkillResult:
        """
        Execute the skill with given parameters.

        Args:
            **params: Skill-specific parameters

        Returns:
            SkillResult indicating success/failure
        """
        pass

    def check_preconditions(self, screen_state: 'ScreenState' = None) -> tuple[bool, str]:
        """
        Check if preconditions are met.

        Args:
            screen_state: Current screen state

        Returns:
            (met, reason)
        """
        # Default: assume preconditions are met
        # Subclasses should override for specific checks
        return True, "Preconditions assumed met"

    def verify_postconditions(self, screen_state: 'ScreenState' = None) -> tuple[bool, str]:
        """
        Verify postconditions after execution.

        Args:
            screen_state: Screen state after execution

        Returns:
            (met, reason)
        """
        # Default: assume success if we got here
        return True, "Postconditions assumed met"

    def _execute_step(self, step: SkillStep, **kwargs) -> bool:
        """Execute a single step with retries."""
        for attempt in range(step.max_retries + 1):
            try:
                logger.info(f"Executing step: {step.name} (attempt {attempt + 1})")
                result = step.action(**kwargs)

                # Check if result indicates success
                success = True
                if hasattr(result, 'success'):
                    success = result.success
                elif isinstance(result, bool):
                    success = result

                if success:
                    # Verify if verification function provided
                    if step.verify:
                        if step.verify():
                            return True
                        logger.warning(f"Step verification failed: {step.name}")
                    else:
                        return True

                if not step.retry_on_fail:
                    break

            except Exception as e:
                logger.error(f"Step {step.name} error: {e}")
                if not step.retry_on_fail:
                    break

            time.sleep(0.5)

        return False

    def _run_steps(self, steps: List[SkillStep], **kwargs) -> SkillResult:
        """Run a sequence of steps."""
        completed = 0
        total = len(steps)

        for step in steps:
            if self._execute_step(step, **kwargs):
                completed += 1
            else:
                return SkillResult(
                    status=SkillStatus.FAILED,
                    message=f"Failed at step: {step.name}",
                    steps_completed=completed,
                    steps_total=total
                )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            message="All steps completed",
            steps_completed=completed,
            steps_total=total
        )


class SkillRegistry:
    """Registry of available skills."""

    def __init__(self):
        self._skills: Dict[str, type] = {}
        self._instances: Dict[str, Skill] = {}

    def register(self, skill_class: type):
        """Register a skill class."""
        if hasattr(skill_class, 'name'):
            self._skills[skill_class.name] = skill_class
            logger.debug(f"Registered skill: {skill_class.name}")

    def get(self, name: str, tools=None, perception=None) -> Optional[Skill]:
        """Get a skill instance by name."""
        if name in self._instances:
            return self._instances[name]

        if name in self._skills:
            instance = self._skills[name](tools=tools, perception=perception)
            self._instances[name] = instance
            return instance

        return None

    def list_skills(self) -> List[Dict[str, str]]:
        """List all registered skills."""
        return [
            {
                "name": skill_class.name,
                "description": skill_class.description,
                "params": skill_class.params
            }
            for skill_class in self._skills.values()
        ]

    def find_skill_for_action(self, action_type: str) -> Optional[str]:
        """Find a skill that can handle an action type."""
        # Map action types to skills
        action_skill_map = {
            "open_app": "open_app",
            "ensure_app_open": "open_app",
            "close_app": "close_app",
            "focus_app": "focus_app",
            "send_message": "send_message",
            "open_chat": "open_chat",
            "navigate_to": "navigate_url",
            "search": "search_web",
            "type_text": "type_text",
            "click_element": "click_element",
            "scroll": "scroll",
        }
        return action_skill_map.get(action_type)


# Global registry
_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """Get or create the global skill registry."""
    global _registry

    if _registry is None:
        _registry = SkillRegistry()
        _register_default_skills(_registry)

    return _registry


def _register_default_skills(registry: SkillRegistry):
    """Register all default skills."""
    from agent.skills.app_skills import OpenAppSkill, CloseAppSkill, FocusAppSkill
    from agent.skills.messaging_skills import SendMessageSkill, OpenChatSkill
    from agent.skills.browser_skills import NavigateToUrlSkill, SearchWebSkill
    from agent.skills.input_skills import TypeTextSkill, ClickElementSkill, ScrollSkill

    for skill_class in [
        OpenAppSkill, CloseAppSkill, FocusAppSkill,
        SendMessageSkill, OpenChatSkill,
        NavigateToUrlSkill, SearchWebSkill,
        TypeTextSkill, ClickElementSkill, ScrollSkill
    ]:
        registry.register(skill_class)
