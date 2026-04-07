"""
VOXCODE Hierarchical Planner
Task decomposition and goal-oriented planning.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("voxcode.brain.planner")

from brain.llm import get_llm_client
from config import config


class TaskStatus(Enum):
    """Status of a task in the plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Already satisfied


@dataclass
class Subtask:
    """A single subtask in the plan."""
    id: int
    description: str
    action_type: str  # e.g., "open_app", "click", "type", "wait"

    # Preconditions (what must be true before this task)
    preconditions: List[str] = field(default_factory=list)

    # Postconditions (what should be true after this task)
    postconditions: List[str] = field(default_factory=list)

    # Execution details
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    attempts: int = 0
    max_attempts: int = 3

    # Parameters for the action
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "status": self.status.value,
            "params": self.params
        }


@dataclass
class TaskPlan:
    """A complete plan for achieving a goal."""
    goal: str
    subtasks: List[Subtask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    current_task_index: int = 0

    def add_subtask(self, subtask: Subtask):
        """Add a subtask to the plan."""
        self.subtasks.append(subtask)

    def get_current_task(self) -> Optional[Subtask]:
        """Get the current task to execute."""
        if self.current_task_index < len(self.subtasks):
            return self.subtasks[self.current_task_index]
        return None

    def advance(self) -> bool:
        """Move to next task. Returns False if no more tasks."""
        self.current_task_index += 1
        return self.current_task_index < len(self.subtasks)

    def get_pending_tasks(self) -> List[Subtask]:
        """Get all pending tasks."""
        return [t for t in self.subtasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> List[Subtask]:
        """Get all completed tasks."""
        return [t for t in self.subtasks if t.status == TaskStatus.COMPLETED]

    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        return all(
            t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            for t in self.subtasks
        )

    def summary(self) -> str:
        """Get plan summary."""
        completed = len(self.get_completed_tasks())
        total = len(self.subtasks)
        return f"Plan: {self.goal} ({completed}/{total} tasks done)"

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "status": self.status.value,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "current_index": self.current_task_index
        }


class HierarchicalPlanner:
    """
    Hierarchical task planner using LLM.

    Decomposes high-level goals into executable subtasks with
    preconditions and postconditions.
    """

    DECOMPOSE_PROMPT = """You are a task planner for a Windows automation system.

Given a user's goal, decompose it into a sequence of executable subtasks.

USER GOAL: {goal}

CURRENT SCREEN CONTEXT:
{screen_context}

AVAILABLE ACTION TYPES:
- ensure_app_open: Make sure an app is open (params: app_name)
- navigate_to: Navigate within an app (params: destination)
- click_element: Click a UI element (params: element_description)
- type_text: Type text (params: text)
- send_message: Send a message in a chat app (params: message, recipient)
- search: Search for something (params: query)
- wait: Wait for something to load (params: seconds)
- scroll: Scroll the screen (params: direction, amount)
- verify: Verify a condition is met (params: condition)

RULES:
1. Each subtask should be a single, atomic action
2. Include preconditions (what must be true before the task)
3. Include postconditions (what should be true after the task)
4. Consider what's ALREADY visible on screen - don't repeat actions
5. Order tasks logically with dependencies

Respond with ONLY a JSON array of subtasks:
```json
[
  {{
    "description": "Brief description of what to do",
    "action_type": "action_type_from_list",
    "params": {{"key": "value"}},
    "preconditions": ["condition that must be true"],
    "postconditions": ["condition that will be true after"]
  }},
  ...
]
```

IMPORTANT: If something is already done (app already open, already on right page), skip that subtask or include it with a note that it may already be satisfied."""

    REPLAN_PROMPT = """The current plan needs adjustment.

ORIGINAL GOAL: {goal}
COMPLETED TASKS: {completed}
CURRENT TASK FAILED: {failed_task}
FAILURE REASON: {failure_reason}

CURRENT SCREEN STATE:
{screen_context}

Generate a NEW plan to achieve the goal from the current state.
Consider what has already been done and what went wrong.

Respond with ONLY a JSON array of new subtasks (same format as before)."""

    def __init__(self, llm = None):
        """Initialize planner."""
        self.llm = llm or get_llm_client()

    def create_plan(
        self,
        goal: str,
        screen_context: str = "No screen context available"
    ) -> TaskPlan:
        """
        Create a plan to achieve a goal.

        Args:
            goal: The high-level goal to achieve
            screen_context: Current screen state description

        Returns:
            TaskPlan with decomposed subtasks
        """
        logger.info(f"Creating plan for: {goal}")

        prompt = self.DECOMPOSE_PROMPT.format(
            goal=goal,
            screen_context=screen_context
        )

        try:
            response = self.llm.generate(prompt)
            subtasks = self._parse_subtasks(response.content)

            plan = TaskPlan(goal=goal)
            for i, task_dict in enumerate(subtasks):
                subtask = Subtask(
                    id=i + 1,
                    description=task_dict.get("description", f"Task {i+1}"),
                    action_type=task_dict.get("action_type", "unknown"),
                    preconditions=task_dict.get("preconditions", []),
                    postconditions=task_dict.get("postconditions", []),
                    params=task_dict.get("params", {})
                )
                plan.add_subtask(subtask)

            logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
            return plan

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Return a simple single-task plan as fallback
            plan = TaskPlan(goal=goal)
            plan.add_subtask(Subtask(
                id=1,
                description=goal,
                action_type="unknown",
                preconditions=[],
                postconditions=[]
            ))
            return plan

    def replan(
        self,
        original_plan: TaskPlan,
        failed_task: Subtask,
        failure_reason: str,
        screen_context: str
    ) -> TaskPlan:
        """
        Create a new plan after a failure.

        Args:
            original_plan: The plan that was being executed
            failed_task: The task that failed
            failure_reason: Why it failed
            screen_context: Current screen state

        Returns:
            New TaskPlan
        """
        logger.info(f"Replanning after failure: {failure_reason}")

        completed = [t.description for t in original_plan.get_completed_tasks()]

        prompt = self.REPLAN_PROMPT.format(
            goal=original_plan.goal,
            completed=json.dumps(completed),
            failed_task=failed_task.description,
            failure_reason=failure_reason,
            screen_context=screen_context
        )

        try:
            response = self.llm.generate(prompt)
            subtasks = self._parse_subtasks(response.content)

            plan = TaskPlan(goal=original_plan.goal)
            for i, task_dict in enumerate(subtasks):
                subtask = Subtask(
                    id=i + 1,
                    description=task_dict.get("description", f"Task {i+1}"),
                    action_type=task_dict.get("action_type", "unknown"),
                    preconditions=task_dict.get("preconditions", []),
                    postconditions=task_dict.get("postconditions", []),
                    params=task_dict.get("params", {})
                )
                plan.add_subtask(subtask)

            logger.info(f"Replanned with {len(plan.subtasks)} new subtasks")
            return plan

        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            return original_plan

    def check_preconditions(
        self,
        task: Subtask,
        screen_context: str
    ) -> tuple[bool, List[str]]:
        """
        Check if a task's preconditions are met.

        Args:
            task: The task to check
            screen_context: Current screen state

        Returns:
            (all_met, unmet_conditions)
        """
        if not task.preconditions:
            return True, []

        # Use LLM to check conditions
        prompt = f"""Check if these conditions are currently true based on the screen state.

CONDITIONS TO CHECK:
{json.dumps(task.preconditions)}

CURRENT SCREEN STATE:
{screen_context}

For each condition, respond with:
- CONDITION: [the condition]
- MET: yes/no
- REASON: [brief reason]

Then summarize: ALL_MET: yes/no"""

        try:
            response = self.llm.generate(prompt)
            content = response.content.lower()

            all_met = "all_met: yes" in content

            unmet = []
            if not all_met:
                for cond in task.preconditions:
                    if f"{cond.lower()}" in content and "met: no" in content:
                        unmet.append(cond)

            return all_met, unmet

        except Exception as e:
            logger.error(f"Precondition check failed: {e}")
            # Assume conditions are met to avoid blocking
            return True, []

    def check_postconditions(
        self,
        task: Subtask,
        screen_context: str
    ) -> tuple[bool, str]:
        """
        Verify if a task's postconditions are satisfied.

        Args:
            task: The completed task
            screen_context: Screen state after execution

        Returns:
            (success, reason)
        """
        if not task.postconditions:
            return True, "No postconditions to verify"

        prompt = f"""Verify if the action was successful.

ACTION TAKEN: {task.description}

EXPECTED RESULTS:
{json.dumps(task.postconditions)}

CURRENT SCREEN STATE:
{screen_context}

Are ALL expected results visible/achieved?
Respond with:
SUCCESS: yes/no
REASON: [brief explanation]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content.lower()

            success = "success: yes" in content

            # Extract reason
            reason = "Verification completed"
            if "reason:" in content:
                reason_start = content.find("reason:") + 7
                reason = content[reason_start:].strip().split('\n')[0]

            return success, reason

        except Exception as e:
            logger.error(f"Postcondition check failed: {e}")
            return True, "Verification skipped due to error"

    def _parse_subtasks(self, response: str) -> List[Dict]:
        """Parse LLM response into subtask dictionaries."""
        # Find JSON array in response
        try:
            # Look for JSON block
            start = response.find('[')
            end = response.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")

        # Fallback: try to extract structured info
        logger.warning("Falling back to simple parsing")
        return [{"description": response.strip(), "action_type": "unknown"}]

    def describe_plan(self, plan: TaskPlan) -> str:
        """Get a human-readable description of the plan."""
        lines = [f"Plan: {plan.goal}", "=" * 40]

        for task in plan.subtasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✓",
                TaskStatus.FAILED: "✗",
                TaskStatus.SKIPPED: "⊘"
            }.get(task.status, "?")

            lines.append(f"{status_icon} [{task.id}] {task.description}")

            if task.preconditions:
                lines.append(f"    Pre: {', '.join(task.preconditions)}")
            if task.postconditions:
                lines.append(f"    Post: {', '.join(task.postconditions)}")

        lines.append("=" * 40)
        return "\n".join(lines)


# Singleton instance
_planner_instance: Optional[HierarchicalPlanner] = None


def get_planner() -> HierarchicalPlanner:
    """Get or create global planner instance."""
    global _planner_instance

    if _planner_instance is None:
        _planner_instance = HierarchicalPlanner()

    return _planner_instance
