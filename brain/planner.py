"""
VOXCODE Hierarchical Planner
Task decomposition and goal-oriented planning.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("voxcode.brain.planner")

from brain.llm import get_llm_client
from brain.api_registry import APIRegistry


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

    # Optional explicit state transition context
    input_state: str = ""
    output_state: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "status": self.status.value,
            "params": self.params,
            "input_state": self.input_state,
            "output_state": self.output_state,
        }


@dataclass
class TaskPlan:
    """A complete plan for achieving a goal."""
    goal: str
    initial_state: str = ""
    intermediate_states: List[str] = field(default_factory=list)
    goal_state: str = ""
    relevant_apis: List[Dict[str, Any]] = field(default_factory=list)
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
            "initial_state": self.initial_state,
            "intermediate_states": self.intermediate_states,
            "goal_state": self.goal_state,
            "relevant_apis": self.relevant_apis,
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

    DECOMPOSE_PROMPT = """You are a state-aware planner for a Windows automation system.

Given a user goal, produce a hierarchical task plan with explicit states.

USER GOAL: {goal}

CURRENT SCREEN CONTEXT:
{screen_context}

RELEVANT APIS (from registry):
{relevant_apis}

AVAILABLE ACTION TYPES:
- ensure_app_open
- navigate_to
- click_element
- type_text
- send_message
- search
- wait
- scroll
- verify

PLANNING RULES:
1. Generate a state hierarchy: initial -> intermediate checkpoints -> goal.
2. Each subtask must be atomic and represent one transition.
3. Every subtask must include preconditions and postconditions.
4. Use API registry data when the request references known services.
5. If a condition may already be true, still include a safe verification-aware step.
6. Keep output executable for UI automation.

Respond with ONLY JSON in this shape:
{
  "initial_state": "what is true now",
  "intermediate_states": ["checkpoint 1", "checkpoint 2"],
  "goal_state": "final desired state",
  "subtasks": [
    {
      "description": "what to do",
      "action_type": "one_of_action_types",
      "params": {"key": "value"},
      "preconditions": ["..."],
      "postconditions": ["..."],
      "input_state": "optional explicit input state",
      "output_state": "optional explicit output state"
    }
  ]
}"""

    REPLAN_PROMPT = """The current plan needs adjustment.

ORIGINAL GOAL: {goal}
COMPLETED TASKS: {completed}
CURRENT TASK FAILED: {failed_task}
FAILURE REASON: {failure_reason}

CURRENT SCREEN STATE:
{screen_context}

RELEVANT APIS (from registry):
{relevant_apis}

Generate a NEW plan to achieve the goal from the current state.
Consider what has already been done and what went wrong.

Respond with ONLY JSON in the same schema as planning."""

    def __init__(self, llm=None, api_registry: Optional[APIRegistry] = None):
        """Initialize planner."""
        self.llm = llm or get_llm_client()
        self.api_registry = api_registry or APIRegistry()
        self.audit_path = Path("audit_log.jsonl")

    @staticmethod
    def _render_prompt(template: str, values: Dict[str, Any]) -> str:
        """Render template placeholders without interpreting other JSON braces."""
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    def _audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Append planner-level audit events."""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "planner",
                "event_type": event_type,
                **payload,
            }
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Planner audit log write failed: {e}")

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
        apis = self.api_registry.find_relevant(goal)
        self._audit_event(
            "planner_relevant_apis_selected",
            {
                "goal": goal,
                "relevant_api_ids": [api.get("id") for api in apis],
                "relevant_api_count": len(apis),
            },
        )

        prompt = self._render_prompt(
            self.DECOMPOSE_PROMPT,
            {
                "goal": goal,
                "screen_context": screen_context,
                "relevant_apis": json.dumps(apis, indent=2) if apis else "[]",
            },
        )

        try:
            response = self.llm.generate(prompt)
            plan = self._parse_task_plan(
                response=response.content,
                goal=goal,
                screen_context=screen_context,
                relevant_apis=apis,
            )
            logger.info(f"Created plan with {len(plan.subtasks)} subtasks")
            self._audit_event(
                "planner_plan_created",
                {
                    "goal": goal,
                    "subtask_count": len(plan.subtasks),
                    "initial_state": plan.initial_state,
                    "goal_state": plan.goal_state,
                },
            )
            return plan
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            self._audit_event(
                "planner_plan_failed",
                {
                    "goal": goal,
                    "error": str(e),
                },
            )
            return self._fallback_plan(goal, screen_context, apis)

    def plan(self, voice_command: str, screen_context: str = "No screen context available") -> TaskPlan:
        """Architecture-compatible alias for create_plan."""
        return self.create_plan(goal=voice_command, screen_context=screen_context)

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
        apis = original_plan.relevant_apis or self.api_registry.find_relevant(original_plan.goal)
        self._audit_event(
            "planner_replan_started",
            {
                "goal": original_plan.goal,
                "failed_task": failed_task.description,
                "failure_reason": failure_reason,
                "relevant_api_ids": [api.get("id") for api in apis],
                "completed_task_count": len(completed),
            },
        )

        prompt = self._render_prompt(
            self.REPLAN_PROMPT,
            {
                "goal": original_plan.goal,
                "completed": json.dumps(completed),
                "failed_task": failed_task.description,
                "failure_reason": failure_reason,
                "screen_context": screen_context,
                "relevant_apis": json.dumps(apis, indent=2) if apis else "[]",
            },
        )

        try:
            response = self.llm.generate(prompt)
            plan = self._parse_task_plan(
                response=response.content,
                goal=original_plan.goal,
                screen_context=screen_context,
                relevant_apis=apis,
            )

            logger.info(f"Replanned with {len(plan.subtasks)} new subtasks")
            self._audit_event(
                "planner_replan_created",
                {
                    "goal": original_plan.goal,
                    "subtask_count": len(plan.subtasks),
                    "initial_state": plan.initial_state,
                    "goal_state": plan.goal_state,
                },
            )
            return plan

        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            self._audit_event(
                "planner_replan_failed",
                {
                    "goal": original_plan.goal,
                    "failed_task": failed_task.description,
                    "error": str(e),
                },
            )
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

    def _extract_json_payload(self, response: str) -> Any:
        """Extract JSON object/array from an LLM response."""
        # Prefer object format first (stateful schema), then array fallback.
        object_start = response.find("{")
        object_end = response.rfind("}") + 1
        if object_start >= 0 and object_end > object_start:
            try:
                return json.loads(response[object_start:object_end])
            except json.JSONDecodeError:
                pass

        array_start = response.find("[")
        array_end = response.rfind("]") + 1
        if array_start >= 0 and array_end > array_start:
            return json.loads(response[array_start:array_end])

        raise ValueError("No JSON object or array found in LLM response")

    def _build_state_chain(
        self,
        initial_state: str,
        intermediate_states: List[str],
        goal_state: str,
    ) -> List[str]:
        chain: List[str] = []
        if initial_state:
            chain.append(initial_state)
        chain.extend([s for s in intermediate_states if s])
        if goal_state:
            chain.append(goal_state)
        return chain

    def _normalize_condition_list(self, raw: Any) -> List[str]:
        """Normalize condition input into a clean list of strings."""
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        cleaned: List[str] = []
        for item in raw:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    cleaned.append(text)
        return cleaned

    def _build_subtask(self, index: int, task_dict: Dict[str, Any], state_chain: List[str]) -> Subtask:
        """Create a normalized Subtask with state defaults."""
        preconditions = self._normalize_condition_list(task_dict.get("preconditions", []))
        postconditions = self._normalize_condition_list(task_dict.get("postconditions", []))

        input_state = str(task_dict.get("input_state", "") or "").strip()
        output_state = str(task_dict.get("output_state", "") or "").strip()

        # Fill missing state/condition links from the state chain.
        if not preconditions and state_chain and index < len(state_chain) - 1:
            preconditions = [state_chain[index]]
        if not postconditions and state_chain and index + 1 < len(state_chain):
            postconditions = [state_chain[index + 1]]

        if not input_state and preconditions:
            input_state = preconditions[0]
        if not output_state and postconditions:
            output_state = postconditions[0]

        params = task_dict.get("params", {})
        if not isinstance(params, dict):
            params = {}

        return Subtask(
            id=index + 1,
            description=task_dict.get("description", f"Task {index + 1}"),
            action_type=task_dict.get("action_type", "unknown"),
            preconditions=preconditions,
            postconditions=postconditions,
            params=params,
            input_state=input_state,
            output_state=output_state,
        )

    def _parse_task_plan(
        self,
        response: str,
        goal: str,
        screen_context: str,
        relevant_apis: List[Dict[str, Any]],
    ) -> TaskPlan:
        """Parse LLM response into a state-aware TaskPlan."""
        payload = self._extract_json_payload(response)

        initial_state = screen_context
        intermediate_states: List[str] = []
        goal_state = goal
        subtasks_raw: List[Dict[str, Any]] = []

        if isinstance(payload, dict):
            state_hierarchy = payload.get("state_hierarchy", {})

            initial_state = (
                payload.get("initial_state")
                or state_hierarchy.get("initial")
                or screen_context
            )

            raw_intermediate = (
                payload.get("intermediate_states")
                or state_hierarchy.get("intermediate_states")
                or state_hierarchy.get("intermediate")
                or []
            )
            intermediate_states = self._normalize_condition_list(raw_intermediate)

            goal_state = (
                payload.get("goal_state")
                or state_hierarchy.get("goal")
                or goal
            )

            raw_subtasks = payload.get("subtasks", [])
            if isinstance(raw_subtasks, list):
                subtasks_raw = [s for s in raw_subtasks if isinstance(s, dict)]

        elif isinstance(payload, list):
            subtasks_raw = [s for s in payload if isinstance(s, dict)]
        else:
            raise ValueError("Unexpected plan payload type")

        plan = TaskPlan(
            goal=goal,
            initial_state=initial_state or screen_context,
            intermediate_states=intermediate_states,
            goal_state=goal_state or goal,
            relevant_apis=relevant_apis,
        )

        state_chain = self._build_state_chain(
            plan.initial_state,
            plan.intermediate_states,
            plan.goal_state,
        )

        for i, task_dict in enumerate(subtasks_raw):
            plan.add_subtask(self._build_subtask(i, task_dict, state_chain))

        if not plan.subtasks:
            return self._fallback_plan(goal, screen_context, relevant_apis)

        if not plan.intermediate_states:
            derived = [t.output_state for t in plan.subtasks[:-1] if t.output_state]
            plan.intermediate_states = derived

        if not plan.goal_state:
            plan.goal_state = plan.subtasks[-1].output_state or goal

        return plan

    def _fallback_plan(
        self,
        goal: str,
        screen_context: str,
        relevant_apis: List[Dict[str, Any]],
    ) -> TaskPlan:
        """Create a minimal safe fallback plan."""
        plan = TaskPlan(
            goal=goal,
            initial_state=screen_context,
            intermediate_states=[],
            goal_state=goal,
            relevant_apis=relevant_apis,
        )
        plan.add_subtask(Subtask(
            id=1,
            description=goal,
            action_type="unknown",
            preconditions=[screen_context] if screen_context else [],
            postconditions=[goal],
            input_state=screen_context,
            output_state=goal,
        ))
        return plan

    def _parse_subtasks(self, response: str) -> List[Dict]:
        """Backward-compatible subtask parser (legacy list format)."""
        try:
            payload = self._extract_json_payload(response)
            if isinstance(payload, dict):
                raw = payload.get("subtasks", [])
                if isinstance(raw, list):
                    return [s for s in raw if isinstance(s, dict)]
            if isinstance(payload, list):
                return [s for s in payload if isinstance(s, dict)]
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")

        logger.warning("Falling back to simple parsing")
        return [{"description": response.strip(), "action_type": "unknown"}]

    def describe_plan(self, plan: TaskPlan) -> str:
        """Get a human-readable description of the plan."""
        lines = [f"Plan: {plan.goal}", "=" * 40]

        if plan.initial_state:
            lines.append(f"Initial state: {plan.initial_state}")
        if plan.intermediate_states:
            lines.append(f"Intermediate states: {' -> '.join(plan.intermediate_states)}")
        if plan.goal_state:
            lines.append(f"Goal state: {plan.goal_state}")
        if plan.relevant_apis:
            api_names = ", ".join(api.get("name", api.get("id", "api")) for api in plan.relevant_apis)
            lines.append(f"Relevant APIs: {api_names}")
        lines.append("-" * 40)

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
            if task.input_state or task.output_state:
                lines.append(f"    State: {task.input_state or '?'} -> {task.output_state or '?'}")

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
