"""
VOXCODE Hierarchical Planner
Task decomposition and goal-oriented planning.
"""

import json
import logging
import time
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("voxcode.brain.planner")

from brain.llm import get_llm_client, get_model_for_role, OllamaClient
from brain.api_registry import APIRegistry
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

    # Optional explicit state transition context
    input_state: str = ""
    output_state: str = ""
    verify_condition: str = ""

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
            "verify_condition": self.verify_condition,
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

    DECOMPOSE_PROMPT = """You are a subgoal planner for a Windows desktop automation agent.

The agent has a vision module (OmniParser) that reads the screen and finds UI elements at runtime.
Your job is ONLY to produce a list of checkpoints from initial state to goal state.
Do NOT specify click coordinates, element selectors, or low-level UI steps.
Each subgoal has ONE job. The vision agent handles how to achieve it.

For app launching, use action_type: "launch_app" with the app name.
For navigation, use action_type: "navigate_to" with a URL.
For input tasks, use action_type: "search" or "type_text" with the text.
For interaction, use action_type: "click_target" with a description of what to click.

Each subgoal must have a verify_condition: a single concrete thing visible on screen that confirms success.

USER GOAL: {goal}
CURRENT SCREEN: {screen_context}
AVAILABLE APPS/APIS: {relevant_apis}

Respond ONLY with JSON:
{
    "initial_state": "one-line description of current screen",
    "goal_state": "one-line description of final desired screen",
    "subgoals": [
        {
            "id": 1,
            "intent": "what this step achieves",
            "action_type": "launch_app | navigate_to | search | type_text | click_target | verify",
            "params": {"key": "value"},
            "verify_condition": "what must be visible/true on screen after this step"
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
        self.llm = llm or get_model_for_role("planner")
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
                    "intermediate_states": plan.intermediate_states,
                    "subtasks": [subtask.to_dict() for subtask in plan.subtasks],
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
            raise RuntimeError(f"Planner failed: {e}") from e

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

    @staticmethod
    def _is_symbolic_state_ref(value: str) -> bool:
        """Detect placeholder-like state names that are not directly verifiable on screen."""
        token = (value or "").strip().lower()
        if not token:
            return False
        if token in {"initial_state", "goal_state", "current_state", "next_state", "input_state", "output_state"}:
            return True
        return bool(re.fullmatch(r"[a-z][a-z0-9_]*", token)) and "_" in token

    def _resolve_state_ref(self, value: str, state_chain: List[str], index: int, fallback: str = "") -> str:
        """Resolve symbolic state references into concrete text."""
        text = (value or "").strip()
        if not text:
            return fallback

        token = text.lower()
        if token in {"initial_state", "current_state", "input_state"}:
            if state_chain and index < len(state_chain):
                return state_chain[index]
            return fallback or text

        if token in {"goal_state", "output_state", "next_state"}:
            if state_chain and (index + 1) < len(state_chain):
                return state_chain[index + 1]
            if state_chain:
                return state_chain[-1]
            return fallback or text

        if self._is_symbolic_state_ref(text):
            return text.replace("_", " ")

        return text

    @staticmethod
    def _normalize_action_type(action_type: str, description: str) -> str:
        """Normalize planner action labels into executor-supported action types."""
        raw = (action_type or "").strip().lower()
        desc = (description or "").strip().lower()

        aliases = {
            "open_app": "launch_app",
            "open_application": "launch_app",
            "ensure_app_open": "launch_app",
            "close_window": "click_target",
            "click": "click_target",
            "click_element": "click_target",
            "type": "type_text",
            "press": "click_target",
            "open_url": "navigate_to",
        }
        raw = aliases.get(raw, raw)

        if raw in {"launch_app", "navigate_to", "search", "type_text", "click_target", "verify"}:
            return raw

        if not raw or raw == "unknown":
            if re.search(r"\b(open|launch|start)\b", desc):
                return "launch_app"
            if re.search(r"\b(navigate|go to|open url|visit|browse)\b", desc):
                return "navigate_to"
            if re.search(r"\b(search|find|look up)\b", desc):
                return "search"
            if re.search(r"\b(type|write|enter text)\b", desc):
                return "type_text"
            if re.search(r"\b(click|tap|select|open)\b", desc):
                return "click_target"

        return raw or "unknown"

    def _build_subtask(self, index: int, task_dict: Dict[str, Any], state_chain: List[str]) -> Subtask:
        """Create a normalized Subtask with state defaults."""
        description = task_dict.get("intent") or task_dict.get("description") or f"Task {index + 1}"
        verify_condition = str(task_dict.get("verify_condition", "") or "").strip()
        preconditions = self._normalize_condition_list(task_dict.get("preconditions", []))
        postconditions = self._normalize_condition_list(task_dict.get("postconditions", []))

        input_state = str(task_dict.get("input_state", "") or "").strip()
        output_state = str(task_dict.get("output_state", "") or "").strip()

        default_input = state_chain[index] if state_chain and index < len(state_chain) else ""
        default_output = state_chain[index + 1] if state_chain and index + 1 < len(state_chain) else ""

        # Fill missing state/condition links from the state chain.
        if not preconditions and state_chain and index < len(state_chain) - 1:
            preconditions = [state_chain[index]]
        if verify_condition and not postconditions:
            postconditions = [verify_condition]
        elif not postconditions and state_chain and index + 1 < len(state_chain):
            postconditions = [state_chain[index + 1]]

        preconditions = [
            self._resolve_state_ref(condition, state_chain, index, fallback=default_input)
            for condition in preconditions
        ]
        postconditions = [
            self._resolve_state_ref(condition, state_chain, index, fallback=default_output)
            for condition in postconditions
        ]
        preconditions = [condition for condition in preconditions if condition]
        postconditions = [condition for condition in postconditions if condition]

        if not input_state and preconditions:
            input_state = preconditions[0]
        if not output_state and postconditions:
            output_state = postconditions[0]
        if not output_state and verify_condition:
            output_state = verify_condition

        input_state = self._resolve_state_ref(input_state, state_chain, index, fallback=default_input)
        output_state = self._resolve_state_ref(output_state, state_chain, index, fallback=default_output)

        params = task_dict.get("params", {})
        if not isinstance(params, dict):
            params = {}

        normalized_action = self._normalize_action_type(task_dict.get("action_type", "unknown"), description)

        # Fill missing critical params for deterministic action handlers.
        if normalized_action == "launch_app" and not any(
            isinstance(params.get(key), str) and params.get(key).strip()
            for key in ("app_name", "path_or_name", "target", "application", "app")
        ):
            match = re.search(r"(?:open|launch|start|ensure)\s+([a-zA-Z0-9 ._:-]+)", description, flags=re.IGNORECASE)
            if match:
                candidate = re.split(
                    r"\s+(?:and|then|to|for)\b",
                    match.group(1).strip(" ."),
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                candidate = re.split(r"\s+(?:is|are|with)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                params["app"] = self._canonicalize_app_name(candidate)

        if normalized_action == "navigate_to" and not isinstance(params.get("url"), str):
            url_match = re.search(r"(https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|io|dev|ai))", description)
            if url_match:
                params["url"] = url_match.group(1).rstrip(".,")

        if normalized_action == "search" and not isinstance(params.get("query"), str):
            params["query"] = self._extract_type_text_from_goal(description) or description

        if normalized_action == "type_text" and not isinstance(params.get("text"), str):
            text_match = re.search(r"['\"]([^'\"]+)['\"]", description)
            if text_match:
                params["text"] = text_match.group(1).strip()

        if normalized_action == "click_target" and not isinstance(params.get("target"), str):
            params["target"] = description

        return Subtask(
            id=index + 1,
            description=description,
            action_type=normalized_action,
            preconditions=preconditions,
            postconditions=postconditions,
            params=params,
            input_state=input_state,
            output_state=output_state,
            verify_condition=verify_condition or (postconditions[0] if postconditions else output_state),
        )

    @staticmethod
    def _extract_url_from_text(text: str) -> str:
        """Extract URL-like target from text and normalize to https."""
        match = re.search(
            r"(https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|io|dev|ai|co)(?:/\S*)?)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""

        url = match.group(1).rstrip(".,)")
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"https://{url}"
        return url

    @staticmethod
    def _extract_type_text_from_goal(goal: str) -> str:
        """Extract text payload from type/write style goals."""
        quoted = re.search(r"(?:type|write|enter(?: text)?)\s+['\"]([^'\"]+)['\"]", goal, flags=re.IGNORECASE)
        if quoted:
            return quoted.group(1).strip()

        typed = re.search(r"\b(?:type|write|enter(?: text)?)\b\s+(.+)", goal, flags=re.IGNORECASE)
        if not typed:
            return ""

        text = typed.group(1).strip()
        text = re.split(r"\s+(?:in|into|on|to|inside)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = text.strip(" .\"'")
        return text

    @staticmethod
    def _canonicalize_app_name(name: str) -> str:
        """Normalize user-facing app names to executable-friendly targets."""
        raw = (name or "").strip()
        if not raw:
            return ""

        normalized = re.sub(r"\s+", " ", raw).strip().lower()
        normalized = re.sub(r"\b(application|app)\b", "", normalized).strip()

        aliases = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "file explorer": "explorer.exe",
            "explorer": "explorer.exe",
            "command prompt": "cmd.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "windows powershell": "powershell.exe",
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "edge": "msedge.exe",
            "microsoft edge": "msedge.exe",
            "vscode": "code.exe",
            "visual studio code": "code.exe",
        }

        if normalized in aliases:
            return aliases[normalized]

        # Keep explicit executable/path targets untouched.
        if raw.lower().endswith(".exe") or "\\" in raw or "/" in raw or ":" in raw:
            return raw

        return normalized or raw

    def _infer_app_target(self, goal: str, relevant_apis: List[Dict[str, Any]]) -> str:
        """Infer app target from command text and API registry hints."""
        goal_text = (goal or "").strip()

        match = re.search(
            r"\b(?:open|launch|start|close|exit|quit)\s+([a-zA-Z0-9 ._:-]+)",
            goal_text,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip(" .")
            candidate = re.split(
                r"\s+(?:and|then|to|for|with|in|on)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .")
            if candidate:
                return self._canonicalize_app_name(candidate)

        if re.search(r"\b([a-z]:\\|[a-z]\s*drive|folder|directory|file explorer|explorer)\b", goal_text, flags=re.IGNORECASE):
            return "explorer.exe"

        for api in relevant_apis or []:
            api_id = str(api.get("id", "")).lower()
            api_kind = str(api.get("kind", "")).lower()
            if api_kind == "app" or api_id.startswith("app_"):
                exe_name = api.get("exe_name")
                if isinstance(exe_name, str) and exe_name.strip():
                    return exe_name.strip()
                api_name = api.get("name")
                if isinstance(api_name, str) and api_name.strip():
                    return self._canonicalize_app_name(api_name)

        if re.search(r"\b(chrome|browser|youtube|google|website|web)\b", goal_text, flags=re.IGNORECASE):
            return "chrome.exe"

        return ""

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

            raw_subtasks = payload.get("subgoals")
            if not isinstance(raw_subtasks, list):
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
            raise ValueError("Planner returned no valid subgoals")

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
        """Create a deterministic fallback plan when LLM planning fails."""
        plan = TaskPlan(
            goal=goal,
            initial_state=screen_context,
            intermediate_states=[],
            goal_state=goal,
            relevant_apis=relevant_apis,
        )

        goal_text = (goal or "").strip()
        goal_lower = goal_text.lower()
        current_state = plan.initial_state or "Current desktop state"

        def add_subtask(
            description: str,
            action_type: str,
            params: Dict[str, Any],
            output_state: str,
            postconditions: List[str],
            verify_condition: str,
        ) -> None:
            nonlocal current_state
            preconditions = [current_state] if current_state else []
            plan.add_subtask(
                Subtask(
                    id=len(plan.subtasks) + 1,
                    description=description,
                    action_type=action_type,
                    preconditions=preconditions,
                    postconditions=postconditions,
                    params=params,
                    input_state=current_state,
                    output_state=output_state,
                    verify_condition=verify_condition,
                )
            )
            if output_state:
                current_state = output_state

        close_intent = bool(re.search(r"\b(close|exit|quit|minimi[sz]e)\b", goal_lower))
        open_intent = bool(re.search(r"\b(open|launch|start)\b", goal_lower))
        navigate_intent = bool(
            re.search(r"\b(navigate|go to|visit|browse|open url|website|web)\b", goal_lower)
        )
        system_intent = bool(re.search(r"\b(mute|unmute|volume|brightness)\b", goal_lower))

        app_target = self._infer_app_target(goal_text, relevant_apis)
        url_target = self._extract_url_from_text(goal_text)
        type_payload = self._extract_type_text_from_goal(goal_text)

        if system_intent:
            add_subtask(
                description=goal_text,
                action_type="click_target",
                params={"target": goal_text},
                output_state="System control command completed",
                postconditions=["System control command completed"],
                verify_condition="System control command completed",
            )
        else:
            if close_intent:
                target = app_target or "active window"
                add_subtask(
                    description=f"Close {target}",
                    action_type="click_target",
                    params={"target": target},
                    output_state=f"{target} is closed",
                    postconditions=[f"{target} is closed"],
                    verify_condition=f"{target} is closed",
                )
            else:
                if open_intent or (app_target and (navigate_intent or bool(type_payload))):
                    target = app_target or "chrome.exe"
                    add_subtask(
                        description=f"Open {target}",
                        action_type="launch_app",
                        params={"app": target},
                        output_state=f"{target} is open and active",
                        postconditions=[f"{target} is open and active"],
                        verify_condition=f"{target} window is visible",
                    )

                if navigate_intent:
                    if not url_target and "youtube" in goal_lower:
                        url_target = "https://youtube.com"
                    elif not url_target and "google" in goal_lower:
                        url_target = "https://google.com"

                    if url_target:
                        add_subtask(
                            description=f"Navigate to {url_target}",
                            action_type="navigate_to",
                            params={"url": url_target},
                            output_state=f"Browser navigated to {url_target}",
                            postconditions=[f"{url_target} is visible in browser"],
                            verify_condition=f"{url_target} page content is visible",
                        )

                if type_payload:
                    add_subtask(
                        description=f"Type '{type_payload}'",
                        action_type="type_text",
                        params={"text": type_payload},
                        output_state=f"Text '{type_payload}' typed",
                        postconditions=[f"Text '{type_payload}' typed"],
                        verify_condition=f"Text '{type_payload}' is visible on screen",
                    )

        if not plan.subtasks:
            add_subtask(
                description=f"Wait before retrying unsafe fallback: {goal_text}",
                action_type="verify",
                params={"target": goal_text},
                output_state="UI state stabilized",
                postconditions=["UI state stabilized"],
                verify_condition="UI state stabilized",
            )

        plan.intermediate_states = [subtask.output_state for subtask in plan.subtasks[:-1] if subtask.output_state]
        plan.goal_state = plan.subtasks[-1].output_state if plan.subtasks else goal

        self._audit_event(
            "planner_fallback_plan_created",
            {
                "goal": goal,
                "subtask_count": len(plan.subtasks),
                "subtasks": [subtask.to_dict() for subtask in plan.subtasks],
            },
        )
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


def _model_name_matches(candidate: str, available_name: str) -> bool:
    """Best-effort match between requested and installed Ollama model names."""
    cand = (candidate or "").strip().lower()
    avail = (available_name or "").strip().lower()
    if not cand or not avail:
        return False

    if cand == avail:
        return True

    cand_parts = cand.split(":", 1)
    avail_parts = avail.split(":", 1)

    cand_base = cand_parts[0]
    cand_tag = cand_parts[1] if len(cand_parts) > 1 else ""
    avail_base = avail_parts[0]
    avail_tag = avail_parts[1] if len(avail_parts) > 1 else ""

    # Different model families should never match.
    if cand_base != avail_base:
        return False

    # Untagged candidate can match any installed tag of the same base model.
    if not cand_tag:
        return True

    # Tagged candidate must match exact tag.
    return cand_tag == avail_tag


def _resolve_planner_model_name() -> str:
    """Resolve a usable local planner model, falling back to installed alternatives."""
    planner_model = getattr(config.llm, "planner_model", config.llm.ollama_model)
    candidates = []
    for model_name in [
        planner_model,
        getattr(config.llm, "executor_model", ""),
        config.llm.ollama_model,
    ]:
        model_name = (model_name or "").strip()
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    if not candidates:
        return planner_model

    try:
        probe = OllamaClient(
            host=config.llm.ollama_host,
            model=candidates[0],
            temperature=min(config.llm.temperature, 0.2),
            max_tokens=config.llm.max_tokens,
            timeout=min(config.llm.timeout, 5),
        )
        models = probe.list_models()
        available_names = [m.get("name", "") for m in models if isinstance(m, dict)]

        for candidate in candidates:
            if any(_model_name_matches(candidate, name) for name in available_names):
                return candidate

        qwen_models = [name for name in available_names if "qwen" in name.lower()]
        if qwen_models:
            return qwen_models[0]
    except Exception as exc:
        logger.warning(f"Planner model resolution failed, using configured planner model: {exc}")

    return planner_model


def get_planner() -> HierarchicalPlanner:
    """Get or create global planner instance."""
    global _planner_instance

    if _planner_instance is None:
        try:
            planner_model = _resolve_planner_model_name()
            planner_llm = OllamaClient(
                host=config.llm.ollama_host,
                model=planner_model,
                temperature=min(config.llm.temperature, 0.2),
                max_tokens=config.llm.max_tokens,
                timeout=config.llm.timeout,
            )
            _planner_instance = HierarchicalPlanner(llm=planner_llm)
        except Exception as e:
            logger.warning(f"Planner model initialization fallback triggered: {e}")
            _planner_instance = HierarchicalPlanner()

    return _planner_instance
