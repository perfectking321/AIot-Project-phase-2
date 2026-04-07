"""
VOXCODE Task Planner
Breaks down user voice commands into executable steps with vision support.
"""

import json
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from brain.llm import get_llm_client, BaseLLMClient
from brain.prompts import PromptBuilder

logger = logging.getLogger("voxcode.planner")


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """A single step in a plan."""
    number: int
    description: str
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """A complete task plan."""
    task: str
    steps: List[Step] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def get_current_step(self) -> Optional[Step]:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self) -> None:
        self.current_step += 1

    def mark_current_completed(self, result: str = None) -> None:
        if step := self.get_current_step():
            step.status = StepStatus.COMPLETED
            step.result = result

    def mark_current_failed(self, error: str) -> None:
        if step := self.get_current_step():
            step.status = StepStatus.FAILED
            step.error = error


# Simple patterns that DON'T need LLM - very basic commands only
SIMPLE_PATTERNS = {
    # Open applications - exact matches
    r"^open\s+(chrome|google\s*chrome)\.?$": [
        {"action": "Open Chrome", "tool": "open_application", "params": {"path_or_name": "chrome"}}
    ],
    r"^open\s+(firefox|mozilla)\.?$": [
        {"action": "Open Firefox", "tool": "open_application", "params": {"path_or_name": "firefox"}}
    ],
    r"^open\s+notepad\.?$": [
        {"action": "Open Notepad", "tool": "open_application", "params": {"path_or_name": "notepad"}}
    ],
    r"^open\s+(explorer|file\s*explorer)\.?$": [
        {"action": "Open Explorer", "tool": "open_application", "params": {"path_or_name": "explorer"}}
    ],
    r"^open\s+(cmd|command\s*prompt|terminal)\.?$": [
        {"action": "Open Command Prompt", "tool": "open_application", "params": {"path_or_name": "cmd"}}
    ],
    r"^open\s+(calc|calculator)\.?$": [
        {"action": "Open Calculator", "tool": "open_application", "params": {"path_or_name": "calc"}}
    ],
    r"^open\s+(code|vs\s*code|visual\s*studio\s*code)\.?$": [
        {"action": "Open VS Code", "tool": "open_application", "params": {"path_or_name": "code"}}
    ],

    # Simple key presses
    r"^press\s+enter\.?$": [
        {"action": "Press Enter", "tool": "press_key", "params": {"key": "enter"}}
    ],
    r"^press\s+escape\.?$": [
        {"action": "Press Escape", "tool": "press_key", "params": {"key": "escape"}}
    ],
    r"^press\s+tab\.?$": [
        {"action": "Press Tab", "tool": "press_key", "params": {"key": "tab"}}
    ],

    # Screenshots
    r"^(take\s+)?(a\s+)?screenshot\.?$": [
        {"action": "Take screenshot", "tool": "take_screenshot", "params": {}}
    ],

    # Scroll
    r"^scroll\s+up\.?$": [
        {"action": "Scroll up", "tool": "scroll", "params": {"amount": 5}}
    ],
    r"^scroll\s+down\.?$": [
        {"action": "Scroll down", "tool": "scroll", "params": {"amount": -5}}
    ],

    # Close window
    r"^close(\s+window)?\.?$": [
        {"action": "Close window", "tool": "close_window", "params": {}}
    ],
}


class TaskPlanner:
    """Plans tasks by breaking them into steps using LLM."""

    def __init__(self, llm_client: BaseLLMClient = None):
        self.llm = llm_client or get_llm_client()
        self.prompt_builder = PromptBuilder()

    def create_plan(self, task: str, context: str = "") -> Plan:
        """Create a plan for the given task."""
        logger.info(f"Creating plan for: {task}")

        # Only use simple patterns for VERY simple commands
        steps = self._try_simple_pattern(task)
        if steps:
            logger.info(f"Simple pattern matched: {len(steps)} steps")
            return self._build_plan(task, steps)

        # Use LLM for everything else (including contextual commands)
        logger.info("Using LLM for planning...")
        prompt = self.prompt_builder.build_planner_prompt(request=task, state=context)

        try:
            response = self.llm.generate(prompt)
            logger.debug(f"LLM response: {response.content[:200]}...")

            steps = self._parse_plan_response(response.content)
            logger.info(f"Parsed {len(steps)} steps from LLM")

            return self._build_plan(task, steps)

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            plan = Plan(task=task)
            plan.status = f"planning_failed: {e}"
            return plan

    def _try_simple_pattern(self, task: str) -> Optional[List[Dict]]:
        """Match only very simple patterns that don't need context understanding."""
        task_lower = task.lower().strip()

        # Remove trailing period for matching
        task_clean = task_lower.rstrip('.')

        # Check static patterns for exact simple commands
        for pattern, steps in SIMPLE_PATTERNS.items():
            if re.match(pattern, task_clean, re.IGNORECASE):
                return [step.copy() for step in steps]

        # Handle simple "click on X" - but only if it's truly simple (no extra context)
        # e.g., "click on Chats" but NOT "click on the button in WhatsApp"
        click_match = re.match(r"^click\s+(on\s+)?['\"]?([a-zA-Z0-9\s]{1,20})['\"]?$", task_clean, re.IGNORECASE)
        if click_match:
            target = click_match.group(2).strip()
            # Only match if target is simple (no prepositions indicating context)
            if not any(word in target.lower() for word in ['in', 'on', 'at', 'the', 'of']):
                return [{"action": f"Click on '{target}'", "tool": "click_text", "params": {"text": target}}]

        return None

    def _build_plan(self, task: str, steps: List[Dict]) -> Plan:
        """Build a Plan object from step dictionaries."""
        plan = Plan(task=task)

        for i, step_data in enumerate(steps, 1):
            plan.steps.append(Step(
                number=i,
                description=step_data.get("action", "Unknown action"),
                tool=step_data.get("tool", "unknown"),
                params=step_data.get("params", {})
            ))

        plan.status = "ready" if plan.steps else "empty"
        return plan

    def _parse_plan_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into step list with robust error handling."""
        try:
            start = response.find("[")
            end = response.rfind("]") + 1

            if start >= 0 and end > start:
                json_str = response[start:end]

                # Try to fix common JSON issues
                # Remove trailing commas before ] or }
                import re
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)

                steps = json.loads(json_str)

                if isinstance(steps, list) and len(steps) > 0:
                    valid_steps = []
                    for step in steps:
                        if isinstance(step, dict) and "tool" in step:
                            valid_steps.append(step)

                    if valid_steps:
                        return valid_steps

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")

            # Try to extract individual step objects
            try:
                import re
                # Find all {...} patterns that look like steps
                step_pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*[^{}]*\}'
                matches = re.findall(step_pattern, response, re.DOTALL)

                if matches:
                    valid_steps = []
                    for match in matches:
                        try:
                            step = json.loads(match)
                            if "tool" in step:
                                valid_steps.append(step)
                        except:
                            continue

                    if valid_steps:
                        logger.info(f"Recovered {len(valid_steps)} steps from malformed JSON")
                        return valid_steps
            except Exception as e2:
                logger.warning(f"Recovery attempt failed: {e2}")

        logger.warning("Could not parse JSON, attempting intelligent fallback")

        # Intelligent fallback - try to understand the command intent
        response_lower = response.lower()

        # Look for specific patterns in the response
        if "search" in response_lower and ("youtube" in response_lower or "search box" in response_lower):
            # Extract search query - look for quoted text or text after specific keywords
            import re
            query_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
            if query_match:
                search_query = query_match.group(1)
            else:
                # Try to extract from the original response
                query_match = re.search(r'(?:search|type)[^"]*"([^"]+)"', response_lower)
                search_query = query_match.group(1) if query_match else "search query"

            return [
                {"action": "Click search box", "tool": "click_text", "params": {"text": "Search"}},
                {"action": "Type search query", "tool": "type_text", "params": {"text": search_query}},
                {"action": "Submit search", "tool": "press_key", "params": {"key": "enter"}}
            ]

        # Generic fallback for other tools
        tools = ["click_text", "open_application", "type_text", "press_key", "hotkey", "click", "wait"]
        for tool in tools:
            if tool in response_lower:
                # Try to extract params
                import re
                text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
                if text_match and tool in ["click_text", "type_text"]:
                    return [{"action": f"Execute {tool}", "tool": tool, "params": {"text": text_match.group(1)}}]

        return [{"action": "Could not parse command", "tool": "unknown", "params": {}}]
