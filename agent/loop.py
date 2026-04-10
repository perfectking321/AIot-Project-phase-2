"""
VOXCODE Agent Loop
Main execution loop with comprehensive logging.
"""

import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from config import config
from brain.llm import get_llm_client
from brain.prompts import PromptBuilder
from agent.tools import WindowsTools, ToolResult, ToolStatus
from agent.planner import TaskPlanner, Plan, Step, StepStatus

logger = logging.getLogger("voxcode.agent")


class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentContext:
    """Runtime context for the agent."""
    state: AgentState = AgentState.IDLE
    current_task: Optional[str] = None
    current_plan: Optional[Plan] = None
    last_result: Optional[ToolResult] = None
    error_count: int = 0
    history: list = field(default_factory=list)


class AgentLoop:
    """
    Main agent execution loop.
    Coordinates between voice input, LLM planning, and tool execution.
    """

    def __init__(
        self,
        on_state_change: Callable[[AgentState], None] = None,
        on_step_complete: Callable[[Step, ToolResult], None] = None,
        on_message: Callable[[str], None] = None,
        vision=None  # Pre-loaded vision instance for faster OCR
    ):
        logger.info("Initializing AgentLoop...")

        self.llm = get_llm_client()
        self.tools = WindowsTools(vision_instance=vision)
        self.planner = TaskPlanner(self.llm)
        self.prompt_builder = PromptBuilder()

        self.on_state_change = on_state_change or (lambda s: None)
        self.on_step_complete = on_step_complete or (lambda s, r: None)
        self.on_message = on_message or (lambda m: None)

        self.context = AgentContext()
        self._running = False
        self._max_iterations = config.agent.max_iterations
        self._retry_attempts = config.agent.retry_attempts

        logger.info("AgentLoop initialized")

    def _set_state(self, state: AgentState) -> None:
        self.context.state = state
        self.on_state_change(state)
        logger.debug(f"State changed to: {state.value}")

    def process_command(self, command: str) -> str:
        """Process a voice command and execute it."""
        logger.info(f"Processing command: {command}")

        self.context.current_task = command
        self.context.error_count = 0

        # PLANNING phase
        self._set_state(AgentState.PLANNING)
        context_summary = self._get_context_summary()
        self.on_message(f"Planning: {command}")
        logger.info("Creating plan...")

        try:
            plan = self.planner.create_plan(command, context_summary)
            self.context.current_plan = plan
            logger.info(f"Plan created: {len(plan.steps)} steps")

            for i, step in enumerate(plan.steps):
                logger.debug(f"  Step {i+1}: {step.tool} - {step.description}")

        except Exception as e:
            logger.error(f"Planning failed: {e}", exc_info=True)
            self._set_state(AgentState.ERROR)
            return f"Planning failed: {e}"

        if plan.status.startswith("planning_failed"):
            logger.error(f"Plan status indicates failure: {plan.status}")
            self._set_state(AgentState.ERROR)
            return f"Failed to create plan: {plan.status}"

        step_count = len(plan.steps)
        self.on_message(f"Plan created with {step_count} steps")

        if step_count > 8 and config.agent.use_reactive_for_complex:
            logger.info(f"Complex plan ({step_count} steps) - routing to ReactiveAgent")
            self.on_message(f"Complex task detected ({step_count} steps). Switching to reactive mode.")
            from agent.reactive_loop import ReactiveAgent

            reactive = ReactiveAgent(
                on_message=self.on_message,
                max_iterations=config.agent.reactive_max_iterations,
                verify_actions=config.agent.verify_actions,
            )
            return reactive.process_goal(command)

        # EXECUTING phase
        self._set_state(AgentState.EXECUTING)
        logger.info("Executing plan...")

        result = self._execute_plan(plan)

        self._set_state(AgentState.COMPLETED)
        self.context.history.append({"task": command, "result": result})

        logger.info(f"Command completed: {result}")
        return result

    def _execute_plan(self, plan: Plan) -> str:
        """Execute all steps in a plan."""
        results = []

        while not plan.is_complete and self.context.error_count < self._max_iterations:
            step = plan.get_current_step()
            if not step:
                logger.warning("No current step found")
                break

            step.status = StepStatus.RUNNING
            self.on_message(f"Step {step.number}: {step.description}")
            logger.info(f"Executing step {step.number}: {step.tool}({step.params})")

            result = self._execute_step(step)
            self.context.last_result = result

            logger.info(f"Step {step.number} result: {result.status.value} - {result.message}")

            if result.success:
                plan.mark_current_completed(result.message)
                results.append(f"Step {step.number}: {result.message}")
                self.on_step_complete(step, result)
                plan.advance()
            else:
                if not self._handle_step_failure(plan, step, result):
                    break

        summary = self._summarize_execution(plan, results)
        return summary

    def _execute_step(self, step: Step) -> ToolResult:
        """Execute a single step."""
        tool_method = getattr(self.tools, step.tool, None)

        if not tool_method:
            logger.error(f"Unknown tool: {step.tool}")
            return ToolResult(ToolStatus.FAILURE, f"Unknown tool: {step.tool}")

        try:
            logger.debug(f"Calling {step.tool} with params: {step.params}")
            return tool_method(**step.params)
        except TypeError as e:
            logger.error(f"Invalid params for {step.tool}: {e}")
            return ToolResult(ToolStatus.FAILURE, f"Invalid params: {e}")
        except Exception as e:
            logger.error(f"Tool {step.tool} failed: {e}", exc_info=True)
            return ToolResult(ToolStatus.FAILURE, str(e))

    def _handle_step_failure(self, plan: Plan, step: Step, result: ToolResult) -> bool:
        """Handle a failed step. Returns True to continue, False to abort."""
        self.context.error_count += 1
        plan.mark_current_failed(result.message)

        logger.warning(f"Step {step.number} failed (attempt {self.context.error_count}/{self._retry_attempts})")

        if self.context.error_count >= self._retry_attempts:
            self.on_message("Max retries reached. Aborting.")
            logger.error("Max retries reached, aborting execution")
            self._set_state(AgentState.ERROR)
            return False

        self.on_message(f"Step failed: {result.message}. Retrying...")
        step.status = StepStatus.PENDING
        return True

    def _summarize_execution(self, plan: Plan, results: list) -> str:
        """Create a summary of the execution."""
        completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
        total = len(plan.steps)

        logger.info(f"Execution summary: {completed}/{total} steps completed")

        if completed == total:
            return f"Task completed successfully. {completed}/{total} steps done."
        else:
            failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
            if failed:
                return f"Task partially completed. {completed}/{total} steps. Failed: {failed[0].error}"
            return f"Task incomplete. {completed}/{total} steps completed."

    def _get_context_summary(self) -> str:
        """Get current context for planning, including active window and screen elements."""
        parts = []

        # Get active window info - crucial for context awareness
        try:
            active_win = self.tools.get_active_window()
            if active_win.success and active_win.data:
                window_title = active_win.data.get("title", "")
                if window_title:
                    parts.append(f"Active window: {window_title}")

                    # Add app-specific hints
                    title_lower = window_title.lower()
                    if "notepad" in title_lower:
                        parts.append("(Notepad is open and ready for typing)")
                    elif "chrome" in title_lower or "edge" in title_lower or "firefox" in title_lower:
                        parts.append("(Browser is active)")
                    elif "whatsapp" in title_lower:
                        parts.append("(WhatsApp is open)")
        except Exception as e:
            logger.debug(f"Could not get active window: {e}")

        # Try to get screen elements via OmniParser for better context
        try:
            if hasattr(self.tools, '_use_omniparser') and self.tools._use_omniparser:
                parse_result = self.tools.parse_screen()
                if parse_result.success and parse_result.data:
                    # Add a summary of visible elements
                    elements = parse_result.data.get("elements", [])
                    if elements:
                        # Get key interactable elements
                        interactable = [e for e in elements if e.get("interactable", True)][:10]
                        if interactable:
                            elem_summary = ", ".join([f'"{e["label"]}"' for e in interactable[:5]])
                            parts.append(f"Visible elements: {elem_summary}")
        except Exception as e:
            logger.debug(f"Could not parse screen: {e}")

        if self.context.last_result:
            parts.append(f"Last action: {self.context.last_result.message}")
        if self.context.history:
            recent = self.context.history[-3:]
            recent_tasks = [h["task"] for h in recent]
            parts.append(f"Recent tasks: {recent_tasks}")
        return "; ".join(parts) if parts else "No prior context"

    def stop(self) -> None:
        """Stop the agent loop."""
        logger.info("Stopping agent loop")
        self._running = False
        self._set_state(AgentState.IDLE)

    def reset(self) -> None:
        """Reset agent state."""
        logger.info("Resetting agent state")
        self.context = AgentContext()
        self.llm.clear_history()
