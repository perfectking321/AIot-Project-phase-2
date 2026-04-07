"""
VOXCODE Autonomous Agent
ReAct-style agent with perception, reasoning, action, observation, and learning.
Handles vague commands by understanding goals and achieving them step-by-step.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime

from brain.llm import get_llm_client, BaseLLMClient
from config import config

logger = logging.getLogger("voxcode.autonomous")


class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Observation:
    """What the agent perceives from the environment."""
    timestamp: float
    screenshot_description: str
    active_window: str
    visible_elements: List[str]
    raw_screenshot: Any = None  # PIL Image


@dataclass
class Thought:
    """Agent's reasoning about what to do."""
    analysis: str  # What the agent understands about current state
    plan: str  # What it intends to do
    action: str  # Specific action to take
    tool: str  # Tool to use
    params: Dict[str, Any]  # Parameters for the tool
    confidence: float = 0.8


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    message: str
    observation: Optional[Observation] = None


@dataclass
class Experience:
    """A complete experience for learning."""
    task: str
    steps: List[Dict[str, Any]]
    success: bool
    reflections: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearnedProcedure:
    """A procedure learned from demonstration or experience."""
    name: str
    trigger_phrases: List[str]  # What commands trigger this
    steps: List[Dict[str, Any]]
    success_criteria: str
    times_used: int = 0
    success_rate: float = 1.0


class AutonomousAgent:
    """
    Autonomous agent that can handle complex, vague commands.
    Uses ReAct (Reasoning + Acting) pattern with memory and self-improvement.
    """

    # System prompt that gives the agent world knowledge
    SYSTEM_PROMPT = """You are an autonomous AI agent controlling a Windows computer.
You can see the screen, understand what's happening, and take actions to achieve goals.

WORLD KNOWLEDGE - How Things Work:
1. To search the web: Open browser (Chrome/Edge/Firefox) → Navigate to search engine → Type query → Press Enter
2. To open applications: Use Start menu, taskbar, or Win+R run dialog
3. To navigate folders: Use File Explorer, double-click to open folders
4. To create files: Right-click → New, or use application's File → New
5. To save: Ctrl+S or File → Save
6. To type text: Click on text field first, then type
7. To interact with UI: Click buttons, links, icons; use keyboard shortcuts

USER PREFERENCES:
- When Chrome shows profile selection ("Who's using Chrome?"), ALWAYS click on "Perfect King" profile
- The user's name is Perfect King - look for this profile when Chrome asks to choose

COMMON PATTERNS:
- "Search for X" → Browser → Google/Bing → Type X → Enter → View results
- "Go to X website" → Browser → Address bar → Type URL → Enter
- "Open X application" → Start menu OR Win+R → Type app name → Enter
- "Create file in folder" → Navigate to folder → Right-click → New → File type
- "Play video on YouTube" → Browser → YouTube → Search → Click video

YOU CAN SEE:
- Current screen state (what windows are open, what's visible)
- UI elements (buttons, text fields, icons, links)
- Application states

YOU CAN DO:
- Click at coordinates or on elements
- Type text
- Press keys and keyboard shortcuts
- Open applications
- Wait for things to load

THINK STEP BY STEP:
1. Observe the current screen state
2. Understand what you see
3. Decide the next single action to take
4. Execute and observe the result
5. Repeat until goal is achieved

ALWAYS explain your reasoning before acting.
"""

    def __init__(
        self,
        vision=None,
        tools=None,
        llm: BaseLLMClient = None,
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None,
    ):
        self.llm = llm or get_llm_client()
        self.vision = vision
        self.tools = tools
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)

        # State
        self.state = AgentState.IDLE
        self.current_goal = ""
        self.history: List[Dict] = []  # Current task history
        self.max_cycles = 15  # Max perception-action cycles
        self._stop_requested = False  # Flag to stop execution

        # Memory systems
        self.episodic_memory: List[Experience] = []  # Past experiences
        self.procedures: Dict[str, LearnedProcedure] = {}  # Learned procedures
        self.reflections: List[str] = []  # Lessons learned

        # Load saved memories
        self._load_memory()

        logger.info("AutonomousAgent initialized with world knowledge")

    def stop(self):
        """Request the agent to stop current execution."""
        self._stop_requested = True
        logger.info("Agent stop requested")

    def _load_memory(self):
        """Load memories from disk."""
        try:
            import os
            memory_file = os.path.join(os.path.dirname(__file__), "..", "memory.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    self.reflections = data.get("reflections", [])
                    # Load procedures
                    for name, proc_data in data.get("procedures", {}).items():
                        self.procedures[name] = LearnedProcedure(**proc_data)
                    logger.info(f"Loaded {len(self.reflections)} reflections, {len(self.procedures)} procedures")
        except Exception as e:
            logger.warning(f"Could not load memory: {e}")

    def _save_memory(self):
        """Save memories to disk."""
        try:
            import os
            memory_file = os.path.join(os.path.dirname(__file__), "..", "memory.json")
            data = {
                "reflections": self.reflections[-50:],  # Keep last 50
                "procedures": {
                    name: {
                        "name": p.name,
                        "trigger_phrases": p.trigger_phrases,
                        "steps": p.steps,
                        "success_criteria": p.success_criteria,
                        "times_used": p.times_used,
                        "success_rate": p.success_rate,
                    }
                    for name, p in self.procedures.items()
                }
            }
            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Memory saved")
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")

    def process_command(self, command: str) -> str:
        """
        Process a voice command autonomously.
        This is the main entry point.
        """
        self.current_goal = command
        self.history = []
        self.state = AgentState.PERCEIVING
        self._stop_requested = False  # Reset stop flag

        logger.info(f"Processing command: {command}")
        self.on_status(f"Understanding: {command}")

        # Check if we have a learned procedure for this
        procedure = self._find_matching_procedure(command)
        if procedure:
            logger.info(f"Found learned procedure: {procedure.name}")
            return self._execute_procedure(procedure)

        # Autonomous ReAct loop
        cycle = 0
        while cycle < self.max_cycles and self.state not in [AgentState.COMPLETE, AgentState.FAILED]:
            # Check for stop request
            if self._stop_requested:
                logger.info("Agent stopped by request")
                self.state = AgentState.FAILED
                return "Stopped by user"

            cycle += 1
            logger.info(f"=== Cycle {cycle}/{self.max_cycles} ===")

            try:
                # 1. PERCEIVE - Understand current screen state
                self.state = AgentState.PERCEIVING
                observation = self._perceive()
                logger.info(f"Perceived: {observation.active_window}")

                # 2. THINK - Reason about what to do
                self.state = AgentState.THINKING
                thought = self._think(observation)
                logger.info(f"Thought: {thought.plan}")

                # Check if goal is achieved
                if self._is_goal_achieved(thought):
                    self.state = AgentState.COMPLETE
                    self.on_step(cycle, "Goal achieved!", "done")
                    break

                # 3. ACT - Execute the decided action
                self.state = AgentState.ACTING
                self.on_step(cycle, thought.action, "running")
                result = self._act(thought)
                logger.info(f"Action result: {result.success} - {result.message}")

                # 4. OBSERVE - See what happened
                self.state = AgentState.OBSERVING
                self.history.append({
                    "cycle": cycle,
                    "observation": observation.screenshot_description,
                    "thought": thought.plan,
                    "action": f"{thought.tool}({thought.params})",
                    "result": result.message,
                    "success": result.success
                })

                if result.success:
                    self.on_step(cycle, thought.action, "done")
                else:
                    self.on_step(cycle, f"{thought.action} - {result.message}", "failed")
                    # 5. REFLECT on failure (skip if rate limited to save API calls)
                    if "429" not in str(result.message):
                        self.state = AgentState.REFLECTING
                        reflection = self._reflect(thought, result)
                        self.reflections.append(reflection)
                        logger.info(f"Reflection: {reflection}")

                # Small delay between cycles (reduced for speed)
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
                self.on_step(cycle, f"Error: {str(e)}", "failed")
                time.sleep(1)  # Brief delay on error

        # Final state
        if self.state == AgentState.COMPLETE:
            self._record_experience(success=True)
            self._save_memory()
            return f"Successfully completed: {command}"
        else:
            self.state = AgentState.FAILED
            self._record_experience(success=False)
            self._save_memory()
            return f"Could not complete: {command}"

    def _perceive(self) -> Observation:
        """Perceive the current screen state."""
        # Get active window first (doesn't need vision)
        active_window = "Unknown"
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win:
                active_window = win.title
        except:
            pass

        if not self.vision:
            # Fallback without vision - still useful!
            return Observation(
                timestamp=time.time(),
                screenshot_description=f"Active window: {active_window}",
                active_window=active_window,
                visible_elements=[]
            )

        try:
            # Take screenshot using correct method name
            screenshot = self.vision.take_screenshot()

            # Find all text elements on screen
            elements = self.vision.find_all_text()

            # Build description from elements
            element_texts = [e.text for e in elements if hasattr(e, 'text') and e.text]
            description = f"Active window: {active_window}. Visible elements: {', '.join(element_texts[:20])}"

            return Observation(
                timestamp=time.time(),
                screenshot_description=description,
                active_window=active_window,
                visible_elements=element_texts,
                raw_screenshot=screenshot
            )

        except Exception as e:
            logger.error(f"Perception error: {e}")
            return Observation(
                timestamp=time.time(),
                screenshot_description=f"Active window: {active_window}. (Perception limited: {e})",
                active_window=active_window,
                visible_elements=[]
            )

    def _think(self, observation: Observation) -> Thought:
        """Reason about what to do next using ReAct pattern."""

        # Build context from history
        history_text = ""
        for h in self.history[-5:]:  # Last 5 actions
            history_text += f"\nStep {h['cycle']}: {h['action']} → {h['result']}"

        # Include relevant reflections/lessons
        relevant_reflections = self._get_relevant_reflections()

        prompt = f"""{self.SYSTEM_PROMPT}

CURRENT GOAL: {self.current_goal}

WHAT I SEE NOW:
{observation.screenshot_description}

Active Window: {observation.active_window}
Visible Elements: {', '.join(observation.visible_elements[:30])}

WHAT I'VE DONE SO FAR:{history_text if history_text else " Nothing yet"}

LESSONS FROM PAST EXPERIENCE:
{relevant_reflections if relevant_reflections else "None yet"}

Based on my goal and current state, I need to decide the next action.

Think step by step:
1. What is my goal?
2. What do I see on screen?
3. What progress have I made?
4. What should I do next?
5. What specific tool and parameters should I use?

Available tools:
- click_text(text): Click on visible text/button/link
- click(x, y): Click at screen coordinates
- type_text(text): Type text (make sure a text field is focused first!)
- press_key(key): Press a key like "enter", "escape", "tab"
- hotkey(keys): Press key combination like ["ctrl", "c"]
- open_application(name): Open an application by name
- wait(seconds): Wait for something to load
- scroll(amount): Scroll up (positive) or down (negative)

Respond in this exact JSON format:
{{
    "analysis": "What I understand about the current state",
    "plan": "What I intend to do and why",
    "action": "Brief description of the action",
    "tool": "tool_name",
    "params": {{"param": "value"}},
    "goal_achieved": false
}}

If the goal is fully achieved, set "goal_achieved": true.
"""

        # Try with exponential backoff for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(prompt)
                return self._parse_thought(response.content)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (attempt + 1) * 1  # 1, 2, 3 seconds (reduced)
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Thinking error: {e}")
                    break

        # Fallback after all retries failed
        return Thought(
            analysis="Error in reasoning",
            plan="Retry or fail",
            action="none",
            tool="wait",
            params={"seconds": 1}
        )

    def _parse_thought(self, response: str) -> Thought:
        """Parse LLM response into structured thought."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                return Thought(
                    analysis=data.get("analysis", ""),
                    plan=data.get("plan", ""),
                    action=data.get("action", "Unknown action"),
                    tool=data.get("tool", "wait"),
                    params=data.get("params", {}),
                    confidence=0.8
                )
        except Exception as e:
            logger.warning(f"Could not parse thought: {e}")

        # Fallback
        return Thought(
            analysis="Could not parse response",
            plan="Will try a safe action",
            action="Wait",
            tool="wait",
            params={"seconds": 1}
        )

    def _is_goal_achieved(self, thought: Thought) -> bool:
        """Check if the agent thinks the goal is achieved."""
        try:
            # Check if the thought indicates completion
            if "goal_achieved" in thought.analysis.lower():
                return True
            if thought.tool == "none" or thought.action.lower() in ["done", "complete", "finished"]:
                return True
        except:
            pass
        return False

    def _act(self, thought: Thought) -> ActionResult:
        """Execute the decided action."""
        if not self.tools:
            return ActionResult(success=False, message="Tools not available")

        tool_name = thought.tool
        params = thought.params.copy()  # Make a copy to modify

        # Normalize parameter names (LLM might use different names)
        param_mappings = {
            "open_application": {"name": "path_or_name", "app": "path_or_name", "application": "path_or_name"},
            "type_text": {"content": "text", "string": "text"},
            "press_key": {"button": "key"},
            "click_text": {"target": "text", "element": "text"},
            "hotkey": {"key_combo": "keys", "shortcut": "keys"},
        }

        if tool_name in param_mappings:
            for old_key, new_key in param_mappings[tool_name].items():
                if old_key in params and new_key not in params:
                    params[new_key] = params.pop(old_key)

        # Map tool names to actual functions
        tool_map = {
            "click_text": self.tools.click_text,
            "click": self.tools.click,
            "type_text": self.tools.type_text,
            "press_key": self.tools.press_key,
            "hotkey": self.tools.hotkey,
            "open_application": self.tools.open_application,
            "wait": self.tools.wait,
            "scroll": self.tools.scroll,
        }

        if tool_name not in tool_map:
            return ActionResult(success=False, message=f"Unknown tool: {tool_name}")

        try:
            tool_func = tool_map[tool_name]
            result = tool_func(**params)

            if hasattr(result, 'success'):
                return ActionResult(success=result.success, message=result.message)
            else:
                return ActionResult(success=True, message=str(result))

        except Exception as e:
            return ActionResult(success=False, message=str(e))

    def _reflect(self, thought: Thought, result: ActionResult) -> str:
        """Reflect on what went wrong and learn from it."""
        prompt = f"""An action failed. Learn from this mistake.

Goal: {self.current_goal}
Action attempted: {thought.action} using {thought.tool}({thought.params})
Error: {result.message}

What went wrong and what should I do differently next time?
Be specific and actionable. One sentence.
"""
        # Simple reflection without retry (non-critical)
        try:
            response = self.llm.generate(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Reflection failed (rate limit?): {e}")
            return f"Action {thought.tool} failed: {result.message}"

    def _get_relevant_reflections(self) -> str:
        """Get reflections relevant to current task."""
        if not self.reflections:
            return ""

        # Simple: return last few reflections
        # TODO: Use embeddings for semantic similarity
        recent = self.reflections[-5:]
        return "\n".join(f"- {r}" for r in recent)

    def _find_matching_procedure(self, command: str) -> Optional[LearnedProcedure]:
        """Find a learned procedure that matches this command."""
        command_lower = command.lower()

        for name, procedure in self.procedures.items():
            for trigger in procedure.trigger_phrases:
                if trigger.lower() in command_lower:
                    return procedure

        return None

    def _execute_procedure(self, procedure: LearnedProcedure) -> str:
        """Execute a learned procedure."""
        logger.info(f"Executing learned procedure: {procedure.name}")
        self.on_status(f"Using learned procedure: {procedure.name}")

        for i, step in enumerate(procedure.steps, 1):
            self.on_step(i, step.get("action", "Step"), "running")

            try:
                result = self._act(Thought(
                    analysis="Executing learned procedure",
                    plan=step.get("action", ""),
                    action=step.get("action", ""),
                    tool=step.get("tool", "wait"),
                    params=step.get("params", {})
                ))

                if result.success:
                    self.on_step(i, step.get("action", "Step"), "done")
                else:
                    self.on_step(i, f"{step.get('action', 'Step')} - {result.message}", "failed")
                    # Fall back to autonomous mode
                    logger.info("Procedure step failed, switching to autonomous mode")
                    return self.process_command(self.current_goal)

                time.sleep(0.3)

            except Exception as e:
                logger.error(f"Procedure step error: {e}")

        procedure.times_used += 1
        self._save_memory()
        return f"Completed: {procedure.name}"

    def _record_experience(self, success: bool):
        """Record this experience for future learning."""
        experience = Experience(
            task=self.current_goal,
            steps=self.history,
            success=success,
            reflections=self.reflections[-3:] if not success else []
        )
        self.episodic_memory.append(experience)

        # Keep memory bounded
        if len(self.episodic_memory) > 100:
            self.episodic_memory = self.episodic_memory[-100:]

    # ==================== LEARNING MODE ====================

    def start_learning_mode(self, task_name: str):
        """Start recording user actions to learn a new procedure."""
        self.state = AgentState.LEARNING
        self._learning_task = task_name
        self._learning_steps = []
        self._learning_start_time = time.time()
        logger.info(f"Learning mode started for: {task_name}")
        return f"Learning mode started. Show me how to: {task_name}"

    def record_learning_step(self, action: str, tool: str, params: Dict):
        """Record a step during learning mode."""
        if self.state != AgentState.LEARNING:
            return

        step = {
            "action": action,
            "tool": tool,
            "params": params,
            "timestamp": time.time() - self._learning_start_time
        }
        self._learning_steps.append(step)
        logger.info(f"Recorded learning step: {action}")

    def finish_learning_mode(self, trigger_phrases: List[str] = None):
        """Finish learning and save the procedure."""
        if self.state != AgentState.LEARNING:
            return "Not in learning mode"

        if not self._learning_steps:
            self.state = AgentState.IDLE
            return "No steps recorded"

        # Create procedure
        procedure = LearnedProcedure(
            name=self._learning_task,
            trigger_phrases=trigger_phrases or [self._learning_task],
            steps=self._learning_steps,
            success_criteria=f"Complete {self._learning_task}"
        )

        self.procedures[self._learning_task] = procedure
        self._save_memory()

        self.state = AgentState.IDLE
        logger.info(f"Learned procedure: {self._learning_task} with {len(self._learning_steps)} steps")

        return f"Learned '{self._learning_task}' with {len(self._learning_steps)} steps"

    def cancel_learning_mode(self):
        """Cancel learning mode without saving."""
        self.state = AgentState.IDLE
        self._learning_steps = []
        return "Learning cancelled"


# Convenience function
def create_autonomous_agent(vision=None, tools=None, **kwargs) -> AutonomousAgent:
    """Create an autonomous agent instance."""
    return AutonomousAgent(vision=vision, tools=tools, **kwargs)
