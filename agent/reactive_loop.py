"""
VOXCODE Reactive Agent Loop
Real-time perceive-think-act-verify cycle with self-correction.
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue

logger = logging.getLogger("voxcode.reactive")

try:
    from PIL import Image
    import pyautogui
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False

from config import config
from brain.llm import get_llm_client
from agent.tools import WindowsTools, ToolResult, ToolStatus
from agent.omniparser import OmniParser, ParsedScreen, get_omniparser


class ReactiveActionResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"
    NEEDS_ALTERNATIVE = "needs_alternative"
    GOAL_ACHIEVED = "goal_achieved"


@dataclass
class ReactiveScreenState:
    """Captured state of the screen."""
    timestamp: float
    screenshot: Optional[Image.Image] = None
    parsed: Optional[ParsedScreen] = None
    active_window: str = ""
    visible_elements: List[str] = field(default_factory=list)
    visible_apps: List[str] = field(default_factory=list)  # Apps detected on screen

    def describe(self) -> str:
        """Get a text description of the screen state."""
        parts = []
        if self.active_window:
            parts.append(f"Window: {self.active_window}")
        if self.visible_apps:
            parts.append(f"Apps visible: {', '.join(self.visible_apps)}")
        if self.visible_elements:
            elements = ", ".join(self.visible_elements[:10])
            parts.append(f"Elements: {elements}")
        return "; ".join(parts) if parts else "Unknown screen state"


@dataclass
class ActionAttempt:
    """Record of an action attempt."""
    action: str
    tool: str
    params: Dict[str, Any]
    result: ReactiveActionResult
    error: str = ""
    before_state: Optional[ReactiveScreenState] = None
    after_state: Optional[ReactiveScreenState] = None


class ReactiveAgent:
    """
    Reactive agent with real-time perception and self-correction.

    Uses a perceive-think-act-verify loop:
    1. PERCEIVE: Capture screen state
    2. THINK: Decide next action based on goal and current state
    3. ACT: Execute the action
    4. VERIFY: Check if action succeeded, decide next step
    """

    # Prompts for reactive reasoning
    DECIDE_ACTION_PROMPT = """You are a Windows automation agent. Decide the SINGLE next action.

GOAL: {goal}

SCREEN STATE:
- Window: {active_window}
- Visible Apps: {visible_apps}
- UI Elements: {visible_elements}

PREVIOUS ACTIONS:
{action_history}

AVAILABLE TOOLS (use exact names):
- click_text: Click text/button. Example: {{"action": "Click Search", "tool": "click_text", "params": {{"text": "Search"}}}}
- type_text: Type text. Example: {{"action": "Type query", "tool": "type_text", "params": {{"text": "hello"}}}}
- press_key: Press key. Example: {{"action": "Press Enter", "tool": "press_key", "params": {{"key": "enter"}}}}
- hotkey: Key combo. Example: {{"action": "New tab", "tool": "hotkey", "params": {{"keys": ["ctrl", "t"]}}}}
- open_application: Open app. Example: {{"action": "Open Chrome", "tool": "open_application", "params": {{"path_or_name": "chrome"}}}}
- wait: Wait. Example: {{"action": "Wait", "tool": "wait", "params": {{"seconds": 2}}}}

RULES:
1. Return ONLY ONE action
2. If app is visible, DON'T open it again - just interact with it
3. If goal is done: {{"action": "GOAL_COMPLETE", "tool": "none", "params": {{}}}}

OUTPUT FORMAT (JSON only, no other text):
{{"action": "description", "tool": "tool_name", "params": {{"key": "value"}}}}"""

    VERIFY_ACTION_PROMPT = """Did the action succeed? Compare the screen before and after.

ACTION TAKEN: {action}

SCREEN BEFORE:
- Window: {before_window}
- Elements: {before_elements}

SCREEN AFTER:
- Window: {after_window}
- Elements: {after_elements}

GOAL: {goal}

Respond with ONLY JSON:
{{
  "success": true/false,
  "reason": "brief explanation",
  "goal_achieved": true/false,
  "suggestion": "what to try next if failed"
}}
"""

    def __init__(
        self,
        on_message: Callable[[str], None] = None,
        on_state_change: Callable[[str], None] = None,
        max_iterations: int = 20,
        verify_actions: bool = True
    ):
        self.llm = get_llm_client()
        self.tools = WindowsTools(use_omniparser=True)
        self.parser = get_omniparser()

        self.on_message = on_message or (lambda m: None)
        self.on_state_change = on_state_change or (lambda s: None)

        self.max_iterations = max_iterations
        self.verify_actions = verify_actions

        # State tracking
        self.action_history: List[ActionAttempt] = []
        self.current_goal: str = ""
        self.is_running = False

        # Screen monitoring
        self._monitor_thread = None
        self._screen_queue = Queue(maxsize=5)
        self._monitor_interval = 0.5  # seconds

        logger.info("ReactiveAgent initialized")

    def capture_screen_state(self) -> ReactiveScreenState:
        """Capture the current screen state with app detection."""
        state = ReactiveScreenState(timestamp=time.time())

        try:
            # Get screenshot
            state.screenshot = pyautogui.screenshot()

            # Parse with OmniParser
            state.parsed = self.parser.parse_screen(state.screenshot)

            # Get active window (focused)
            try:
                win = pyautogui.getActiveWindow()
                if win:
                    state.active_window = win.title
            except:
                pass

            # Get visible element labels
            if state.parsed:
                state.visible_elements = [
                    e.label for e in state.parsed.elements
                    if e.is_interactable and e.label != "icon"
                ][:20]

                # Detect visible apps from screen content
                state.visible_apps = self._detect_visible_apps(state.parsed, state.visible_elements)

        except Exception as e:
            logger.error(f"Failed to capture screen state: {e}")

        return state

    def _detect_visible_apps(self, parsed: ParsedScreen, elements: List[str]) -> List[str]:
        """Detect which apps are visible on screen based on UI elements."""
        visible_apps = []
        all_text = " ".join(elements).lower()

        # App detection patterns
        app_indicators = {
            "WhatsApp": ["whatsapp", "type a message", "chats", "status", "communities"],
            "Chrome": ["chrome", "new tab", "google", "search google"],
            "Firefox": ["firefox", "mozilla"],
            "Edge": ["edge", "microsoft edge"],
            "YouTube": ["youtube", "subscribe", "watch later", "shorts"],
            "Notepad": ["notepad", "untitled - notepad", "edit", "format", "view"],
            "VS Code": ["visual studio code", "vscode", "explorer", "extensions"],
            "File Explorer": ["file explorer", "this pc", "documents", "downloads"],
            "Settings": ["settings", "system", "bluetooth", "network"],
            "Discord": ["discord", "servers", "friends", "nitro"],
            "Telegram": ["telegram", "saved messages"],
            "Spotify": ["spotify", "play", "shuffle", "queue"],
        }

        # Context indicators (specific UI states)
        context_indicators = {
            "WhatsApp Chat Open": ["type a message", "bharwa", "chats"],
            "YouTube Search": ["search youtube", "filter"],
            "Browser Address Bar": ["search google or type a url"],
        }

        for app, indicators in app_indicators.items():
            for indicator in indicators:
                if indicator in all_text:
                    if app not in visible_apps:
                        visible_apps.append(app)
                    break

        # Detect specific contexts
        for context, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in all_text:
                    if context not in visible_apps:
                        visible_apps.append(context)
                    break

        # Also check window titles from parsed elements
        for elem in parsed.elements:
            label_lower = elem.label.lower()
            for app, indicators in app_indicators.items():
                if app.lower() in label_lower and app not in visible_apps:
                    visible_apps.append(app)
                    break

        return visible_apps

    def decide_next_action(self, goal: str, state: ReactiveScreenState) -> Dict[str, Any]:
        """Use LLM to decide the next action based on current screen state."""

        # Format action history
        history_lines = []
        for i, attempt in enumerate(self.action_history[-5:], 1):  # Last 5 actions
            status = "OK" if attempt.result == ReactiveActionResult.SUCCESS else "FAILED"
            history_lines.append(f"{i}. [{status}] {attempt.action}")

        history_str = "\n".join(history_lines) if history_lines else "(No actions yet)"

        # Format visible apps
        visible_apps_str = ", ".join(state.visible_apps) if state.visible_apps else "None detected"

        # Format visible elements (filter out just "icon")
        meaningful_elements = [e for e in state.visible_elements if e.lower() != "icon"][:15]
        elements_str = ", ".join(meaningful_elements) if meaningful_elements else "None detected"

        # Build prompt
        prompt = self.DECIDE_ACTION_PROMPT.format(
            goal=goal,
            active_window=state.active_window or "Unknown",
            visible_apps=visible_apps_str,
            visible_elements=elements_str,
            action_history=history_str
        )

        try:
            response = self.llm.generate(prompt)
            content = response.content.strip()

            # Parse JSON response
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])

                # Validate the result has required fields
                if "tool" not in result:
                    logger.warning(f"Missing 'tool' in response: {result}")
                    # Try to infer tool from action
                    action = result.get("action", "").lower()
                    if "click" in action:
                        result["tool"] = "click_text"
                    elif "type" in action:
                        result["tool"] = "type_text"
                    elif "wait" in action:
                        result["tool"] = "wait"
                    else:
                        result["tool"] = "wait"

                # Validate tool is a known tool
                valid_tools = ["click_text", "type_text", "press_key", "hotkey",
                               "open_application", "wait", "scroll", "none"]
                if result.get("tool") not in valid_tools:
                    logger.warning(f"Unknown tool '{result.get('tool')}', defaulting to wait")
                    result["tool"] = "wait"
                    result["params"] = {"seconds": 1}

                return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {content[:200]}")
        except Exception as e:
            logger.error(f"Failed to decide action: {e}")

        return {"action": "wait", "tool": "wait", "params": {"seconds": 1}}

    def verify_action_result(
        self,
        action: str,
        before: ReactiveScreenState,
        after: ReactiveScreenState,
        goal: str
    ) -> Dict[str, Any]:
        """Verify if an action succeeded by comparing before/after states."""

        prompt = self.VERIFY_ACTION_PROMPT.format(
            action=action,
            before_window=before.active_window,
            before_elements=", ".join(before.visible_elements[:10]),
            after_window=after.active_window,
            after_elements=", ".join(after.visible_elements[:10]),
            goal=goal
        )

        try:
            response = self.llm.generate(prompt)
            content = response.content.strip()

            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])

        except Exception as e:
            logger.error(f"Failed to verify action: {e}")

        # Default: assume success if we can't verify
        return {"success": True, "goal_achieved": False}

    def execute_action(self, tool: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a single action using the tools."""
        tool_method = getattr(self.tools, tool, None)

        if not tool_method:
            return ToolResult(ToolStatus.FAILURE, f"Unknown tool: {tool}")

        try:
            return tool_method(**params)
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def process_goal(self, goal: str) -> str:
        """
        Process a goal using reactive perceive-think-act-verify loop.

        Returns a summary of what happened.
        """
        self.current_goal = goal
        self.action_history = []
        self.is_running = True

        self.on_message(f"Starting: {goal}")
        self.on_state_change("perceiving")

        iteration = 0
        while self.is_running and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # 1. PERCEIVE - Capture current screen state
            self.on_state_change("perceiving")
            before_state = self.capture_screen_state()

            self.on_message(f"[{iteration}] Analyzing screen...")
            logger.info(f"Screen: {before_state.describe()}")

            # Log detected apps for debugging
            if before_state.visible_apps:
                self.on_message(f"    Detected apps: {', '.join(before_state.visible_apps)}")

            # 2. THINK - Decide next action
            self.on_state_change("thinking")
            decision = self.decide_next_action(goal, before_state)

            # Check for goal completion
            if decision.get("action") == "GOAL_COMPLETE":
                self.on_message(f"Goal achieved: {decision.get('reason', 'Success')}")
                self.is_running = False
                return f"Completed: {decision.get('reason', goal)}"

            action_desc = decision.get("action", "unknown")
            tool = decision.get("tool", "wait")
            params = decision.get("params", {})

            self.on_message(f"[{iteration}] {action_desc}")
            logger.info(f"Action: {tool}({params})")

            # 3. ACT - Execute the action
            self.on_state_change("acting")
            result = self.execute_action(tool, params)

            # Small delay for UI to update
            time.sleep(0.3)

            # 4. VERIFY - Check if action succeeded
            self.on_state_change("verifying")
            after_state = self.capture_screen_state()

            # Determine action result
            if result.success:
                action_result = ReactiveActionResult.SUCCESS

                # Optional: Use LLM to verify
                if self.verify_actions:
                    verification = self.verify_action_result(
                        action_desc, before_state, after_state, goal
                    )

                    if verification.get("goal_achieved"):
                        self.on_message("Goal achieved!")
                        self.is_running = False
                        return f"Completed: {goal}"

                    if not verification.get("success", True):
                        action_result = ReactiveActionResult.NEEDS_ALTERNATIVE
                        self.on_message(f"Action may have failed: {verification.get('reason')}")
            else:
                action_result = ReactiveActionResult.FAILED
                self.on_message(f"Action failed: {result.message}")

            # Record attempt
            attempt = ActionAttempt(
                action=action_desc,
                tool=tool,
                params=params,
                result=action_result,
                error=result.message if not result.success else "",
                before_state=before_state,
                after_state=after_state
            )
            self.action_history.append(attempt)

            # Check for too many failures
            recent_failures = sum(
                1 for a in self.action_history[-3:]
                if a.result in [ReactiveActionResult.FAILED, ReactiveActionResult.NEEDS_ALTERNATIVE]
            )
            if recent_failures >= 3:
                self.on_message("Too many failures, stopping...")
                break

        self.is_running = False

        # Summarize
        successful = sum(1 for a in self.action_history if a.result == ReactiveActionResult.SUCCESS)
        total = len(self.action_history)

        if iteration >= self.max_iterations:
            return f"Reached max iterations. {successful}/{total} actions succeeded."

        return f"Completed {successful}/{total} actions for: {goal}"

    def stop(self):
        """Stop the reactive loop."""
        self.is_running = False
        logger.info("Reactive agent stopped")


class ScreenMonitor:
    """
    Continuous screen monitoring for real-time awareness.
    Runs in background and provides screen state updates.
    """

    def __init__(self, interval: float = 0.5, on_change: Callable[[ReactiveScreenState], None] = None):
        self.interval = interval
        self.on_change = on_change or (lambda s: None)
        self.parser = get_omniparser()

        self._running = False
        self._thread = None
        self._last_state: Optional[ReactiveScreenState] = None

    def start(self):
        """Start continuous monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Screen monitor started (interval: {self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Screen monitor stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                state = self._capture_state()

                # Detect significant changes
                if self._has_significant_change(state):
                    self.on_change(state)
                    self._last_state = state

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self.interval)

    def _capture_state(self) -> ReactiveScreenState:
        """Capture current screen state."""
        state = ReactiveScreenState(timestamp=time.time())

        try:
            state.screenshot = pyautogui.screenshot()

            # Lightweight parsing (just OCR, no full parse for speed)
            try:
                win = pyautogui.getActiveWindow()
                if win:
                    state.active_window = win.title
            except:
                pass

        except Exception as e:
            logger.debug(f"State capture error: {e}")

        return state

    def _has_significant_change(self, new_state: ReactiveScreenState) -> bool:
        """Check if screen state changed significantly."""
        if not self._last_state:
            return True

        # Window change is significant
        if new_state.active_window != self._last_state.active_window:
            return True

        # TODO: Could add image comparison for more precision

        return False

    def get_current_state(self) -> Optional[ReactiveScreenState]:
        """Get the most recent screen state."""
        return self._last_state
