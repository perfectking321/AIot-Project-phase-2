"""
VOXCODE Reactive Agent Loop
Real-time perceive-think-act-verify cycle with self-correction.
"""

import os
import time
import logging
import threading
import json
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("voxcode.reactive")

try:
    from PIL import Image
    import pyautogui
    import pygetwindow as gw
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    gw = None

from config import config
from brain.llm import get_model_for_role
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
    # Store only text descriptions, NOT full ReactiveScreenState with PIL images
    before_desc: str = ""
    after_desc: str = ""


class ReactiveAgent:
    """
    Reactive agent with real-time perception and self-correction.

    Uses a perceive-think-act-verify loop:
    1. PERCEIVE: Capture screen state
    2. THINK: Decide next action based on goal and current state
    3. ACT: Execute the action
    4. VERIFY: Check if action succeeded, decide next step
    """

    DECIDE_ACTION_PROMPT = """You are a Windows automation agent. Your ONLY job: look at the current screen state and return the single best next action as JSON.

GOAL: {goal}
ACTIVE WINDOW: {active_window}
VISIBLE APPS: {visible_apps}
SCREEN ELEMENTS: {visible_elements}
PREVIOUS ACTIONS:
{action_history}

AVAILABLE TOOLS — pick whichever one gets you closest to the goal:

UI INTERACTION:
- open_application  → {{"tool":"open_application","params":{{"path_or_name":"chrome"}}}}
- click_text        → {{"tool":"click_text","params":{{"text":"Search"}}}}
- type_text         → {{"tool":"type_text","params":{{"text":"hello world"}}}}
- press_key         → {{"tool":"press_key","params":{{"key":"enter"}}}}
- hotkey            → {{"tool":"hotkey","params":{{"keys":["ctrl","l"]}}}}
- scroll            → {{"tool":"scroll","params":{{"amount":3}}}}
- wait              → {{"tool":"wait","params":{{"seconds":1}}}}
- focus_window      → {{"tool":"focus_window","params":{{"title":"Chrome"}}}}
- parse_screen      → {{"tool":"parse_screen","params":{{}}}}

BROWSER DOM (requires Chrome on port 9222):
- dom_search_extract → {{"tool":"dom_search_extract","params":{{"query":"search terms"}}}}
  USE THIS to search Google and get real results as structured data
- dom_read_page      → {{"tool":"dom_read_page","params":{{}}}}
  USE THIS to read the actual text/URL of the current page
- dom_click          → {{"tool":"dom_click","params":{{"text":"link or button text"}}}}
- dom_fill           → {{"tool":"dom_fill","params":{{"by_label":"Search","value":"cats"}}}}
- dom_wait           → {{"tool":"dom_wait","params":{{"wait_for":"networkidle"}}}}

SYSTEM / WINDOWS:
- system_command    → {{"tool":"system_command","params":{{"command":"Get-Process | Select -First 5"}}}}
  USE THIS for anything PowerShell can do: files, folders, settings, processes
- brightness_control → {{"tool":"brightness_control","params":{{"level":"min"}}}}
- bluetooth_control  → {{"tool":"bluetooth_control","params":{{"action":"on"}}}}
- network_info       → {{"tool":"network_info","params":{{"action":"ip"}}}}
- process_manager    → {{"tool":"process_manager","params":{{"action":"list"}}}}
- system_info        → {{"tool":"system_info","params":{{}}}}
- window_manager     → {{"tool":"window_manager","params":{{"action":"list"}}}}

RULES:
1. ONE action per response. Smallest useful step.
2. If app already in VISIBLE APPS, do NOT open it again.
3. Use dom_search_extract to SEARCH the web — never guess URLs.
4. Use system_command to create/modify files: New-Item, Set-Content, etc.
5. Use brightness_control for brightness — NOT system_command.
6. Never repeat a failed action. Adapt.
7. When the goal is fully achieved, signal completion.

COMPLETION: Signal goal done with:
{{"action":"GOAL_COMPLETE","tool":"none","params":{{}},"reason":"brief explanation of what was done"}}

Return ONLY valid JSON. No other text."""

    VERIFY_ACTION_PROMPT = """Did the action succeed?
ACTION: {action}
BEFORE: window={before_window}, elements={before_elements}
AFTER: window={after_window}, elements={after_elements}
GOAL: {goal}
JSON: {{"success":true/false,"goal_achieved":true/false,"reason":"brief"}}"""

    def __init__(
        self,
        on_message: Callable[[str], None] = None,
        on_state_change: Callable[[str], None] = None,
        max_iterations: int = 20,
        verify_actions: bool = False
    ):
        self.llm = get_model_for_role("executor")
        self.tools = WindowsTools(use_omniparser=True)
        self.parser = get_omniparser()

        self.on_message = on_message or (lambda m: None)
        self.on_state_change = on_state_change or (lambda s: None)

        self.max_iterations = max_iterations
        self.verify_actions = verify_actions

        # State tracking
        self.action_history: List[ActionAttempt] = []
        self.current_goal: str = ""
        self._cancel_event = threading.Event()
        self.is_running = False
        self._last_screen_state: Optional[ReactiveScreenState] = None
        self._last_screen_time: float = 0.0
        self._screen_cache_ttl: float = 0.4

        logger.info("ReactiveAgent initialized")

    def capture_screen_state(self, force: bool = False) -> ReactiveScreenState:
        """Capture the current screen state with app detection."""
        now = time.time()
        if (
            not force
            and self._last_screen_state is not None
            and (now - self._last_screen_time) < self._screen_cache_ttl
        ):
            return self._last_screen_state

        state = ReactiveScreenState(timestamp=time.time())

        try:
            # Get screenshot
            screenshot = pyautogui.screenshot()

            # Parse with OmniParser
            state.parsed = self.parser.parse_screen(screenshot)

            # Get active window (focused)
            try:
                win = gw.getActiveWindow()
                if win and hasattr(win, 'title'):
                    state.active_window = win.title or ""
            except Exception:
                pass

            # Get visible element labels
            if state.parsed:
                state.visible_elements = [
                    e.label for e in state.parsed.elements
                    if e.is_interactable and e.label != "icon"
                ][:20]

                # Detect visible apps from screen content
                state.visible_apps = self._detect_visible_apps(state.parsed, state.visible_elements)

            # Release the screenshot from state after parsing (don't keep it in memory)
            state.screenshot = None  # Parsed data is enough; screenshot is expensive

        except Exception as e:
            logger.error(f"Failed to capture screen state: {e}")

        self._last_screen_state = state
        self._last_screen_time = now
        return state

    def _detect_visible_apps(self, parsed: ParsedScreen, elements: List[str]) -> List[str]:
        """Detect which apps are visible on screen based on window title and UI elements."""
        visible_apps = []

        # Primary signal: active window title (most reliable)
        active_title = ""
        try:
            import pygetwindow as gw
            w = gw.getActiveWindow()
            if w:
                active_title = w.title.lower()
        except Exception:
            pass

        # Detect from window title
        title_app_map = {
            "chrome": "Chrome",
            "firefox": "Firefox",
            "edge": "Edge",
            "youtube": "YouTube",
            "notepad": "Notepad",
            "visual studio code": "VS Code",
            "code -": "VS Code",
            "file explorer": "File Explorer",
            "discord": "Discord",
            "spotify": "Spotify",
            "whatsapp": "WhatsApp",
            "telegram": "Telegram",
        }

        for keyword, app_name in title_app_map.items():
            if keyword in active_title and app_name not in visible_apps:
                visible_apps.append(app_name)

        # Secondary signal: filter elements — exclude file paths and only match UI-specific text
        # Filter out file paths (C:\..., /, etc.) and only keep short UI elements
        ui_elements = []
        for elem in elements:
            # Skip file paths and long strings
            if any(c in elem for c in ["\\", ":/", "C:", "Program Files"]):
                continue
            if len(elem) > 50:
                continue
            ui_elements.append(elem.lower())
        ui_text = " ".join(ui_elements)

        # Only detect apps from clear UI indicators (not file path mentions)
        ui_app_indicators = {
            "Chrome": ["new tab - google chrome", "google chrome"],
            "YouTube": ["youtube.com", "search youtube", "watch later", "subscribe"],
            "WhatsApp": ["type a message", "whatsapp web"],
            "Notepad": ["untitled - notepad"],
            "Spotify": ["spotify -", "spotify premium"],
            "Discord": ["discord -"],
            "Settings": ["settings"],
        }

        for app, indicators in ui_app_indicators.items():
            for indicator in indicators:
                if indicator in ui_text and app not in visible_apps:
                    visible_apps.append(app)
                    break

        return visible_apps

    def _format_action_history(self) -> str:
        """Format recent action history compactly for the decision prompt."""
        recent = self.action_history[-5:]
        if not recent:
            return "(No actions yet)"

        lines = []
        for i, attempt in enumerate(recent, 1):
            status = "OK" if attempt.result == ReactiveActionResult.SUCCESS else "FAIL"
            # Include tool and key params so LLM sees exactly what was done
            params_short = str(attempt.params)[:40] if attempt.params else ""
            lines.append(f"{i}. [{status}] {attempt.tool}({params_short}) - {attempt.action[:50]}")

        # Detect consecutive duplicate actions (same tool + same params)
        consecutive_dupes = 0
        if len(recent) >= 2:
            last_sig = (recent[-1].tool, str(recent[-1].params))
            for attempt in reversed(recent[:-1]):
                if (attempt.tool, str(attempt.params)) == last_sig:
                    consecutive_dupes += 1
                else:
                    break

        consecutive_fails = 0
        for attempt in reversed(recent):
            if attempt.result in [ReactiveActionResult.FAILED, ReactiveActionResult.NEEDS_ALTERNATIVE]:
                consecutive_fails += 1
            else:
                break

        history = "\n".join(lines)
        if consecutive_dupes >= 1:
            history += f"\n⚠ STUCK: Same action repeated {consecutive_dupes + 1}x. You MUST try a DIFFERENT action now!"
        if consecutive_fails >= 2:
            history += f"\n⚠ {consecutive_fails} consecutive failures — try a completely different approach"
        return history

    def decide_next_action(self, goal: str, state: ReactiveScreenState) -> Dict[str, Any]:
        """Use LLM to decide the next action based on current screen state."""
        history_str = self._format_action_history()

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
            response = self.llm.generate_short(prompt)
            content = response.content.strip()

            # Strip markdown code fences if present (```json ... ```)
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            # Parse JSON response
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

                # --- FIX 2: Normalize GOAL_COMPLETE signal from LLM ---
                tool = result.get("tool", "wait")
                action_val = result.get("action", "")
                if (
                    tool in ("none", "goal_complete", "done", "GOAL_COMPLETE")
                    or (isinstance(action_val, str) and action_val.upper() in (
                        "GOAL_COMPLETE", "DONE", "COMPLETE", "FINISHED"))
                    or result.get("done") is True
                    or result.get("goal_achieved") is True
                ):
                    result["action"] = "GOAL_COMPLETE"
                    result["tool"] = "none"
                    result["reason"] = result.get("reason", result.get("action", "Goal achieved"))

                # --- FIX 1: No whitelist — warn but pass through ---
                known_tools = list(self.PARAM_ALIASES.keys()) + [
                    "click", "click_element_by_id", "parse_screen", "focus_window",
                    "take_screenshot", "find_text", "scroll", "none",
                    "dom_search_extract", "dom_read_page", "dom_click", "dom_fill",
                    "dom_extract", "dom_wait", "system_command", "brightness_control",
                    "bluetooth_control", "network_info", "process_manager", "system_info",
                    "window_manager",
                ]
                if result.get("tool") not in known_tools:
                    logger.warning(f"Unrecognized tool '{result.get('tool')}' — passing through, LLM may know something we don't")

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
        """
        Verify action success using fast heuristics first, LLM as last resort.
        This approach is 10-20x faster than always calling LLM.
        """
        # Heuristic 1: Window changed = action had effect
        if before.active_window != after.active_window:
            return {"success": True, "goal_achieved": False, "reason": "Window changed"}

        # Heuristic 2: Visible elements changed significantly = something happened
        before_set = set(before.visible_elements)
        after_set = set(after.visible_elements)
        diff_count = len(before_set.symmetric_difference(after_set))
        if diff_count > 2:
            return {"success": True, "goal_achieved": False, "reason": "UI updated"}

        # Heuristic 3: Typing/keyboard actions always "succeed" locally
        action_lower = action.lower()
        if any(word in action_lower for word in ["type", "typed", "wrote", "pressed enter",
                                                   "hotkey", "press", "key"]):
            return {"success": True, "goal_achieved": False, "reason": "Input action assumed ok"}

        # Heuristic 4: Goal keywords found in after state
        goal_words = set(goal.lower().split())
        after_words = set(" ".join(after.visible_elements).lower().split())
        overlap = goal_words & after_words
        if len(overlap) >= 2:
            return {"success": True, "goal_achieved": True, "reason": f"Goal keywords visible"}

        # Only call LLM if heuristics are inconclusive AND verify_actions is forced
        if not self.verify_actions:
            return {"success": True, "goal_achieved": False}

        # LLM fallback (only for ambiguous cases)
        prompt = self.VERIFY_ACTION_PROMPT.format(
            action=action,
            before_window=before.active_window,
            before_elements=", ".join(before.visible_elements[:8]),
            after_window=after.active_window,
            after_elements=", ".join(after.visible_elements[:8]),
            goal=goal
        )
        try:
            response = self.llm.generate(prompt)
            content = response.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")

        return {"success": True, "goal_achieved": False}

    # Map of LLM param aliases -> correct tool parameter names
    PARAM_ALIASES = {
        "open_application": {
            "application": "path_or_name",
            "app_name": "path_or_name",
            "app": "path_or_name",
            "name": "path_or_name",
            "target": "path_or_name",
        },
        "click_text": {
            "target": "text",
            "element": "text",
            "label": "text",
        },
        "type_text": {
            "content": "text",
            "input": "text",
            "value": "text",
        },
        "press_key": {
            "button": "key",
            "name": "key",
        },
        "hotkey": {},
        "scroll": {
            "direction": "amount",
            "clicks": "amount",
        },
        "wait": {
            "duration": "seconds",
            "time": "seconds",
        },
        # DOM browser skills
        "dom_search_extract": {"search": "query", "q": "query"},
        "dom_read_page": {},
        "dom_click": {"element": "text", "label": "text", "target": "text"},
        "dom_fill": {},
        "dom_extract": {"code": "js_code", "javascript": "js_code"},
        "dom_wait": {},
        # System skills
        "brightness_control": {"value": "level", "brightness": "level"},
        "bluetooth_control": {},
        "network_info": {},
        "process_manager": {},
        "system_command": {"cmd": "command", "powershell": "command", "run": "command"},
        "system_info": {},
        "window_manager": {},
    }

    def _normalize_params(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM parameter names to match actual tool signatures."""
        aliases = self.PARAM_ALIASES.get(tool, {})
        if not aliases:
            return params

        normalized = {}
        for key, value in params.items():
            canonical = aliases.get(key, key)
            normalized[canonical] = value
        return normalized

    def execute_action(self, tool: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a single action — routes to WindowsTools, DOM skills, or system skills."""
        params = self._normalize_params(tool, params)

        # 1. Try WindowsTools directly (covers most UI tools)
        tool_method = getattr(self.tools, tool, None)
        if tool_method and callable(tool_method):
            try:
                return tool_method(**params)
            except TypeError as e:
                logger.warning(f"Param mismatch for {tool}: {e}, trying fallback")
                return self._execute_with_fallback(tool, params)
            except Exception as e:
                logger.error(f"Tool {tool} raised: {e}")
                return ToolResult(ToolStatus.FAILURE, str(e))

        # 2. DOM browser skills
        if tool.startswith("dom_"):
            return self._execute_dom_tool(tool, params)

        # 3. System skills
        if tool in ("brightness_control", "bluetooth_control", "network_info",
                    "process_manager", "system_info", "window_manager", "system_command"):
            return self._execute_system_tool(tool, params)

        # 4. No-op / done signal
        if tool in ("none", "done", "wait"):
            seconds = params.get("seconds", 0)
            if seconds and float(seconds) > 0:
                time.sleep(float(seconds))
            return ToolResult(ToolStatus.SUCCESS, "No-op / goal complete signal")

        return ToolResult(ToolStatus.FAILURE, f"Unknown tool: {tool}")

    def _execute_with_fallback(self, tool: str, params: Dict[str, Any]) -> ToolResult:
        """Positional fallback execution for common UI tools when param names mismatch."""
        try:
            if tool == "open_application":
                name = params.get("path_or_name") or next(iter(params.values()), "")
                return self.tools.open_application(path_or_name=str(name))
            elif tool == "click_text":
                text = params.get("text") or next(iter(params.values()), "")
                return self.tools.click_text(text=str(text))
            elif tool == "type_text":
                text = params.get("text") or next(iter(params.values()), "")
                return self.tools.type_text(text=str(text))
            elif tool == "press_key":
                key = params.get("key") or next(iter(params.values()), "enter")
                return self.tools.press_key(key=str(key))
            elif tool == "hotkey":
                keys = params.get("keys", list(params.values()))
                if isinstance(keys, list):
                    return self.tools.hotkey(*keys)
                return self.tools.hotkey(str(keys))
            elif tool == "wait":
                secs = params.get("seconds", 1)
                return self.tools.wait(seconds=float(secs))
            elif tool == "scroll":
                amt = params.get("amount", 3)
                return self.tools.scroll(amount=int(amt))
        except Exception as fallback_e:
            logger.error(f"Fallback execution also failed: {fallback_e}")
            return ToolResult(ToolStatus.FAILURE, str(fallback_e))
        return ToolResult(ToolStatus.FAILURE, f"No fallback for tool: {tool}")

    def _execute_dom_tool(self, tool: str, params: Dict[str, Any]) -> ToolResult:
        """Route to DOM browser skill classes. Ensures Chrome is available first."""
        try:
            from agent.skills.dom_browser_skills import (
                DOMSearchExtractSkill, DOMReadPageSkill, DOMClickSkill,
                DOMFillSkill, DOMExtractSkill, DOMWaitSkill, PLAYWRIGHT_AVAILABLE
            )
            if not PLAYWRIGHT_AVAILABLE:
                return ToolResult(ToolStatus.FAILURE, "Playwright not installed. Run: pip install playwright && playwright install chromium")

            # Check if Chrome CDP is reachable; if not, open Chrome with debug port
            try:
                import requests as _req
                _req.get("http://localhost:9222/json/version", timeout=2)
            except Exception:
                logger.info("Chrome CDP not available — launching Chrome with debug port")
                self.on_message("Opening Chrome for browser automation...")
                import subprocess
                bat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "start_chrome_for_voxcode.bat")
                if os.path.exists(bat_path):
                    subprocess.Popen(bat_path, shell=True)
                else:
                    subprocess.Popen('start chrome --remote-debugging-port=9222', shell=True)
                time.sleep(3)

            skill_map = {
                "dom_search_extract": DOMSearchExtractSkill,
                "dom_read_page": DOMReadPageSkill,
                "dom_click": DOMClickSkill,
                "dom_fill": DOMFillSkill,
                "dom_extract": DOMExtractSkill,
                "dom_wait": DOMWaitSkill,
            }
            cls = skill_map.get(tool)
            if not cls:
                return ToolResult(ToolStatus.FAILURE, f"Unknown DOM tool: {tool}")
            skill = cls()
            result = skill.execute(**params)
            # Convert SkillResult -> ToolResult
            status = ToolStatus.SUCCESS if result.success else ToolStatus.FAILURE
            return ToolResult(status, result.message, result.data)
        except Exception as e:
            logger.error(f"DOM tool {tool} failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def _execute_system_tool(self, tool: str, params: Dict[str, Any]) -> ToolResult:
        """Route to system skill classes."""
        try:
            from agent.skills.system_skills import (
                BrightnessControlSkill, BluetoothControlSkill,
                NetworkInfoSkill, ProcessManagerSkill,
                SystemInfoSkill, WindowManagerSkill, SystemCommandSkill
            )
            skill_map = {
                "brightness_control": BrightnessControlSkill,
                "bluetooth_control": BluetoothControlSkill,
                "network_info": NetworkInfoSkill,
                "process_manager": ProcessManagerSkill,
                "system_info": SystemInfoSkill,
                "window_manager": WindowManagerSkill,
                "system_command": SystemCommandSkill,
            }
            cls = skill_map.get(tool)
            if not cls:
                return ToolResult(ToolStatus.FAILURE, f"Unknown system tool: {tool}")
            skill = cls()
            result = skill.execute(**params)
            status = ToolStatus.SUCCESS if result.success else ToolStatus.FAILURE
            return ToolResult(status, result.message, result.data)
        except Exception as e:
            logger.error(f"System tool {tool} failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def process_goal(self, goal: str) -> str:
        """
        Process a goal using reactive perceive-think-act-verify loop.

        Returns a summary of what happened.
        """
        self.current_goal = goal
        self.action_history = []
        self._cancel_event.clear()
        self.is_running = True
        self._last_screen_state = None
        self._last_screen_time = 0.0

        self.on_message(f"Starting: {goal}")
        self.on_state_change("perceiving")

        iteration = 0
        while not self._cancel_event.is_set() and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # 1. PERCEIVE - Capture current screen state
            self.on_state_change("perceiving")
            before_state = self.capture_screen_state(force=True)

            self.on_message(f"[{iteration}] Analyzing screen...")
            logger.info(f"Screen: {before_state.describe()}")

            # Log detected apps for debugging
            if before_state.visible_apps:
                self.on_message(f"    Detected apps: {', '.join(before_state.visible_apps)}")

            # 2. THINK - Decide next action
            self.on_state_change("thinking")
            decision = self.decide_next_action(goal, before_state)

            # Check for goal completion
            if decision.get("action") == "GOAL_COMPLETE" or decision.get("tool") in ("none", "done"):
                reason = decision.get("reason", "Task complete")
                self.on_message(f"✓ Goal achieved: {reason}")
                self.is_running = False
                return f"Completed: {reason}"

            action_desc = decision.get("action", "unknown")
            tool = decision.get("tool", "wait")
            params = decision.get("params", {})

            # Detect if LLM is stuck repeating the same action
            action_sig = (tool, str(params))
            consecutive_repeats = 0
            for prev in reversed(self.action_history):
                if (prev.tool, str(prev.params)) == action_sig:
                    consecutive_repeats += 1
                else:
                    break

            if consecutive_repeats >= 2:
                logger.warning(f"Action loop detected: {tool}({params}) repeated {consecutive_repeats + 1}x")
                self.on_message(f"[{iteration}] Stuck in loop, skipping repeated action")
                # Record as needs_alternative and continue to next iteration
                attempt = ActionAttempt(
                    action=action_desc, tool=tool, params=params,
                    result=ReactiveActionResult.NEEDS_ALTERNATIVE,
                    error="Action loop detected - same action repeated too many times",
                    before_desc=before_state.describe(), after_desc=before_state.describe()
                )
                self.action_history.append(attempt)
                continue

            self.on_message(f"[{iteration}] {action_desc}")
            logger.info(f"Action: {tool}({params})")

            # 3. ACT - Execute the action
            self.on_state_change("acting")
            result = self.execute_action(tool, params)

            # Small delay for UI to update
            time.sleep(0.3)

            # 4. VERIFY - Check if action succeeded
            self.on_state_change("verifying")
            after_state = self.capture_screen_state(force=True)

            # Determine action result
            if result.success:
                action_result = ReactiveActionResult.SUCCESS
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
                before_desc=before_state.describe(),
                after_desc=after_state.describe()
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

        if self._cancel_event.is_set():
            return f"Cancelled: {goal}"

        # Summarize
        successful = sum(1 for a in self.action_history if a.result == ReactiveActionResult.SUCCESS)
        total = len(self.action_history)

        if iteration >= self.max_iterations:
            return f"Reached max iterations. {successful}/{total} actions succeeded."

        return f"Completed {successful}/{total} actions for: {goal}"

    def stop(self):
        """Stop the reactive loop."""
        self.cancel()

    def cancel(self):
        """Cancel current execution immediately."""
        self._cancel_event.set()
        self.is_running = False
        logger.info("ReactiveAgent execution cancelled")


class ScreenMonitor:
    """
    Continuous screen monitoring for real-time awareness.
    Runs in background and provides screen state updates.

    TODO: ScreenMonitor can be used for background awareness between commands.
    To activate: monitor = ScreenMonitor(interval=0.5, on_change=self._on_screen_change)
    monitor.start() after agent starts, monitor.stop() before agent stops.
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
                win = gw.getActiveWindow()
                if win and hasattr(win, 'title'):
                    state.active_window = win.title or ""
            except Exception:
                pass

            state.screenshot = None

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
