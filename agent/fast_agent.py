"""
VOXCODE Fast Visual Agent
Uses Screenshot + OmniParser for perception, Groq for fast reasoning.
Hybrid approach: Visual parsing locally, reasoning in cloud (fast).
"""

import logging
import time
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
from PIL import Image
import json

from brain.llm import get_llm_client
from config import config

logger = logging.getLogger("voxcode.fast_agent")


class FastAgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class FastScreenState:
    """Current state of the screen."""
    active_window: str
    window_title: str
    visible_text: List[str]
    clickable_elements: List[Dict]
    screen_description: str


class FastVisualAgent:
    """
    Fast agent that:
    1. Takes screenshot
    2. Uses OmniParser to extract elements (local, fast)
    3. Sends description to Groq for reasoning (cloud, fast)
    4. Executes action
    5. Repeats

    This is much faster than using a local VLM.
    """

    SYSTEM_PROMPT = """You are an AI agent controlling a Windows computer. You receive descriptions of what's on screen and decide what action to take.

CRITICAL: Pay attention to the WINDOW TITLE to understand where you are:
- "Google Chrome" with "Who's using Chrome" = Profile selector → Click profile name
- "YouTube - Google Chrome" = YouTube page → You're in browser, can interact
- "New Tab - Google Chrome" = Empty browser tab → Type in address bar
- "This PC" or "C:\\" = File Explorer → Navigate folders
- "Notepad" = Text editor → Can type content

GOAL EXECUTION RULES:
1. First, understand WHERE you are from the window title
2. Then decide what action moves you toward the goal
3. If you just clicked something, WAIT for the screen to change
4. Don't repeat the same action if it didn't work

USER PREFERENCES:
- Chrome profile: Select "Perfect King" when profile selector appears
- After selecting profile, Chrome opens → proceed with the task

HOW TO DO COMMON TASKS:
- Open Chrome: open_app "chrome" → if profile selector appears, click profile
- Go to website: When in browser, press Ctrl+L → type URL → press Enter
- Open File Explorer: open_app "explorer" or press Win+E
- Navigate folders: click on folder name to open it
- Create file: right-click → New → Text Document (or use Notepad)

ACTION FORMAT - Respond with JSON only:
{
    "current_screen": "What type of screen am I on (browser/explorer/desktop/etc)",
    "understanding": "What I see and where I am",
    "next_step": "What I need to do to progress toward the goal",
    "action": {
        "type": "click_text|click|type_text|press_key|hotkey|open_app|wait|done",
        "target": "element to interact with",
        "text": "text to type (if typing)",
        "key": "key to press (if pressing)",
        "keys": ["ctrl", "l"] (if hotkey)
    },
    "goal_achieved": false
}

AVAILABLE ACTIONS:
- click_text: Click visible text {"type": "click_text", "target": "Search"}
- type_text: Type text {"type": "type_text", "text": "youtube.com"}
- press_key: Press key {"type": "press_key", "key": "enter"}
- hotkey: Key combo {"type": "hotkey", "keys": ["ctrl", "l"]}
- open_app: Open app {"type": "open_app", "target": "chrome"}
- wait: Wait {"type": "wait", "seconds": 2}
- done: Complete {"type": "done"}
"""

    def __init__(
        self,
        vision=None,
        tools=None,
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None,
    ):
        self.vision = vision
        self.tools = tools
        self.llm = get_llm_client()
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)

        self.state = FastAgentState.IDLE
        self.current_goal = ""
        self.history: List[Dict] = []
        self.max_cycles = 12
        self._stop_requested = False

        logger.info("FastVisualAgent initialized")

    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        logger.info("Agent stop requested")

    def _get_screen_state(self) -> FastScreenState:
        """Capture and analyze current screen state."""
        import pyautogui
        import pygetwindow as gw

        # Get active window
        active_window = "Unknown"
        window_title = ""
        try:
            win = gw.getActiveWindow()
            if win:
                window_title = win.title or ""
                active_window = window_title.split(" - ")[-1] if " - " in window_title else window_title
        except:
            pass

        # Detect screen type and elements using vision/OmniParser
        visible_text = []
        clickable_elements = []
        screen_description = f"Active window: {window_title}"

        if self.vision:
            try:
                # Use OmniParser to find all text on screen
                elements = self.vision.find_all_text()
                visible_text = [e.text for e in elements if hasattr(e, 'text') and e.text]

                # Smart screen type detection based on window title AND content
                window_lower = window_title.lower()
                text_joined = ' '.join(visible_text).lower()

                # Priority 1: Check window title first (most reliable)
                if "who's using chrome" in text_joined or ("chrome" in window_lower and "profile" in text_joined):
                    # This is ONLY the profile selector if we see "Who's using Chrome"
                    screen_description = f"Chrome Profile Selector. Profiles: {', '.join(visible_text[:15])}"

                elif "youtube" in window_lower:
                    screen_description = f"YouTube - {window_title}. Elements: {', '.join(visible_text[:20])}"

                elif "google" in window_lower and "chrome" in window_lower:
                    # Regular Chrome browser (not profile selector)
                    screen_description = f"Chrome Browser: {window_title}. Page content: {', '.join(visible_text[:20])}"

                elif "chrome" in window_lower or "edge" in window_lower or "firefox" in window_lower:
                    # Browser with a webpage
                    screen_description = f"Browser: {window_title}. Content: {', '.join(visible_text[:20])}"

                elif "explorer" in window_lower or "this pc" in window_lower or ":" in window_title:
                    # File Explorer
                    screen_description = f"File Explorer: {window_title}. Items: {', '.join(visible_text[:20])}"

                elif "notepad" in window_lower:
                    screen_description = f"Notepad: {window_title}. Content: {', '.join(visible_text[:10])}"

                elif window_title == "" or "program manager" in window_lower:
                    # Desktop
                    screen_description = f"Desktop/Start Menu. Visible: {', '.join(visible_text[:15])}"

                else:
                    # Generic window
                    screen_description = f"Application: {window_title}. Content: {', '.join(visible_text[:15])}"

            except Exception as e:
                logger.warning(f"Vision error: {e}")
                screen_description = f"Active window: {window_title} (vision limited)"

        return FastScreenState(
            active_window=active_window,
            window_title=window_title,
            visible_text=visible_text,
            clickable_elements=clickable_elements,
            screen_description=screen_description
        )

    def _think(self, screen_state: FastScreenState) -> Dict[str, Any]:
        """Use Groq to decide next action based on screen state."""

        # Build history context
        history_text = ""
        for h in self.history[-4:]:
            history_text += f"\n- {h.get('action', 'unknown')} → {h.get('result', 'unknown')}"

        prompt = f"""{self.SYSTEM_PROMPT}

CURRENT GOAL: {self.current_goal}

CURRENT SCREEN STATE:
{screen_state.screen_description}

Window Title: {screen_state.window_title}
Visible Text Elements: {', '.join(screen_state.visible_text[:25]) if screen_state.visible_text else 'None detected'}

PREVIOUS ACTIONS:{history_text if history_text else " None yet"}

Based on the screen state and goal, what single action should I take next?
Respond with JSON only."""

        try:
            response = self.llm.generate(prompt)
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Thinking error: {e}")
            return {
                "understanding": f"Error: {e}",
                "action": {"type": "wait", "seconds": 1},
                "goal_achieved": False
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass

        return {
            "understanding": response[:100] if response else "No response",
            "action": {"type": "wait", "seconds": 1},
            "goal_achieved": False
        }

    def _execute_action(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute the decided action."""
        if not self.tools:
            return False, "No tools available"

        action_type = action.get("type", "wait")

        try:
            if action_type == "click_text":
                target = action.get("target", "")
                if target:
                    result = self.tools.click_text(text=target)
                    return result.success, f"Clicked: {target}"
                return False, "No target specified"

            elif action_type == "click":
                x = action.get("x", 0)
                y = action.get("y", 0)
                result = self.tools.click(x=x, y=y)
                return result.success, f"Clicked at ({x}, {y})"

            elif action_type == "type_text":
                text = action.get("text", "")
                if text:
                    result = self.tools.type_text(text=text)
                    return result.success, f"Typed: {text}"
                return False, "No text specified"

            elif action_type == "press_key":
                key = action.get("key", "enter")
                result = self.tools.press_key(key=key)
                return result.success, f"Pressed: {key}"

            elif action_type == "hotkey":
                keys = action.get("keys", [])
                if keys:
                    result = self.tools.hotkey(keys=keys)
                    return result.success, f"Hotkey: {'+'.join(keys)}"
                return False, "No keys specified"

            elif action_type == "open_app":
                target = action.get("target", "")
                if target:
                    result = self.tools.open_application(path_or_name=target)
                    return result.success, f"Opened: {target}"
                return False, "No app specified"

            elif action_type == "wait":
                seconds = action.get("seconds", 1)
                time.sleep(seconds)
                return True, f"Waited {seconds}s"

            elif action_type == "done":
                return True, "Goal achieved"

            else:
                logger.warning(f"Unknown action: {action_type}")
                return False, f"Unknown action: {action_type}"

        except Exception as e:
            return False, str(e)

    def process_command(self, command: str) -> str:
        """Process a command with fast visual feedback."""
        self.current_goal = command
        self.history = []
        self.state = FastAgentState.PERCEIVING
        self._stop_requested = False

        logger.info(f"Processing: {command}")
        self.on_status(f"Goal: {command}")

        cycle = 0
        while cycle < self.max_cycles and self.state not in [FastAgentState.COMPLETE, FastAgentState.FAILED]:
            if self._stop_requested:
                logger.info("Stopped by request")
                return "Stopped by user"

            cycle += 1
            logger.info(f"=== Cycle {cycle}/{self.max_cycles} ===")

            try:
                # 1. PERCEIVE - Get screen state
                self.state = FastAgentState.PERCEIVING
                self.on_step(cycle, "Analyzing screen...", "running")
                screen_state = self._get_screen_state()
                logger.info(f"Screen: {screen_state.screen_description[:80]}...")

                # 2. THINK - Decide action (Groq - fast!)
                self.state = FastAgentState.THINKING
                decision = self._think(screen_state)

                understanding = decision.get("understanding", "")
                action = decision.get("action", {"type": "wait"})
                goal_achieved = decision.get("goal_achieved", False)

                logger.info(f"Understanding: {understanding[:60]}...")
                logger.info(f"Action: {action}")

                # Check if done
                if goal_achieved or action.get("type") == "done":
                    self.state = FastAgentState.COMPLETE
                    self.on_step(cycle, "Goal achieved!", "done")
                    break

                # 3. ACT - Execute
                self.state = FastAgentState.ACTING
                action_desc = f"{action.get('type')}: {action.get('target', action.get('text', action.get('key', '')))}"
                self.on_step(cycle, action_desc, "running")

                success, message = self._execute_action(action)

                # Record history
                self.history.append({
                    "cycle": cycle,
                    "screen": screen_state.screen_description[:50],
                    "action": action_desc,
                    "result": "success" if success else f"failed: {message}"
                })

                if success:
                    self.on_step(cycle, action_desc, "done")
                else:
                    self.on_step(cycle, f"{action_desc} - {message}", "failed")

                # Brief pause for UI to update
                time.sleep(0.3)

            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self.on_step(cycle, f"Error: {e}", "failed")
                time.sleep(0.5)

        if self.state == FastAgentState.COMPLETE:
            return f"Successfully completed: {command}"
        else:
            return f"Could not complete: {command}"


def create_fast_agent(vision=None, tools=None, **kwargs) -> FastVisualAgent:
    """Create a fast visual agent."""
    return FastVisualAgent(vision=vision, tools=tools, **kwargs)
