"""
VOXCODE Visual Agent
Uses Vision-Language Model (VLM) to truly SEE and UNDERSTAND the screen like a human.
The agent perceives screenshots visually, not just through text extraction.
"""

import base64
import json
import logging
import time
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
from PIL import Image
import requests

logger = logging.getLogger("voxcode.visual_agent")


@dataclass
class VisualObservation:
    """What the agent visually perceives."""
    screenshot: Image.Image
    understanding: str  # VLM's description of what it sees
    screen_type: str  # e.g., "profile_selector", "browser", "file_explorer"
    visible_elements: List[str]
    active_window: str


@dataclass
class VisualAction:
    """Action determined by visual understanding."""
    action_type: str  # click, type, scroll, wait, press_key, hotkey
    target: str  # Description of what to interact with
    coordinates: Optional[Tuple[int, int]] = None  # For clicks
    text: Optional[str] = None  # For typing
    key: Optional[str] = None  # For key presses
    confidence: float = 0.8
    reasoning: str = ""  # Why this action


class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETE = "complete"
    FAILED = "failed"


class VisualAgent:
    """
    Agent that truly SEES the screen using a Vision-Language Model.

    Flow:
    1. Capture screenshot
    2. Send screenshot to VLM (Qwen2.5-VL or similar)
    3. VLM describes what it sees and suggests action
    4. Execute action
    5. Repeat until goal achieved
    """

    # System prompt for visual understanding
    VISION_SYSTEM_PROMPT = """You are a visual AI agent that controls a Windows computer.
You can SEE the screen through screenshots and must understand what you're looking at.

YOUR CAPABILITIES:
- You can see exactly what's on the screen
- You can identify UI elements (buttons, text fields, icons, menus)
- You can understand the context (what application, what screen/page)
- You can determine what action to take to achieve a goal

USER PREFERENCES:
- When Chrome shows a profile selection screen ("Who's using Chrome?"), ALWAYS select "Perfect King" profile
- The user's preferred Chrome profile is "Perfect King"

RESPONSE FORMAT (JSON):
{
    "understanding": "Description of what you see on the screen",
    "screen_type": "profile_selector|browser|file_explorer|desktop|login|search_results|video_player|other",
    "visible_elements": ["list", "of", "important", "elements", "you", "see"],
    "action": {
        "type": "click|type|scroll|wait|press_key|hotkey|done",
        "target": "Description of element to interact with",
        "coordinates": [x, y],  // Approximate center coordinates if clicking
        "text": "text to type",  // If typing
        "key": "enter|escape|tab",  // If pressing key
        "keys": ["ctrl", "l"],  // If hotkey
        "reason": "Why this action helps achieve the goal"
    },
    "goal_achieved": false,
    "confidence": 0.9
}

IMPORTANT:
- Look carefully at the screenshot before deciding
- If you see a profile selector, click the correct profile first
- Estimate coordinates based on where elements appear in the image
- The image is typically 1920x1080 or similar resolution
- Be specific about what you see and why you're taking an action
"""

    def __init__(
        self,
        tools=None,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None,
    ):
        self.tools = tools
        self.ollama_host = ollama_host
        self.model = model
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)

        self.state = AgentState.IDLE
        self.current_goal = ""
        self.history: List[Dict] = []
        self.max_cycles = 15
        self._stop_requested = False

        logger.info(f"VisualAgent initialized with VLM: {model}")

    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        logger.info("Visual agent stop requested")

    def _capture_screenshot(self) -> Image.Image:
        """Capture the current screen."""
        import pyautogui
        screenshot = pyautogui.screenshot()
        return screenshot

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _query_vlm(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Send screenshot to VLM and get understanding."""

        # Convert image to base64
        img_base64 = self._image_to_base64(image)

        # Build the prompt
        full_prompt = f"""{self.VISION_SYSTEM_PROMPT}

CURRENT GOAL: {self.current_goal}

{prompt}

Analyze the screenshot and respond with JSON only. No other text."""

        # Call Ollama with the image
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower for more deterministic
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # Parse the JSON response
            response_text = result.get("response", "{}")
            logger.debug(f"VLM response: {response_text[:500]}...")

            # Extract JSON from response
            return self._parse_vlm_response(response_text)

        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            return {
                "understanding": f"VLM query failed: {e}",
                "screen_type": "unknown",
                "visible_elements": [],
                "action": {"type": "wait", "key": None, "reason": "VLM error"},
                "goal_achieved": False,
                "confidence": 0.0
            }

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from VLM response."""
        try:
            # Try to find JSON in the response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")

        # Fallback
        return {
            "understanding": response[:200] if response else "No response",
            "screen_type": "unknown",
            "visible_elements": [],
            "action": {"type": "wait", "reason": "Could not parse response"},
            "goal_achieved": False,
            "confidence": 0.5
        }

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the action determined by VLM."""
        if not self.tools:
            logger.error("No tools available")
            return False

        action_type = action.get("type", "wait")

        try:
            if action_type == "click":
                coords = action.get("coordinates", [])
                if coords and len(coords) >= 2:
                    x, y = int(coords[0]), int(coords[1])
                    result = self.tools.click(x=x, y=y)
                    return result.success if hasattr(result, 'success') else True
                else:
                    # Try clicking by text if coordinates not available
                    target = action.get("target", "")
                    if target:
                        result = self.tools.click_text(text=target)
                        return result.success if hasattr(result, 'success') else True
                    return False

            elif action_type == "type":
                text = action.get("text", "")
                if text:
                    result = self.tools.type_text(text=text)
                    return result.success if hasattr(result, 'success') else True
                return False

            elif action_type == "press_key":
                key = action.get("key", "enter")
                result = self.tools.press_key(key=key)
                return result.success if hasattr(result, 'success') else True

            elif action_type == "hotkey":
                keys = action.get("keys", [])
                if keys:
                    result = self.tools.hotkey(keys=keys)
                    return result.success if hasattr(result, 'success') else True
                return False

            elif action_type == "scroll":
                amount = action.get("amount", -3)
                result = self.tools.scroll(amount=amount)
                return result.success if hasattr(result, 'success') else True

            elif action_type == "wait":
                seconds = action.get("seconds", 1)
                time.sleep(seconds)
                return True

            elif action_type == "done":
                return True

            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return False

    def process_command(self, command: str) -> str:
        """
        Process a command using visual understanding.

        The agent will:
        1. Capture screenshot
        2. Send to VLM to understand what's on screen
        3. VLM decides action based on visual understanding
        4. Execute action
        5. Repeat until goal achieved or max cycles
        """
        self.current_goal = command
        self.history = []
        self.state = AgentState.PERCEIVING
        self._stop_requested = False

        logger.info(f"Processing command visually: {command}")
        self.on_status(f"Understanding: {command}")

        cycle = 0
        while cycle < self.max_cycles and self.state not in [AgentState.COMPLETE, AgentState.FAILED]:
            if self._stop_requested:
                logger.info("Agent stopped by request")
                return "Stopped by user"

            cycle += 1
            logger.info(f"=== Visual Cycle {cycle}/{self.max_cycles} ===")

            try:
                # 1. PERCEIVE - Capture screenshot
                self.state = AgentState.PERCEIVING
                self.on_step(cycle, "Capturing screen...", "running")
                screenshot = self._capture_screenshot()
                logger.info(f"Screenshot captured: {screenshot.size}")

                # 2. THINK - Send to VLM for visual understanding
                self.state = AgentState.THINKING
                self.on_step(cycle, "Analyzing screen...", "thinking")

                # Build context from history
                history_text = ""
                for h in self.history[-3:]:
                    history_text += f"\n- Action: {h.get('action', 'unknown')} → Result: {h.get('result', 'unknown')}"

                prompt = f"""Look at this screenshot and determine the next action.

Previous actions:{history_text if history_text else " None yet"}

What do you see? What should I do next to achieve the goal?"""

                vlm_response = self._query_vlm(screenshot, prompt)

                understanding = vlm_response.get("understanding", "Unknown")
                action = vlm_response.get("action", {"type": "wait"})
                goal_achieved = vlm_response.get("goal_achieved", False)

                logger.info(f"VLM understands: {understanding[:100]}...")
                logger.info(f"VLM suggests: {action}")

                # Check if goal achieved
                if goal_achieved or action.get("type") == "done":
                    self.state = AgentState.COMPLETE
                    self.on_step(cycle, "Goal achieved!", "done")
                    break

                # 3. ACT - Execute the suggested action
                self.state = AgentState.ACTING
                action_desc = f"{action.get('type', 'unknown')}: {action.get('target', action.get('text', action.get('key', '')))}"
                self.on_step(cycle, action_desc, "running")

                success = self._execute_action(action)

                # Record in history
                self.history.append({
                    "cycle": cycle,
                    "understanding": understanding,
                    "action": action_desc,
                    "result": "success" if success else "failed"
                })

                if success:
                    self.on_step(cycle, action_desc, "done")
                else:
                    self.on_step(cycle, f"{action_desc} - failed", "failed")

                # Small delay for UI to update
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
                self.on_step(cycle, f"Error: {str(e)}", "failed")
                time.sleep(1)

        # Final result
        if self.state == AgentState.COMPLETE:
            return f"Successfully completed: {command}"
        else:
            self.state = AgentState.FAILED
            return f"Could not complete: {command} (completed {cycle} cycles)"


def create_visual_agent(tools=None, **kwargs) -> VisualAgent:
    """Create a visual agent instance."""
    return VisualAgent(tools=tools, **kwargs)
