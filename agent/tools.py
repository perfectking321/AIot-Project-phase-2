"""
VOXCODE Windows Automation Tools
UI automation with vision capabilities.
"""

import os
import time
import subprocess
import logging
import json
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("voxcode.tools")

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from config import config

# Import vision module
try:
    from agent.vision import ScreenVision, vision, OCR_AVAILABLE
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    OCR_AVAILABLE = False

# Import OmniParser for advanced screen parsing
try:
    from agent.omniparser import OmniParser, get_omniparser, ParsedScreen, UIElement
    OMNIPARSER_AVAILABLE = True
except ImportError:
    OMNIPARSER_AVAILABLE = False

# Import LLM for semantic matching
try:
    from brain.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import API registry for richer app launch resolution
try:
    from brain.api_registry import APIRegistry
    API_REGISTRY_AVAILABLE = True
except ImportError:
    API_REGISTRY_AVAILABLE = False


class ToolStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolStatus
    message: str
    data: Optional[Any] = None
    screenshot: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


# Common Windows applications and their launch commands
APP_COMMANDS = {
    # Browsers
    "chrome": "start chrome",
    "google chrome": "start chrome",
    "google": "start chrome",  # Common alias
    "firefox": "start firefox",
    "edge": "start msedge",
    "microsoft edge": "start msedge",
    "brave": "start brave",

    # System apps
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "this pc": "explorer",
    "my computer": "explorer",
    "control panel": "control",
    "settings": "start ms-settings:",
    "task manager": "taskmgr",

    # Terminal
    "cmd": "start cmd",
    "command prompt": "start cmd",
    "terminal": "start wt",  # Windows Terminal
    "windows terminal": "start wt",
    "powershell": "start powershell",

    # Code editors
    "code": "code",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "visual studio": "start devenv",
    "notepad++": "start notepad++",
    "sublime": "start sublime_text",

    # Office
    "word": "start winword",
    "microsoft word": "start winword",
    "excel": "start excel",
    "microsoft excel": "start excel",
    "powerpoint": "start powerpnt",
    "outlook": "start outlook",
    "onenote": "start onenote",

    # Media
    "paint": "mspaint",
    "photos": "start ms-photos:",
    "camera": "start microsoft.windows.camera:",
    "spotify": "start spotify",
    "vlc": "start vlc",

    # Communication
    "whatsapp": "start whatsapp",
    "discord": "start discord",
    "telegram": "start telegram",
    "teams": "start msteams",
    "microsoft teams": "start msteams",
    "zoom": "start zoom",
    "skype": "start skype",
    "slack": "start slack",

    # Other
    "snipping tool": "start snippingtool",
    "screenshot": "start snippingtool",
    "magnifier": "magnify",
    "sticky notes": "start ms-stickynotes:",
    "clock": "start ms-clock:",
    "calendar": "start outlookcal:",
    "mail": "start outlookmail:",
    "store": "start ms-windows-store:",
    "xbox": "start xbox:",
}


class WindowsTools:
    """Collection of Windows automation tools with vision capabilities."""

    def __init__(self, vision_instance=None, use_omniparser=True):
        self._delay = config.agent.action_delay
        self._safe_mode = config.agent.safe_mode
        self._screenshot_dir = "screenshots"
        self._audit_path = Path("audit_log.jsonl")
        # Use provided vision instance, or fall back to global one
        self._vision = vision_instance if vision_instance else (vision if VISION_AVAILABLE else None)

        # OmniParser for advanced screen understanding
        self._omniparser = None
        self._use_omniparser = use_omniparser and OMNIPARSER_AVAILABLE
        self._last_parsed_screen = None  # Cache for performance

        # LLM for semantic matching
        self._llm = None
        self._use_semantic_matching = LLM_AVAILABLE

        # API registry for application resolution
        self._api_registry = APIRegistry() if API_REGISTRY_AVAILABLE else None

        if self._use_omniparser:
            logger.info("OmniParser mode enabled for advanced UI detection")

    def _get_omniparser(self):
        """Get or initialize OmniParser instance."""
        if self._omniparser is None and self._use_omniparser:
            self._omniparser = get_omniparser(preload=False)
        return self._omniparser

    def _get_llm(self):
        """Get or initialize LLM client for semantic matching."""
        if self._llm is None and self._use_semantic_matching:
            self._llm = get_llm_client()
        return self._llm

    def _audit_event(self, event_type: str, payload: dict) -> None:
        """Append a tool-level audit event to audit_log.jsonl."""
        try:
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "windows_tools",
                "event_type": event_type,
                **payload,
            }
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Audit event write failed: {e}")

    def _resolve_app_from_registry(self, query: str) -> Optional[dict]:
        """
        Resolve an app from API registry by command text/keyword match.
        """
        if not self._api_registry:
            return None

        try:
            matches = [
                item for item in self._api_registry.find_relevant(query)
                if item.get("kind") == "app"
            ]
            if not matches:
                return None

            normalized = query.lower().strip()
            for item in matches:
                name = str(item.get("name", "")).lower()
                api_id = str(item.get("id", "")).lower()
                if normalized == name or normalized == api_id:
                    return item

            return matches[0]
        except Exception as e:
            logger.warning(f"Registry app resolution failed: {e}")
            return None

    def _semantic_match(self, user_query: str, elements: list) -> Optional[dict]:
        """
        Use LLM to find the best matching element for a user query.
        Handles cases like "C drive" -> "Local Disk (C:)" or
        "click the video" -> video title instead of channel icon.
        """
        if not self._use_semantic_matching or not elements:
            return None

        llm = self._get_llm()
        if not llm:
            return None

        # Build element list for LLM
        element_list = []
        for elem in elements[:30]:  # Limit to 30 elements for speed
            element_list.append(f"[{elem.get('id', 0)}] \"{elem.get('label', '')}\" (type: {elem.get('type', 'unknown')})")

        prompt = f"""You are a UI element matcher. Given a user's request and a list of visible screen elements, find the BEST matching element.

USER WANTS TO CLICK: "{user_query}"

VISIBLE ELEMENTS ON SCREEN:
{chr(10).join(element_list)}

MATCHING RULES:
1. Match by MEANING, not just exact text:
   - "C drive" = "Local Disk (C:)" or "C:" or "Local Disk C"
   - "video about X" = click the VIDEO TITLE containing X, NOT the channel icon
   - "search bar" = "Search" input field
   - "close button" = "X" or "Close"

2. For YouTube/videos: prefer clicking the VIDEO TITLE (text) over thumbnails or channel icons
3. For drives: "C drive", "D drive" means the drive letter in parentheses like "(C:)"
4. Prefer buttons/links over static text when user wants to "click" something

Respond with ONLY a JSON object:
{{"match": true/false, "element_id": <id>, "element_label": "<label>", "reason": "brief explanation"}}

If no good match exists, respond: {{"match": false, "reason": "why"}}"""

        try:
            response = llm.generate(prompt)
            content = response.content.strip()

            # Parse JSON response
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                if result.get("match"):
                    logger.info(f"Semantic match: '{user_query}' -> [{result.get('element_id')}] '{result.get('element_label')}' ({result.get('reason')})")
                    return result
                else:
                    logger.info(f"No semantic match for '{user_query}': {result.get('reason')}")
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")

        return None

    def parse_screen(self) -> ToolResult:
        """Parse the screen and return all detected UI elements."""
        if not self._use_omniparser:
            return ToolResult(ToolStatus.FAILURE, "OmniParser not available")

        try:
            parser = self._get_omniparser()
            self._last_parsed_screen = parser.parse_screen()

            # Format for LLM
            prompt_text = self._last_parsed_screen.to_prompt_format()
            elements_json = self._last_parsed_screen.to_json()

            return ToolResult(
                ToolStatus.SUCCESS,
                f"Found {len(self._last_parsed_screen.elements)} UI elements",
                data={
                    "prompt": prompt_text,
                    "elements": elements_json,
                    "count": len(self._last_parsed_screen.elements)
                }
            )
        except Exception as e:
            logger.error(f"Screen parsing failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def click_element_by_id(self, element_id: int) -> ToolResult:
        """Click on a UI element by its ID (requires prior parse_screen call)."""
        if not self._last_parsed_screen:
            # Parse screen first
            parse_result = self.parse_screen()
            if not parse_result.success:
                return parse_result

        try:
            # Find element by ID
            element = None
            for e in self._last_parsed_screen.elements:
                if e.id == element_id:
                    element = e
                    break

            if not element:
                return ToolResult(ToolStatus.FAILURE, f"Element with ID {element_id} not found")

            # Click it
            cx, cy = element.center
            logger.info(f"Clicking element [{element_id}] '{element.label}' at ({cx}, {cy})")
            pyautogui.click(cx, cy)
            time.sleep(self._delay)

            return ToolResult(ToolStatus.SUCCESS, f"Clicked [{element_id}] '{element.label}' at ({cx}, {cy})")

        except Exception as e:
            logger.error(f"click_element_by_id failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    # ==================== CLICK ACTIONS ====================

    def click(self, x: int = None, y: int = None, button: str = "left", clicks: int = 1) -> ToolResult:
        """Click at screen coordinates."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")

        # If no coordinates, click at current position
        if x is None or y is None:
            x, y = pyautogui.position()

        try:
            logger.info(f"Clicking at ({x}, {y}) with {button} button")
            pyautogui.click(x, y, button=button, clicks=clicks)
            time.sleep(self._delay)
            return ToolResult(ToolStatus.SUCCESS, f"Clicked {button} at ({x}, {y})")
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def click_text(self, text: str) -> ToolResult:
        """Find text on screen and click it. Uses semantic matching for better understanding."""
        logger.info(f"Looking for text: '{text}'")

        # Try OmniParser first (more intelligent)
        if self._use_omniparser:
            try:
                parser = self._get_omniparser()
                parsed = parser.parse_screen()
                self._last_parsed_screen = parsed

                # First try exact/fuzzy string matching
                matches = parsed.find_by_label(text)

                if matches:
                    element = matches[0]
                    cx, cy = element.center
                    logger.info(f"OmniParser found [{element.id}] '{element.label}' at ({cx}, {cy})")

                    pyautogui.click(cx, cy)
                    time.sleep(self._delay)

                    return ToolResult(ToolStatus.SUCCESS, f"Clicked on '{element.label}' at ({cx}, {cy})")

                # No direct match - try SEMANTIC matching with LLM
                logger.info(f"No direct match for '{text}', trying semantic matching...")

                elements_json = parsed.to_json()
                semantic_result = self._semantic_match(text, elements_json)

                if semantic_result and semantic_result.get("match"):
                    element_id = semantic_result.get("element_id")
                    # Find element by ID
                    for elem in parsed.elements:
                        if elem.id == element_id:
                            cx, cy = elem.center
                            logger.info(f"Semantic match: clicking [{elem.id}] '{elem.label}' at ({cx}, {cy})")

                            pyautogui.click(cx, cy)
                            time.sleep(self._delay)

                            return ToolResult(
                                ToolStatus.SUCCESS,
                                f"Clicked on '{elem.label}' at ({cx}, {cy}) (semantic match for '{text}')"
                            )

                logger.info(f"OmniParser didn't find '{text}', falling back to basic OCR")

            except Exception as e:
                logger.warning(f"OmniParser failed, falling back to basic OCR: {e}")

        # Fallback to basic vision/OCR
        if not self._vision or not OCR_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "Vision/OCR not available")

        try:
            elements = self._vision.find_text_on_screen(text)

            if not elements:
                return ToolResult(ToolStatus.FAILURE, f"Text not found on screen: '{text}'")

            # Click the best match
            element = elements[0]
            cx, cy = element.center
            logger.info(f"Found '{element.text}' at ({cx}, {cy}), clicking...")

            pyautogui.click(cx, cy)
            time.sleep(self._delay)

            return ToolResult(ToolStatus.SUCCESS, f"Clicked on '{element.text}' at ({cx}, {cy})")

        except Exception as e:
            logger.error(f"click_text failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def double_click(self, x: int = None, y: int = None) -> ToolResult:
        """Double click at coordinates or current position."""
        if x is None or y is None:
            x, y = pyautogui.position()
        return self.click(x, y, clicks=2)

    def right_click(self, x: int = None, y: int = None) -> ToolResult:
        """Right click at coordinates or current position."""
        if x is None or y is None:
            x, y = pyautogui.position()
        return self.click(x, y, button="right")

    # ==================== TYPING ACTIONS ====================

    def type_text(self, text: str, interval: float = 0.02) -> ToolResult:
        """Type text using keyboard."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            logger.info(f"Typing: {text[:50]}...")
            # Use typewrite for basic ASCII, write for unicode
            pyautogui.write(text, interval=interval)
            time.sleep(self._delay)
            preview = text[:50] + "..." if len(text) > 50 else text
            return ToolResult(ToolStatus.SUCCESS, f"Typed: {preview}")
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def type_in_field(self, field_label: str, text: str) -> ToolResult:
        """Find a labeled field and type in it."""
        if not self._vision:
            return ToolResult(ToolStatus.FAILURE, "Vision not available")

        # First find and click the field
        click_result = self.click_text(field_label)
        if not click_result.success:
            return click_result

        time.sleep(0.3)

        # Then type the text
        return self.type_text(text)

    def press_key(self, key: str) -> ToolResult:
        """Press a single key."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            logger.info(f"Pressing key: {key}")
            pyautogui.press(key)
            time.sleep(self._delay)
            return ToolResult(ToolStatus.SUCCESS, f"Pressed: {key}")
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    def hotkey(self, *keys, **kwargs) -> ToolResult:
        """Press a key combination."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            # Handle both hotkey("ctrl", "c") and hotkey(keys=["ctrl", "c"])
            if 'keys' in kwargs:
                key_list = kwargs['keys']
                if isinstance(key_list, list):
                    keys = tuple(key_list)
                else:
                    keys = (key_list,)
            if not keys:
                return ToolResult(ToolStatus.FAILURE, "No keys specified")
            logger.info(f"Pressing hotkey: {'+'.join(keys)}")
            pyautogui.hotkey(*keys)
            time.sleep(self._delay)
            combo = "+".join(keys)
            return ToolResult(ToolStatus.SUCCESS, f"Hotkey: {combo}")
        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            return ToolResult(ToolStatus.FAILURE, str(e))

    # ==================== SCROLL/MOUSE ACTIONS ====================

    def scroll(self, amount: int, x: int = None, y: int = None) -> ToolResult:
        """Scroll the mouse wheel."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(amount)
            direction = "up" if amount > 0 else "down"
            return ToolResult(ToolStatus.SUCCESS, f"Scrolled {direction} by {abs(amount)}")
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    def move_mouse(self, x: int, y: int) -> ToolResult:
        """Move mouse to coordinates."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            pyautogui.moveTo(x, y, duration=0.25)
            return ToolResult(ToolStatus.SUCCESS, f"Moved to ({x}, {y})")
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    # ==================== APPLICATION ACTIONS ====================

    def get_active_window(self) -> ToolResult:
        """Get the currently focused window title."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            win = pyautogui.getActiveWindow()
            if win:
                return ToolResult(ToolStatus.SUCCESS, f"Active window: {win.title}", data={"title": win.title})
            return ToolResult(ToolStatus.FAILURE, "No active window")
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    def open_application(self, path_or_name: str) -> ToolResult:
        """Open an application by name or path. Supports Windows Search for unknown apps."""
        app_name = path_or_name.lower().strip()
        logger.info(f"Opening application: {app_name}")
        self._audit_event("open_application_started", {"query": path_or_name, "normalized_query": app_name})

        try:
            # Check known apps first
            if app_name in APP_COMMANDS:
                cmd = APP_COMMANDS[app_name]
                logger.info(f"Using command: {cmd}")
                subprocess.Popen(cmd, shell=True)
                time.sleep(2.0)
                self._audit_event("open_application_succeeded", {
                    "query": path_or_name,
                    "method": "app_commands",
                    "command": cmd,
                })
                return ToolResult(ToolStatus.SUCCESS, f"Opened: {path_or_name}")

            # Try rich API registry app entries
            registry_match = self._resolve_app_from_registry(app_name)
            if registry_match:
                exe_name = registry_match.get("exe_name")
                if exe_name:
                    launch_cmd = f"start {exe_name}" if ":" in exe_name else f'start "" "{exe_name}"'
                    logger.info(f"Using registry launch command: {launch_cmd}")
                    subprocess.Popen(launch_cmd, shell=True)
                    time.sleep(2.0)
                    self._audit_event("open_application_succeeded", {
                        "query": path_or_name,
                        "method": "api_registry",
                        "api_id": registry_match.get("id"),
                        "api_name": registry_match.get("name"),
                        "exe_name": exe_name,
                        "command": launch_cmd,
                    })
                    return ToolResult(ToolStatus.SUCCESS, f"Opened via registry: {path_or_name}")

            # Check if it's a file path
            if os.path.exists(path_or_name):
                logger.info(f"Opening file path: {path_or_name}")
                os.startfile(path_or_name)
                time.sleep(2.0)
                self._audit_event("open_application_succeeded", {
                    "query": path_or_name,
                    "method": "path",
                    "path": path_or_name,
                })
                return ToolResult(ToolStatus.SUCCESS, f"Opened: {path_or_name}")

            # Try using Windows Search (Win key + type + enter)
            logger.info(f"Using Windows Search for: {app_name}")
            pyautogui.press('win')
            time.sleep(0.5)
            pyautogui.write(app_name, interval=0.02)
            time.sleep(1.0)
            pyautogui.press('enter')
            time.sleep(2.0)
            self._audit_event("open_application_succeeded", {
                "query": path_or_name,
                "method": "windows_search",
            })
            return ToolResult(ToolStatus.SUCCESS, f"Searched and opened: {path_or_name}")

        except Exception as e:
            logger.error(f"Failed to open {path_or_name}: {e}")
            self._audit_event("open_application_failed", {
                "query": path_or_name,
                "error": str(e),
            })
            return ToolResult(ToolStatus.FAILURE, f"Failed to open {path_or_name}: {e}")

    def focus_window(self, title: str) -> ToolResult:
        """Focus a window by title."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            windows = pyautogui.getWindowsWithTitle(title)
            if windows:
                win = windows[0]
                try:
                    win.activate()
                except:
                    # Fallback: minimize then restore
                    win.minimize()
                    time.sleep(0.2)
                    win.restore()
                time.sleep(self._delay)
                return ToolResult(ToolStatus.SUCCESS, f"Focused: {win.title}")
            return ToolResult(ToolStatus.FAILURE, f"Window not found: {title}")
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    def close_window(self) -> ToolResult:
        """Close the current window using Alt+F4."""
        return self.hotkey("alt", "f4")

    # ==================== VISION ACTIONS ====================

    def take_screenshot(self, filename: str = None) -> ToolResult:
        """Take a screenshot."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        try:
            os.makedirs(self._screenshot_dir, exist_ok=True)
            if not filename:
                filename = f"screenshot_{int(time.time())}.png"
            filepath = os.path.join(self._screenshot_dir, filename)
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            logger.info(f"Screenshot saved: {filepath}")
            return ToolResult(ToolStatus.SUCCESS, f"Screenshot saved: {filepath}", screenshot=filepath)
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    def find_text(self, text: str) -> ToolResult:
        """Find text on screen and return its location."""
        if not self._vision or not OCR_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "Vision/OCR not available")

        try:
            elements = self._vision.find_text_on_screen(text)

            if not elements:
                return ToolResult(ToolStatus.FAILURE, f"Text not found: '{text}'")

            # Return info about found elements
            found = [{"text": e.text, "x": e.center[0], "y": e.center[1]} for e in elements[:5]]
            return ToolResult(ToolStatus.SUCCESS, f"Found {len(elements)} matches", data=found)

        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    def read_screen(self) -> ToolResult:
        """Read all visible text on screen."""
        if not self._vision or not OCR_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "Vision/OCR not available")

        try:
            elements = self._vision.find_all_text()
            texts = [e.text for e in elements]
            combined = " ".join(texts)
            return ToolResult(ToolStatus.SUCCESS, f"Found {len(texts)} text elements", data=texts)
        except Exception as e:
            return ToolResult(ToolStatus.FAILURE, str(e))

    # ==================== UTILITY ACTIONS ====================

    def wait(self, seconds: float) -> ToolResult:
        """Wait for specified seconds."""
        logger.info(f"Waiting {seconds} seconds...")
        time.sleep(seconds)
        return ToolResult(ToolStatus.SUCCESS, f"Waited {seconds}s")

    def get_screen_size(self) -> ToolResult:
        """Get screen dimensions."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        size = pyautogui.size()
        return ToolResult(ToolStatus.SUCCESS, f"Screen: {size.width}x{size.height}",
                         data={"width": size.width, "height": size.height})

    def get_mouse_position(self) -> ToolResult:
        """Get current mouse position."""
        if not PYAUTOGUI_AVAILABLE:
            return ToolResult(ToolStatus.FAILURE, "pyautogui not available")
        pos = pyautogui.position()
        return ToolResult(ToolStatus.SUCCESS, f"Mouse at ({pos.x}, {pos.y})",
                         data={"x": pos.x, "y": pos.y})
