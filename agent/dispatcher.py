"""
VOXCODE Agent Dispatcher
Single entry point for routing commands to the appropriate execution engine.

Includes SkillRouter for pre-LLM intent routing to execution channels:
  BROWSER_DOM  — web tasks (search, navigate, click links)
  NATIVE_UIA   — native Windows app control (future)
  TERMINAL     — file system operations
  SYSTEM_API   — system settings (volume, brightness, bluetooth)
  OMNIPARSER   — fallback for unknown/visual-only UIs
"""

import logging
import re
import threading
import time
from queue import Empty, Queue
from typing import Callable, Optional

from config import config

logger = logging.getLogger("voxcode.dispatcher")

FASTPATH_RULES = [
    (r"\b(mute|unmute|silence)\b", "system_control"),
    (r"\b(volume up|volume down|louder|quieter)\b", "system_control"),
    (r"\b(take|capture|grab)\s+.{0,10}screenshot\b", "screenshot"),
    (r"\b(what apps|running apps)\b", "get_running_apps"),
    (r"\b(brightness up|brightness down|set brightness)\b", "system_control"),
]

SIMPLE_OPEN_PATTERN = re.compile(
    r"^(open|launch|start|run)\s+(\w[\w\s]*?)(\s+app(lication)?)?\s*$",
    re.IGNORECASE,
)

# Words that indicate a multi-step command (not a simple "open X")
COMPLEX_INDICATORS = re.compile(
    r"\b(and|then|after|search|play|go\s+to|navigate|type|find|click|browse|watch|visit)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════
# SkillRouter — Pre-LLM intent classification
# ═══════════════════════════════════════════════════════════════

class SkillRouter:
    """
    Decides BEFORE calling the LLM which execution channel to use.
    This prevents wasteful OmniParser calls and narrows the tool space.

    Channels:
      BROWSER_DOM  — anything web (search, open URL, YouTube, etc.)
      SYSTEM_API   — system settings (volume, wifi, bluetooth, brightness)
      TERMINAL     — file operations (list files, create folder, etc.)
      NATIVE_UIA   — native app control (Notepad, Settings, Calculator)
      OMNIPARSER   — fallback for unknown/visual-only UIs
    """

    # Patterns that indicate browser intent
    BROWSER_PATTERNS = re.compile(
        r"\b(search|google|youtube|browse|website|web|http|url|"
        r"open\s+(?:chrome|browser|firefox|edge)|"
        r"go\s+to|navigate|wikipedia|amazon|reddit|stackoverflow|github|"
        r"click\s+(?:link|button)|login|sign\s+in|download\s+from|"
        r"watch|stream|play\s+video|online)\b",
        re.IGNORECASE,
    )

    # Patterns that indicate system API
    SYSTEM_PATTERNS = re.compile(
        r"\b(volume|mute|unmute|brightness|bluetooth|wifi|"
        r"network|battery|shutdown|restart|lock|sleep|"
        r"night\s+light|dark\s+mode|display|resolution|"
        r"screenshot|screen\s+capture|clipboard|"
        r"running\s+apps|what\s+apps|kill\s+process|"
        r"ip\s+address|show\s+ip)\b",
        re.IGNORECASE,
    )

    # Patterns that indicate terminal / file operations
    # NOTE: These are specific to avoid false positives like "python tutorials"
    TERMINAL_PATTERNS = re.compile(
        r"\b(create\s+file|delete\s+file|"
        r"list\s+files|find\s+file|move\s+file|copy\s+file|"
        r"rename\s+file|read\s+file|write\s+file|save\s+as|"
        r"terminal|command\s+prompt|powershell|cmd\b|"
        r"pip\s+install|npm\s+install|git\s+clone|git\s+push|git\s+pull|"
        r"python\s+(?:script|run|execute)|run\s+python|"
        r"create\s+folder|delete\s+folder|make\s+directory)\b",
        re.IGNORECASE,
    )

    # Patterns that indicate native app control
    NATIVE_UIA_PATTERNS = re.compile(
        r"\b(notepad|calculator|settings|file\s+explorer|"
        r"paint|word|excel|powerpoint|outlook|"
        r"task\s+manager|control\s+panel|"
        r"menu\s+bar|right.click|context\s+menu)\b",
        re.IGNORECASE,
    )

    @classmethod
    def route(cls, command: str) -> str:
        """
        Classify user intent and return the execution channel name.
        Returns one of: BROWSER_DOM, SYSTEM_API, TERMINAL, NATIVE_UIA, OMNIPARSER

        Priority order: SYSTEM_API > BROWSER_DOM > TERMINAL > NATIVE_UIA > OMNIPARSER
        Browser is checked before Terminal because many queries mention tools
        (e.g. "search for python tutorials") that could false-positive on terminal patterns.
        """
        # System API has highest priority (specific, fast-path commands)
        if cls.SYSTEM_PATTERNS.search(command):
            return "SYSTEM_API"

        # Browser/web tasks — checked BEFORE terminal to avoid false positives
        if cls.BROWSER_PATTERNS.search(command):
            return "BROWSER_DOM"

        # Terminal/file operations
        if cls.TERMINAL_PATTERNS.search(command):
            return "TERMINAL"

        # Native Windows app control
        if cls.NATIVE_UIA_PATTERNS.search(command):
            return "NATIVE_UIA"

        # Default fallback
        return "OMNIPARSER"


class CommandDispatcher:
    """
    Routes commands:
    1. Fastpath direct tool calls
    2. AgentLoop planning path for simple commands
    3. ReactiveAgent for complex commands
    """

    def __init__(
        self,
        on_message: Callable[[str], None] = None,
        on_state_change: Callable[[str], None] = None,
    ):
        self.on_message = on_message or (lambda m: None)
        self.on_state_change = on_state_change or (lambda s: None)

        self._agent_loop = None
        self._reactive_agent = None
        self._tools = None

        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._dispatch_lock = threading.Lock()

    def _get_tools(self):
        if self._tools is None:
            from agent.tools import WindowsTools

            self._tools = WindowsTools(use_omniparser=True)
        return self._tools

    def _get_agent_loop(self):
        if self._agent_loop is None:
            from agent.loop import AgentLoop

            self._agent_loop = AgentLoop(
                on_message=self.on_message,
                on_state_change=self.on_state_change,
            )
        return self._agent_loop

    def _get_reactive_agent(self):
        if self._reactive_agent is None:
            from agent.reactive_loop import ReactiveAgent

            self._reactive_agent = ReactiveAgent(
                on_message=self.on_message,
                on_state_change=self.on_state_change,
                max_iterations=config.agent.reactive_max_iterations,
                verify_actions=config.agent.verify_actions,
            )
        return self._reactive_agent

    def start(self):
        """Start async command worker."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="dispatcher-worker",
        )
        self._worker_thread.start()
        logger.info("CommandDispatcher started")

    def stop(self):
        """Stop async command worker."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        logger.info("CommandDispatcher stopped")

    def submit(self, command: str):
        """Submit a command for async processing."""
        command = (command or "").strip()
        if not command:
            return
        if self._reactive_agent and self._reactive_agent.is_running:
            self._reactive_agent.cancel()
            time.sleep(0.1)
        self._queue.put(command)

    def dispatch(self, command: str) -> str:
        """Synchronously dispatch a command."""
        command = (command or "").strip()
        if not command:
            return "No command provided"

        with self._dispatch_lock:
            return self._dispatch(command)

    def _process_queue(self):
        while self._running:
            try:
                command = self._queue.get(timeout=1.0)
                if command is None:
                    break
                self.dispatch(command)
            except Empty:
                continue
            except Exception as exc:
                logger.error(f"Dispatcher worker error: {exc}", exc_info=True)

    def _cancel_active_reactive(self):
        if self._reactive_agent and self._reactive_agent.is_running:
            self._reactive_agent.cancel()
            time.sleep(0.1)

    def _dispatch(self, command: str) -> str:
        """Route command to the best handler tier."""
        self._cancel_active_reactive()

        # Step 1: SkillRouter pre-classification (logged for observability)
        channel = SkillRouter.route(command)
        logger.info(f"SkillRouter: '{command}' → {channel}")

        # Step 2: Fastpath (direct tool calls, no LLM needed)
        fastpath_result = self._try_fastpath(command)
        if fastpath_result is not None:
            self.on_message(f"✓ {fastpath_result}")
            return fastpath_result

        if SIMPLE_OPEN_PATTERN.match(command) and not COMPLEX_INDICATORS.search(command):
            logger.info(f"Simple open command: {command}")
            return self._get_agent_loop().process_command(command)

        if config.agent.use_reactive_for_complex:
            logger.info(f"Reactive command: {command}")
            return self._get_reactive_agent().process_goal(command)

        return self._get_agent_loop().process_command(command)

    def _try_fastpath(self, command: str) -> Optional[str]:
        cmd_lower = command.lower().strip()
        tools = self._get_tools()

        if re.search(r"\b(unmute|un-mute|audio on|sound on)\b", cmd_lower):
            result = tools.control_system_setting("unmute")
            return result.message if result.success else None

        if re.search(r"\bmute\b", cmd_lower) and "unmute" not in cmd_lower:
            result = tools.control_system_setting("mute")
            return result.message if result.success else None

        if re.search(r"\b(take|capture|grab)\s+.{0,10}screenshot\b", cmd_lower):
            result = tools.take_screenshot()
            return result.message if result.success else None

        if re.search(r"\bvolume\b", cmd_lower):
            result = tools.control_system_setting(command)
            return result.message if result.success else None

        # ── Brightness fastpath (max/min/level) ──
        if re.search(r"\bbrightness\b", cmd_lower):
            result = tools.control_system_setting(command)
            if result.success:
                return result.message
            # WMI failed — try BrightnessControlSkill directly as fallback
            try:
                from agent.skills.system_skills import BrightnessControlSkill
                skill = BrightnessControlSkill()
                level = "min" if any(t in cmd_lower for t in ["min", "minimum", "lowest", "zero"]) else \
                        "max" if any(t in cmd_lower for t in ["max", "maximum", "full", "highest"]) else \
                        next((w for w in cmd_lower.split() if w.isdigit()), "50")
                skill_result = skill.execute(level=level)
                if skill_result.success:
                    return skill_result.message
            except Exception as e:
                logger.warning(f"BrightnessControlSkill fallback failed: {e}")
            return None  # Let reactive agent try as last resort

        if re.search(r"\b(running apps|what apps)\b", cmd_lower):
            result = tools.get_running_apps()
            return result.message if result.success else None

        # ── Show IP fastpath ──
        if re.search(r"\b(show|what('?s| is)|get|display|my)\b.*\b(ip|ip\s*address)\b", cmd_lower):
            from agent.skills.system_skills import NetworkInfoSkill
            skill = NetworkInfoSkill()
            result = skill.execute(action="ip")
            if result.success:
                output = result.data.get("output", "") if result.data else ""
                return f"Your IP addresses:\n{output}" if output else result.message
            return result.message

        # ── Bluetooth fastpath ──
        if re.search(r"\bbluetooth\b", cmd_lower):
            from agent.skills.system_skills import BluetoothControlSkill
            skill = BluetoothControlSkill()
            if re.search(r"\b(off|disable|turn\s*off)\b", cmd_lower):
                result = skill.execute(action="off")
            elif re.search(r"\b(on|enable|turn\s*on)\b", cmd_lower):
                result = skill.execute(action="on")
            else:
                result = skill.execute(action="status")
            return result.message if result.success else result.message

        # ── DOM Browser Search fastpath ──
        # Detect: "search for X", "google X", "search about X in wikipedia", etc.
        if re.search(r"\b(search|google|look\s*up|find)\b.*\b(about|for|me)\b", cmd_lower) or \
           re.search(r"\b(open google|search google)\b", cmd_lower):
            dom_result = self._try_dom_search(command, cmd_lower)
            if dom_result is not None:
                return dom_result

        return None

    def _try_dom_search(self, command: str, cmd_lower: str) -> Optional[str]:
        """Handle browser search commands directly via DOM skills."""
        try:
            from agent.skills.dom_browser_skills import DOMSearchExtractSkill, DOMClickRefSkill, DOMReadPageSkill, PLAYWRIGHT_AVAILABLE
            if not PLAYWRIGHT_AVAILABLE:
                return None
        except ImportError:
            return None

        # Extract search query from command
        query = self._extract_search_query(command, cmd_lower)
        if not query:
            return None

        self.on_message(f"🔍 Searching: {query}")
        logger.info(f"DOM search fastpath: query='{query}'")

        # Step 1: Search Google
        search_skill = DOMSearchExtractSkill()
        search_result = search_skill.execute(query=query)

        if not search_result.success:
            logger.warning(f"DOM search failed: {search_result.message}")
            return None

        results = search_result.data.get("results", []) if search_result.data else []
        if not results:
            return f"No results found for: {query}"

        # Step 2: Check if user wants Wikipedia specifically
        wants_wikipedia = "wikipedia" in cmd_lower
        target_result = None

        if wants_wikipedia:
            for r in results:
                if "wikipedia" in r.get("link", "").lower():
                    target_result = r
                    break

        if not target_result:
            target_result = results[0]

        self.on_message(f"📄 Opening: {target_result.get('title', '')}")

        # Step 3: Click the result
        click_skill = DOMClickRefSkill()
        click_result = click_skill.execute(text=target_result.get("title", ""))

        if not click_result.success:
            # Fallback: try direct navigation
            try:
                from agent.skills.dom_browser_skills import DOMNavigateSkill
                nav_skill = DOMNavigateSkill()
                nav_skill.execute(url=target_result["link"])
            except Exception as nav_e:
                logger.warning(f"Navigation fallback failed: {nav_e}")

        import time
        time.sleep(2)

        # Step 4: Read the page content
        read_skill = DOMReadPageSkill()
        read_result = read_skill.execute(max_chars=2000)

        if read_result.success and read_result.data:
            title = read_result.data.get("title", "")
            url = read_result.data.get("url", "")
            body = read_result.data.get("body_text", "")[:800]
            return f"Opened: {title}\nURL: {url}\n\nContent:\n{body}"
        else:
            return f"Opened: {target_result.get('title', '')} ({target_result.get('link', '')})"

    @staticmethod
    def _extract_search_query(command: str, cmd_lower: str) -> str:
        """Extract the search query from a natural-language browser command."""
        import re as _re

        # "search me about the movie john wick in wikipedia"
        # → "john wick movie wikipedia"
        patterns = [
            r"search\s+(?:me\s+)?(?:about|for)\s+(.+?)(?:\s+in\s+(?:google|wikipedia|the\s+web))?$",
            r"google\s+(.+?)(?:\s+in\s+(?:wikipedia))?$",
            r"look\s*up\s+(.+?)$",
            r"find\s+(?:me\s+)?(?:about|info|information)?\s*(.+?)$",
            r"search\s+(.+?)$",
        ]
        for pattern in patterns:
            match = _re.search(pattern, cmd_lower.strip())
            if match:
                query = match.group(1).strip()
                # Remove filler words
                query = _re.sub(r"\b(the|a|an|me|please|can you|could you)\b", "", query).strip()
                query = _re.sub(r"\s+", " ", query).strip()
                if query:
                    # Add "wikipedia" if user mentioned it
                    if "wikipedia" in cmd_lower and "wikipedia" not in query:
                        query += " wikipedia"
                    return query

        return ""


_dispatcher: Optional[CommandDispatcher] = None


def get_dispatcher(**kwargs) -> CommandDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = CommandDispatcher(**kwargs)
    else:
        if "on_message" in kwargs and kwargs["on_message"] is not None:
            _dispatcher.on_message = kwargs["on_message"]
        if "on_state_change" in kwargs and kwargs["on_state_change"] is not None:
            _dispatcher.on_state_change = kwargs["on_state_change"]
    return _dispatcher
