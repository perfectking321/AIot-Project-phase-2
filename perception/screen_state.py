"""
VOXCODE Screen State Management
Semantic state tracking for intelligent decision making.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("voxcode.perception.state")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class AppState(Enum):
    """Common application states."""
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"
    VISIBLE = "visible"
    FOCUSED = "focused"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"

    # Messaging app states
    CHAT_LIST = "chat_list"
    CHAT_OPEN = "chat_open"
    COMPOSING = "composing"

    # Browser states
    NEW_TAB = "new_tab"
    NAVIGATING = "navigating"
    PAGE_LOADED = "page_loaded"
    SEARCH_RESULTS = "search_results"

    # Media states
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class AppInfo:
    """Information about a detected application."""
    name: str
    state: AppState = AppState.VISIBLE
    details: str = ""
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.time)

    def is_ready_for_input(self) -> bool:
        """Check if app is ready to receive input."""
        return self.state in [
            AppState.READY,
            AppState.CHAT_OPEN,
            AppState.COMPOSING,
            AppState.PAGE_LOADED
        ]


@dataclass
class InteractableElement:
    """An element that can be interacted with."""
    name: str
    element_type: str  # button, input, link, etc.
    description: str = ""
    location_hint: str = ""  # e.g., "bottom right", "center"
    ready: bool = True


@dataclass
class SemanticState:
    """
    Semantic understanding of the current screen state.
    Goes beyond just elements to understand meaning and context.
    """
    timestamp: float = field(default_factory=time.time)

    # App awareness
    active_app: Optional[AppInfo] = None
    visible_apps: List[AppInfo] = field(default_factory=list)

    # What can be done
    ready_actions: List[str] = field(default_factory=list)
    interactable_elements: List[InteractableElement] = field(default_factory=list)

    # Context
    current_context: str = ""  # e.g., "WhatsApp chat with John open, message field ready"
    raw_description: str = ""  # Full VLM description

    # Confidence
    confidence: float = 1.0

    def get_app(self, name: str) -> Optional[AppInfo]:
        """Get info about a specific app."""
        name_lower = name.lower()

        if self.active_app and name_lower in self.active_app.name.lower():
            return self.active_app

        for app in self.visible_apps:
            if name_lower in app.name.lower():
                return app

        return None

    def is_app_visible(self, name: str) -> bool:
        """Check if an app is visible on screen."""
        return self.get_app(name) is not None

    def is_app_ready(self, name: str) -> bool:
        """Check if an app is visible and ready for input."""
        app = self.get_app(name)
        return app is not None and app.is_ready_for_input()

    def can_perform_action(self, action: str) -> bool:
        """Check if an action is currently possible."""
        action_lower = action.lower()
        for ready_action in self.ready_actions:
            if action_lower in ready_action.lower():
                return True
        return False

    def describe(self) -> str:
        """Get a concise description of current state."""
        parts = []

        if self.active_app:
            parts.append(f"Active: {self.active_app.name} ({self.active_app.state.value})")

        if self.visible_apps:
            visible = [a.name for a in self.visible_apps if a != self.active_app]
            if visible:
                parts.append(f"Also visible: {', '.join(visible)}")

        if self.ready_actions:
            parts.append(f"Can: {', '.join(self.ready_actions[:5])}")

        return " | ".join(parts) if parts else "Unknown state"


@dataclass
class ScreenState:
    """
    Complete screen state combining element detection and semantic understanding.
    """
    timestamp: float = field(default_factory=time.time)

    # Raw data
    screenshot: Optional[Image.Image] = None

    # Element detection (from OmniParser)
    elements: List[Dict[str, Any]] = field(default_factory=list)
    element_count: int = 0

    # Semantic understanding (from VLM)
    semantic: Optional[SemanticState] = None

    # Active window (from PyAutoGUI)
    active_window: str = ""

    def merge_element_detection(self, elements: List[Dict], count: int):
        """Add element detection results."""
        self.elements = elements
        self.element_count = count

    def merge_semantic(self, semantic: SemanticState):
        """Add semantic understanding."""
        self.semantic = semantic

    def get_element_by_label(self, label: str) -> Optional[Dict]:
        """Find an element by label."""
        label_lower = label.lower()
        for elem in self.elements:
            if label_lower in elem.get('label', '').lower():
                return elem
        return None

    def get_interactable_elements(self) -> List[Dict]:
        """Get all interactable elements."""
        return [e for e in self.elements if e.get('interactable', True)]

    def describe(self) -> str:
        """Get a description of the screen state."""
        parts = []

        if self.active_window:
            parts.append(f"Window: {self.active_window}")

        if self.semantic:
            parts.append(self.semantic.describe())
        elif self.element_count > 0:
            parts.append(f"Elements: {self.element_count} detected")

        return " | ".join(parts) if parts else "Unknown"

    def to_prompt_context(self) -> str:
        """Format screen state for LLM prompt."""
        lines = []

        lines.append("=== CURRENT SCREEN STATE ===")

        if self.active_window:
            lines.append(f"Focused Window: {self.active_window}")

        if self.semantic:
            if self.semantic.active_app:
                app = self.semantic.active_app
                lines.append(f"Active App: {app.name} (State: {app.state.value})")
                if app.details:
                    lines.append(f"  Details: {app.details}")

            if self.semantic.visible_apps:
                visible = [f"{a.name}" for a in self.semantic.visible_apps]
                lines.append(f"Visible Apps: {', '.join(visible)}")

            if self.semantic.ready_actions:
                lines.append(f"Ready Actions: {', '.join(self.semantic.ready_actions)}")

            if self.semantic.interactable_elements:
                lines.append("Interactable Elements:")
                for elem in self.semantic.interactable_elements[:10]:
                    lines.append(f"  - {elem.name} ({elem.element_type})")

            if self.semantic.current_context:
                lines.append(f"Context: {self.semantic.current_context}")

        elif self.elements:
            lines.append(f"Detected {self.element_count} UI elements:")
            for elem in self.elements[:15]:
                label = elem.get('label', 'unknown')
                etype = elem.get('type', 'element')
                lines.append(f"  [{elem.get('id', '?')}] {label} ({etype})")

        return "\n".join(lines)


class ScreenStateParser:
    """Parse VLM responses into structured SemanticState."""

    # Keywords to detect app states
    STATE_KEYWORDS = {
        AppState.CHAT_OPEN: ["chat open", "conversation", "messaging", "type a message"],
        AppState.CHAT_LIST: ["chat list", "conversations list", "chats"],
        AppState.COMPOSING: ["typing", "composing", "message field focused"],
        AppState.LOADING: ["loading", "please wait", "connecting"],
        AppState.SEARCH_RESULTS: ["search results", "showing results"],
        AppState.PLAYING: ["playing", "now playing"],
        AppState.PAUSED: ["paused"],
        AppState.ERROR: ["error", "failed", "not responding"],
    }

    # App name patterns
    APP_PATTERNS = {
        "WhatsApp": ["whatsapp", "wa"],
        "Chrome": ["chrome", "google chrome"],
        "Firefox": ["firefox", "mozilla"],
        "Edge": ["edge", "microsoft edge"],
        "Notepad": ["notepad"],
        "VS Code": ["vs code", "vscode", "visual studio code"],
        "Discord": ["discord"],
        "Telegram": ["telegram"],
        "YouTube": ["youtube"],
        "Spotify": ["spotify"],
        "File Explorer": ["file explorer", "explorer", "this pc"],
    }

    @classmethod
    def parse_vlm_response(cls, response: str) -> SemanticState:
        """Parse VLM response into SemanticState."""
        state = SemanticState()
        state.raw_description = response

        response_lower = response.lower()

        # Detect visible apps
        for app_name, patterns in cls.APP_PATTERNS.items():
            for pattern in patterns:
                if pattern in response_lower:
                    app_state = cls._detect_app_state(response_lower, app_name)
                    app_info = AppInfo(
                        name=app_name,
                        state=app_state,
                        details=cls._extract_app_details(response, app_name)
                    )
                    state.visible_apps.append(app_info)

                    # First detected app with good state is likely active
                    if state.active_app is None and app_state != AppState.UNKNOWN:
                        state.active_app = app_info
                    break

        # Extract ready actions
        action_phrases = [
            "can type", "can click", "can send", "can search",
            "ready to type", "message field", "input field",
            "send button", "search box"
        ]
        for phrase in action_phrases:
            if phrase in response_lower:
                state.ready_actions.append(phrase)

        # Build context summary
        if state.active_app:
            state.current_context = f"{state.active_app.name} - {state.active_app.state.value}"
            if state.active_app.details:
                state.current_context += f" ({state.active_app.details})"

        return state

    @classmethod
    def _detect_app_state(cls, text: str, app_name: str) -> AppState:
        """Detect the state of an app from text."""
        for state, keywords in cls.STATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return state

        # Default states based on app type
        messaging_apps = ["whatsapp", "telegram", "discord", "teams"]
        if any(app in app_name.lower() for app in messaging_apps):
            if "type a message" in text or "message" in text:
                return AppState.CHAT_OPEN

        return AppState.VISIBLE

    @classmethod
    def _extract_app_details(cls, text: str, app_name: str) -> str:
        """Extract relevant details about an app."""
        # This could be enhanced with more sophisticated NLP
        details = []

        # Look for chat/contact names
        if "chat with" in text.lower():
            start = text.lower().find("chat with") + 10
            end = min(start + 30, len(text))
            snippet = text[start:end].split()[0] if text[start:end] else ""
            if snippet:
                details.append(f"chat with {snippet}")

        return ", ".join(details) if details else ""
