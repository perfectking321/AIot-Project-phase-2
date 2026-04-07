"""
VOXCODE Working Memory
Short-term memory for current task context.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("voxcode.memory.working")


class AppStatus(Enum):
    """Status of an application."""
    UNKNOWN = "unknown"
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    FOCUSED = "focused"
    BACKGROUND = "background"


@dataclass
class AppMemory:
    """Memory of an application's state."""
    name: str
    status: AppStatus = AppStatus.UNKNOWN
    last_seen: float = 0
    window_title: str = ""
    state_details: str = ""  # e.g., "chat with John open"
    last_action: str = ""

    def is_open(self) -> bool:
        return self.status in [AppStatus.OPEN, AppStatus.FOCUSED, AppStatus.BACKGROUND]

    def is_focused(self) -> bool:
        return self.status == AppStatus.FOCUSED

    def update(self, status: AppStatus = None, details: str = None):
        """Update app state."""
        if status:
            self.status = status
        if details:
            self.state_details = details
        self.last_seen = time.time()


@dataclass
class CurrentTask:
    """Information about the current task."""
    goal: str
    started_at: float = field(default_factory=time.time)
    current_subtask: str = ""
    subtask_index: int = 0
    total_subtasks: int = 0
    actions_taken: List[str] = field(default_factory=list)
    last_error: str = ""

    def add_action(self, action: str):
        """Record an action taken."""
        self.actions_taken.append(action)

    def progress_text(self) -> str:
        """Get progress description."""
        if self.total_subtasks > 0:
            return f"{self.subtask_index}/{self.total_subtasks}: {self.current_subtask}"
        return self.current_subtask or self.goal


class WorkingMemory:
    """
    Short-term working memory for current context.

    Tracks:
    - Current task and progress
    - Application states
    - Recent observations
    - Temporary data
    """

    def __init__(self):
        """Initialize working memory."""
        self._current_task: Optional[CurrentTask] = None
        self._apps: Dict[str, AppMemory] = {}
        self._observations: List[str] = []
        self._scratch: Dict[str, Any] = {}  # Temporary data
        self._max_observations = 20

    # ==================== Task Management ====================

    def start_task(self, goal: str, total_subtasks: int = 0) -> CurrentTask:
        """Start a new task."""
        self._current_task = CurrentTask(
            goal=goal,
            total_subtasks=total_subtasks
        )
        logger.info(f"Working memory: Started task '{goal}'")
        return self._current_task

    def get_current_task(self) -> Optional[CurrentTask]:
        """Get the current task."""
        return self._current_task

    def update_subtask(self, subtask: str, index: int = None):
        """Update current subtask."""
        if self._current_task:
            self._current_task.current_subtask = subtask
            if index is not None:
                self._current_task.subtask_index = index

    def record_action(self, action: str):
        """Record an action taken."""
        if self._current_task:
            self._current_task.add_action(action)

    def record_error(self, error: str):
        """Record an error."""
        if self._current_task:
            self._current_task.last_error = error

    def end_task(self):
        """End the current task."""
        self._current_task = None

    # ==================== App State Management ====================

    def update_app(
        self,
        app_name: str,
        status: AppStatus = None,
        window_title: str = None,
        details: str = None,
        last_action: str = None
    ):
        """Update or create app memory."""
        app_name_lower = app_name.lower()

        if app_name_lower not in self._apps:
            self._apps[app_name_lower] = AppMemory(name=app_name)

        app = self._apps[app_name_lower]
        app.last_seen = time.time()

        if status:
            app.status = status
        if window_title:
            app.window_title = window_title
        if details:
            app.state_details = details
        if last_action:
            app.last_action = last_action

        logger.debug(f"Updated app memory: {app_name} -> {app.status.value}")

    def get_app(self, app_name: str) -> Optional[AppMemory]:
        """Get app memory."""
        return self._apps.get(app_name.lower())

    def is_app_open(self, app_name: str) -> bool:
        """Check if an app is remembered as open."""
        app = self.get_app(app_name)
        return app.is_open() if app else False

    def get_focused_app(self) -> Optional[AppMemory]:
        """Get the currently focused app."""
        for app in self._apps.values():
            if app.is_focused():
                return app
        return None

    def get_open_apps(self) -> List[AppMemory]:
        """Get all apps remembered as open."""
        return [app for app in self._apps.values() if app.is_open()]

    def mark_app_closed(self, app_name: str):
        """Mark an app as closed."""
        if app_name.lower() in self._apps:
            self._apps[app_name.lower()].status = AppStatus.CLOSED

    # ==================== Observations ====================

    def add_observation(self, observation: str):
        """Add an observation about the current state."""
        self._observations.append(observation)
        if len(self._observations) > self._max_observations:
            self._observations = self._observations[-self._max_observations:]

    def get_recent_observations(self, count: int = 5) -> List[str]:
        """Get recent observations."""
        return self._observations[-count:]

    # ==================== Scratch Space ====================

    def set(self, key: str, value: Any):
        """Set a scratch value."""
        self._scratch[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a scratch value."""
        return self._scratch.get(key, default)

    def clear_scratch(self):
        """Clear scratch space."""
        self._scratch = {}

    # ==================== Context Generation ====================

    def get_context(self) -> str:
        """Get full working memory context for prompts."""
        lines = ["=== WORKING MEMORY ==="]

        # Current task
        if self._current_task:
            lines.append(f"Current Goal: {self._current_task.goal}")
            lines.append(f"Progress: {self._current_task.progress_text()}")
            if self._current_task.actions_taken:
                recent_actions = self._current_task.actions_taken[-5:]
                lines.append(f"Recent Actions: {', '.join(recent_actions)}")
            if self._current_task.last_error:
                lines.append(f"Last Error: {self._current_task.last_error}")
        else:
            lines.append("No active task")

        # Open apps
        open_apps = self.get_open_apps()
        if open_apps:
            lines.append("")
            lines.append("Open Applications:")
            for app in open_apps:
                status = "FOCUSED" if app.is_focused() else "open"
                detail = f" ({app.state_details})" if app.state_details else ""
                lines.append(f"  - {app.name}: {status}{detail}")

        # Recent observations
        observations = self.get_recent_observations(3)
        if observations:
            lines.append("")
            lines.append("Recent Observations:")
            for obs in observations:
                lines.append(f"  - {obs}")

        lines.append("======================")
        return "\n".join(lines)

    def get_app_context(self, app_name: str) -> str:
        """Get context for a specific app."""
        app = self.get_app(app_name)
        if not app:
            return f"{app_name}: No information available"

        parts = [f"{app.name}: {app.status.value}"]
        if app.window_title:
            parts.append(f"Window: {app.window_title}")
        if app.state_details:
            parts.append(f"State: {app.state_details}")
        if app.last_action:
            parts.append(f"Last action: {app.last_action}")

        return " | ".join(parts)

    def clear(self):
        """Clear all working memory."""
        self._current_task = None
        self._apps = {}
        self._observations = []
        self._scratch = {}
        logger.info("Working memory cleared")
