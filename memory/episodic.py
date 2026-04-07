"""
VOXCODE Episodic Memory
Long-term memory of session events and actions.
"""

import time
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger("voxcode.memory.episodic")


class EventType(Enum):
    """Types of events to remember."""
    GOAL_STARTED = "goal_started"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"
    ACTION_TAKEN = "action_taken"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_FAILED = "action_failed"
    APP_OPENED = "app_opened"
    APP_CLOSED = "app_closed"
    SCREEN_CHANGED = "screen_changed"
    USER_FEEDBACK = "user_feedback"
    ERROR = "error"


@dataclass
class Episode:
    """A single episode/event in memory."""
    timestamp: float
    event_type: EventType
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    related_goal: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "description": self.description,
            "data": self.data,
            "success": self.success,
            "related_goal": self.related_goal
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        return cls(
            timestamp=data["timestamp"],
            event_type=EventType(data["event_type"]),
            description=data["description"],
            data=data.get("data", {}),
            success=data.get("success", True),
            related_goal=data.get("related_goal", "")
        )

    def age_seconds(self) -> float:
        """How old is this episode in seconds."""
        return time.time() - self.timestamp

    def age_readable(self) -> str:
        """Human-readable age."""
        seconds = self.age_seconds()
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        else:
            return f"{int(seconds/3600)}h ago"


class EpisodicMemory:
    """
    Long-term episodic memory.

    Stores a history of events, actions, and outcomes for learning
    from experience and providing context.
    """

    def __init__(self, max_episodes: int = 1000, persist_path: str = None):
        """
        Initialize episodic memory.

        Args:
            max_episodes: Maximum episodes to keep
            persist_path: Optional file path for persistence
        """
        self.max_episodes = max_episodes
        self.persist_path = persist_path
        self._episodes: List[Episode] = []

        if persist_path:
            self._load_from_file()

    def add(
        self,
        event_type: EventType,
        description: str,
        data: Dict = None,
        success: bool = True,
        related_goal: str = ""
    ) -> Episode:
        """
        Add an episode to memory.

        Args:
            event_type: Type of event
            description: Human-readable description
            data: Additional data
            success: Whether the event was successful
            related_goal: The goal this relates to

        Returns:
            The created Episode
        """
        episode = Episode(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            data=data or {},
            success=success,
            related_goal=related_goal
        )

        self._episodes.append(episode)

        # Trim if needed
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

        logger.debug(f"Added episode: {event_type.value} - {description}")
        return episode

    def get_recent(self, count: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        return self._episodes[-count:]

    def get_by_type(self, event_type: EventType, limit: int = 50) -> List[Episode]:
        """Get episodes of a specific type."""
        matching = [e for e in self._episodes if e.event_type == event_type]
        return matching[-limit:]

    def get_by_goal(self, goal: str, limit: int = 50) -> List[Episode]:
        """Get episodes related to a goal."""
        goal_lower = goal.lower()
        matching = [
            e for e in self._episodes
            if goal_lower in e.related_goal.lower() or goal_lower in e.description.lower()
        ]
        return matching[-limit:]

    def get_failures(self, limit: int = 20) -> List[Episode]:
        """Get recent failures for learning."""
        failures = [e for e in self._episodes if not e.success]
        return failures[-limit:]

    def get_successes_for_action(self, action: str, limit: int = 10) -> List[Episode]:
        """Get successful episodes for a specific action type."""
        action_lower = action.lower()
        successes = [
            e for e in self._episodes
            if e.success and action_lower in e.description.lower()
        ]
        return successes[-limit:]

    def search(self, query: str, limit: int = 20) -> List[Episode]:
        """Search episodes by keyword."""
        query_lower = query.lower()
        matching = [
            e for e in self._episodes
            if query_lower in e.description.lower()
            or query_lower in str(e.data).lower()
        ]
        return matching[-limit:]

    def get_context_summary(self, max_episodes: int = 5) -> str:
        """Get a summary of recent context for prompts."""
        recent = self.get_recent(max_episodes)
        if not recent:
            return "No recent activity."

        lines = ["Recent activity:"]
        for ep in recent:
            status = "✓" if ep.success else "✗"
            lines.append(f"  [{status}] {ep.age_readable()}: {ep.description}")

        return "\n".join(lines)

    def get_goal_history(self) -> List[Dict]:
        """Get history of goals attempted."""
        goals = []
        for ep in self._episodes:
            if ep.event_type == EventType.GOAL_STARTED:
                goal_data = {
                    "goal": ep.description,
                    "started_at": ep.timestamp,
                    "completed": False,
                    "success": False
                }
                # Look for completion
                for later_ep in self._episodes:
                    if later_ep.timestamp > ep.timestamp:
                        if (later_ep.event_type == EventType.GOAL_COMPLETED
                            and later_ep.related_goal == ep.description):
                            goal_data["completed"] = True
                            goal_data["success"] = later_ep.success
                            break
                        elif (later_ep.event_type == EventType.GOAL_FAILED
                              and later_ep.related_goal == ep.description):
                            goal_data["completed"] = True
                            goal_data["success"] = False
                            break
                goals.append(goal_data)

        return goals

    def clear(self):
        """Clear all episodes."""
        self._episodes = []
        logger.info("Episodic memory cleared")

    def _load_from_file(self):
        """Load episodes from persistence file."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self._episodes = [Episode.from_dict(e) for e in data]
            logger.info(f"Loaded {len(self._episodes)} episodes from {path}")
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")

    def save(self):
        """Save episodes to persistence file."""
        if not self.persist_path:
            return

        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = [e.to_dict() for e in self._episodes]
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._episodes)} episodes to {path}")
        except Exception as e:
            logger.error(f"Failed to save episodes: {e}")

    def __len__(self) -> int:
        return len(self._episodes)
