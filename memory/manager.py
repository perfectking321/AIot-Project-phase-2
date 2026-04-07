"""
VOXCODE Memory Manager
Unified interface for all memory types.
"""

import logging
from typing import Optional, Dict, Any

from memory.episodic import EpisodicMemory, EventType, Episode
from memory.working import WorkingMemory, AppStatus, AppMemory, CurrentTask

logger = logging.getLogger("voxcode.memory")


class MemoryManager:
    """
    Unified memory manager.

    Provides a single interface to:
    - Episodic memory (long-term history)
    - Working memory (current context)
    """

    def __init__(self, persist_path: str = None):
        """
        Initialize memory manager.

        Args:
            persist_path: Optional path for episodic memory persistence
        """
        self.episodic = EpisodicMemory(persist_path=persist_path)
        self.working = WorkingMemory()

    # ==================== Goal/Task Management ====================

    def start_goal(self, goal: str, subtasks: int = 0) -> CurrentTask:
        """
        Start working on a new goal.

        Args:
            goal: The goal description
            subtasks: Number of subtasks

        Returns:
            CurrentTask object
        """
        # Record in episodic
        self.episodic.add(
            EventType.GOAL_STARTED,
            goal,
            data={"subtasks": subtasks}
        )

        # Start in working memory
        return self.working.start_task(goal, subtasks)

    def complete_goal(self, success: bool = True, message: str = ""):
        """
        Complete the current goal.

        Args:
            success: Whether the goal was achieved
            message: Completion message
        """
        task = self.working.get_current_task()
        if task:
            event_type = EventType.GOAL_COMPLETED if success else EventType.GOAL_FAILED

            self.episodic.add(
                event_type,
                message or f"{'Completed' if success else 'Failed'}: {task.goal}",
                data={
                    "goal": task.goal,
                    "actions_count": len(task.actions_taken),
                    "last_error": task.last_error
                },
                success=success,
                related_goal=task.goal
            )

        self.working.end_task()

    def update_progress(self, subtask: str, index: int = None):
        """Update current subtask progress."""
        self.working.update_subtask(subtask, index)

    # ==================== Action Recording ====================

    def record_action(
        self,
        action: str,
        success: bool = True,
        data: Dict = None
    ):
        """
        Record an action taken.

        Args:
            action: Description of the action
            success: Whether it succeeded
            data: Additional data
        """
        # Working memory
        self.working.record_action(action)

        # Episodic memory
        event_type = EventType.ACTION_SUCCEEDED if success else EventType.ACTION_FAILED
        goal = ""
        task = self.working.get_current_task()
        if task:
            goal = task.goal
            if not success:
                task.last_error = action

        self.episodic.add(
            event_type,
            action,
            data=data or {},
            success=success,
            related_goal=goal
        )

    def record_error(self, error: str, data: Dict = None):
        """Record an error."""
        self.working.record_error(error)
        self.episodic.add(
            EventType.ERROR,
            error,
            data=data or {},
            success=False
        )

    # ==================== App State Management ====================

    def app_opened(self, app_name: str, details: str = ""):
        """Record that an app was opened."""
        self.working.update_app(
            app_name,
            status=AppStatus.FOCUSED,
            details=details
        )
        self.episodic.add(
            EventType.APP_OPENED,
            f"Opened {app_name}",
            data={"app": app_name, "details": details}
        )

    def app_closed(self, app_name: str):
        """Record that an app was closed."""
        self.working.mark_app_closed(app_name)
        self.episodic.add(
            EventType.APP_CLOSED,
            f"Closed {app_name}",
            data={"app": app_name}
        )

    def update_app_state(
        self,
        app_name: str,
        status: AppStatus = None,
        details: str = None
    ):
        """Update app state in working memory."""
        self.working.update_app(app_name, status=status, details=details)

    def is_app_open(self, app_name: str) -> bool:
        """Check if an app is open."""
        return self.working.is_app_open(app_name)

    def get_app_state(self, app_name: str) -> Optional[AppMemory]:
        """Get current state of an app."""
        return self.working.get_app(app_name)

    # ==================== Observations ====================

    def observe(self, observation: str):
        """Record an observation about the screen/state."""
        self.working.add_observation(observation)

    def screen_changed(self, description: str):
        """Record a significant screen change."""
        self.observe(description)
        self.episodic.add(
            EventType.SCREEN_CHANGED,
            description
        )

    # ==================== Context Retrieval ====================

    def get_full_context(self) -> str:
        """Get complete context for LLM prompts."""
        parts = []

        # Working memory context
        parts.append(self.working.get_context())

        # Recent episodic history
        parts.append("")
        parts.append(self.episodic.get_context_summary(5))

        return "\n".join(parts)

    def get_relevant_history(self, query: str, limit: int = 5) -> str:
        """Get relevant history for a query."""
        episodes = self.episodic.search(query, limit)
        if not episodes:
            return "No relevant history found."

        lines = [f"Relevant history for '{query}':"]
        for ep in episodes:
            status = "✓" if ep.success else "✗"
            lines.append(f"  [{status}] {ep.age_readable()}: {ep.description}")

        return "\n".join(lines)

    def get_failure_learnings(self, action_type: str = None) -> str:
        """Get learnings from past failures."""
        failures = self.episodic.get_failures(10)

        if action_type:
            failures = [f for f in failures if action_type.lower() in f.description.lower()]

        if not failures:
            return "No relevant failures to learn from."

        lines = ["Past failures to avoid:"]
        for f in failures[-5:]:
            lines.append(f"  - {f.description}")

        return "\n".join(lines)

    # ==================== Persistence ====================

    def save(self):
        """Save memory to disk."""
        self.episodic.save()

    def clear_all(self):
        """Clear all memory."""
        self.episodic.clear()
        self.working.clear()
        logger.info("All memory cleared")


# Global memory instance
_memory_instance: Optional[MemoryManager] = None


def get_memory(persist_path: str = None) -> MemoryManager:
    """Get or create global memory manager."""
    global _memory_instance

    if _memory_instance is None:
        _memory_instance = MemoryManager(persist_path=persist_path)

    return _memory_instance
