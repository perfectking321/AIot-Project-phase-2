"""
VOXCODE Application Skills
Skills for opening, closing, and managing applications.
"""

import time
import logging
from typing import Optional

from agent.skills.base import Skill, SkillResult, SkillStatus, SkillStep

logger = logging.getLogger("voxcode.skills.app")


class OpenAppSkill(Skill):
    """Skill to open an application."""

    name = "open_app"
    description = "Open an application by name"
    params = ["app_name"]
    preconditions = []
    postconditions = ["app_visible"]

    def execute(self, app_name: str = None, **kwargs) -> SkillResult:
        """
        Open an application.

        Args:
            app_name: Name of the application to open

        Returns:
            SkillResult
        """
        if not app_name:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="app_name is required"
            )

        tools = self._get_tools()

        # First check if app is already open
        try:
            import pyautogui
            windows = pyautogui.getWindowsWithTitle(app_name)
            if windows:
                logger.info(f"{app_name} is already open, focusing it")
                try:
                    windows[0].activate()
                    time.sleep(0.5)
                    return SkillResult(
                        status=SkillStatus.SUCCESS,
                        message=f"{app_name} was already open, focused it",
                        data={"already_open": True}
                    )
                except:
                    pass
        except:
            pass

        # Open the application
        result = tools.open_application(app_name)

        if result.success:
            # Wait for app to open
            time.sleep(1.5)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Opened {app_name}",
                data={"app_name": app_name}
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Failed to open {app_name}: {result.message}"
            )

    def check_preconditions(self, screen_state=None) -> tuple[bool, str]:
        """No preconditions for opening an app."""
        return True, "No preconditions"

    def verify_postconditions(self, screen_state=None) -> tuple[bool, str]:
        """Verify the app is now visible."""
        # Could use VLM to verify app opened
        return True, "App should be open"


class CloseAppSkill(Skill):
    """Skill to close an application."""

    name = "close_app"
    description = "Close an application"
    params = ["app_name"]
    preconditions = ["app_visible"]
    postconditions = ["app_closed"]

    def execute(self, app_name: str = None, **kwargs) -> SkillResult:
        """
        Close an application.

        Args:
            app_name: Name of the application to close

        Returns:
            SkillResult
        """
        if not app_name:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="app_name is required"
            )

        tools = self._get_tools()

        try:
            import pyautogui

            # Find windows matching the app name
            windows = pyautogui.getWindowsWithTitle(app_name)

            if not windows:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"{app_name} is not open",
                    data={"already_closed": True}
                )

            # Focus and close
            win = windows[0]
            try:
                win.activate()
                time.sleep(0.3)
            except:
                pass

            # Close with Alt+F4
            result = tools.hotkey("alt", "f4")

            if result.success:
                time.sleep(0.5)
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Closed {app_name}"
                )

        except Exception as e:
            logger.error(f"Failed to close {app_name}: {e}")

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Failed to close {app_name}"
        )


class FocusAppSkill(Skill):
    """Skill to focus/bring to front an application."""

    name = "focus_app"
    description = "Focus an application window"
    params = ["app_name"]
    preconditions = ["app_running"]
    postconditions = ["app_focused"]

    def execute(self, app_name: str = None, **kwargs) -> SkillResult:
        """
        Focus an application.

        Args:
            app_name: Name of the application to focus

        Returns:
            SkillResult
        """
        if not app_name:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="app_name is required"
            )

        tools = self._get_tools()
        result = tools.focus_window(app_name)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Focused {app_name}",
                data={"app_name": app_name}
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Failed to focus {app_name}: {result.message}"
            )
