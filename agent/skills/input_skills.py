"""
VOXCODE Input Skills
Skills for typing, clicking, and scrolling.
"""

import time
import logging
from typing import Optional, Tuple

from agent.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger("voxcode.skills.input")


class TypeTextSkill(Skill):
    """Skill to type text."""

    name = "type_text"
    description = "Type text at the current cursor position"
    params = ["text"]
    preconditions = ["input_focused"]
    postconditions = ["text_typed"]

    def execute(self, text: str = None, **kwargs) -> SkillResult:
        """
        Type text.

        Args:
            text: The text to type

        Returns:
            SkillResult
        """
        if not text:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="text is required"
            )

        tools = self._get_tools()
        result = tools.type_text(text)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Typed: {text[:30]}...",
                data={"text": text}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Failed to type: {result.message}"
        )


class ClickElementSkill(Skill):
    """Skill to click on a UI element."""

    name = "click_element"
    description = "Click on a UI element by description"
    params = ["element_description"]
    preconditions = ["element_visible"]
    postconditions = ["element_clicked"]

    def execute(
        self,
        element_description: str = None,
        coordinates: Tuple[int, int] = None,
        **kwargs
    ) -> SkillResult:
        """
        Click an element.

        Args:
            element_description: Description of element to click
            coordinates: Optional exact coordinates

        Returns:
            SkillResult
        """
        tools = self._get_tools()

        # If coordinates provided, use them directly
        if coordinates:
            x, y = coordinates
            result = tools.click(x, y)
            if result.success:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Clicked at ({x}, {y})",
                    data={"x": x, "y": y}
                )
            return SkillResult(
                status=SkillStatus.FAILED,
                message=f"Failed to click at ({x}, {y})"
            )

        # Otherwise, find element by description
        if not element_description:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="element_description or coordinates required"
            )

        # Try using OmniParser / click_text
        result = tools.click_text(element_description)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Clicked: {element_description}",
                data={"element": element_description}
            )

        # Try element grounder for more precise location
        try:
            from perception.grounder import get_grounder
            import pyautogui

            grounder = get_grounder()
            screenshot = pyautogui.screenshot()

            ground_result = grounder.ground_element(screenshot, element_description)

            if ground_result.found and ground_result.center:
                x, y = ground_result.center
                click_result = tools.click(x, y)

                if click_result.success:
                    return SkillResult(
                        status=SkillStatus.SUCCESS,
                        message=f"Clicked (via grounder): {element_description}",
                        data={"element": element_description, "x": x, "y": y}
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Grounder fallback failed: {e}")

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Could not find element: {element_description}"
        )


class ScrollSkill(Skill):
    """Skill to scroll the screen."""

    name = "scroll"
    description = "Scroll the screen up or down"
    params = ["direction", "amount"]
    preconditions = []
    postconditions = ["scrolled"]

    def execute(
        self,
        direction: str = "down",
        amount: int = 3,
        **kwargs
    ) -> SkillResult:
        """
        Scroll the screen.

        Args:
            direction: "up" or "down"
            amount: Number of scroll units

        Returns:
            SkillResult
        """
        tools = self._get_tools()

        # Convert direction to scroll amount
        scroll_amount = amount if direction.lower() == "up" else -amount

        result = tools.scroll(scroll_amount)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Scrolled {direction} by {amount}",
                data={"direction": direction, "amount": amount}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Failed to scroll: {result.message}"
        )


class HotkeySkill(Skill):
    """Skill to press keyboard shortcuts."""

    name = "hotkey"
    description = "Press a keyboard shortcut"
    params = ["keys"]
    preconditions = []
    postconditions = ["hotkey_pressed"]

    # Common hotkeys
    COMMON_HOTKEYS = {
        "copy": ("ctrl", "c"),
        "paste": ("ctrl", "v"),
        "cut": ("ctrl", "x"),
        "undo": ("ctrl", "z"),
        "redo": ("ctrl", "y"),
        "select_all": ("ctrl", "a"),
        "save": ("ctrl", "s"),
        "find": ("ctrl", "f"),
        "new": ("ctrl", "n"),
        "open": ("ctrl", "o"),
        "close": ("ctrl", "w"),
        "close_window": ("alt", "f4"),
        "switch_window": ("alt", "tab"),
        "new_tab": ("ctrl", "t"),
        "refresh": ("ctrl", "r"),
        "address_bar": ("ctrl", "l"),
    }

    def execute(
        self,
        keys: list = None,
        shortcut_name: str = None,
        **kwargs
    ) -> SkillResult:
        """
        Press a keyboard shortcut.

        Args:
            keys: List of keys to press together
            shortcut_name: Name of common shortcut

        Returns:
            SkillResult
        """
        tools = self._get_tools()

        # If shortcut name provided, look it up
        if shortcut_name:
            shortcut_lower = shortcut_name.lower().replace(" ", "_")
            if shortcut_lower in self.COMMON_HOTKEYS:
                keys = list(self.COMMON_HOTKEYS[shortcut_lower])

        if not keys:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="keys or shortcut_name is required"
            )

        result = tools.hotkey(*keys)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Pressed: {'+'.join(keys)}",
                data={"keys": keys}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Failed to press hotkey: {result.message}"
        )


class WaitSkill(Skill):
    """Skill to wait for a specified time."""

    name = "wait"
    description = "Wait for a specified number of seconds"
    params = ["seconds"]
    preconditions = []
    postconditions = []

    def execute(self, seconds: float = 1.0, **kwargs) -> SkillResult:
        """
        Wait for specified time.

        Args:
            seconds: Time to wait

        Returns:
            SkillResult
        """
        tools = self._get_tools()
        result = tools.wait(seconds)

        return SkillResult(
            status=SkillStatus.SUCCESS,
            message=f"Waited {seconds} seconds"
        )
