"""
Low-level UI action executor.

Architecture-aligned execution wrapper around pyautogui actions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger("voxcode.execution")

try:
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False


@dataclass
class ExecutionResult:
    success: bool
    message: str


class ActionExecutor:
    """Execute action JSON produced by the action model."""

    def __init__(self, settle_time: float = 0.3):
        self.settle_time = settle_time

    def execute(self, action: Dict[str, Any]) -> ExecutionResult:
        if not PYAUTOGUI_AVAILABLE:
            return ExecutionResult(False, "pyautogui not available")

        action_name = str(action.get("action", "wait")).lower().strip()

        try:
            if action_name == "click":
                x = int(action.get("x", 0))
                y = int(action.get("y", 0))
                if x <= 0 or y <= 0:
                    return ExecutionResult(False, f"Invalid click coordinates: ({x}, {y})")
                pyautogui.click(x, y)
                self._settle()
                return ExecutionResult(True, f"Clicked at ({x}, {y})")

            if action_name == "type":
                text = str(action.get("text", ""))
                if not text:
                    return ExecutionResult(False, "Missing text for type action")
                if self._requires_clipboard_paste(text):
                    if not PYPERCLIP_AVAILABLE:
                        logger.warning("pyperclip not available; non-ASCII text may not type correctly")
                    self._paste_text(text)
                else:
                    pyautogui.write(text, interval=0.02)
                self._settle()
                return ExecutionResult(True, f"Typed {len(text)} chars")

            if action_name == "press":
                key = str(action.get("key", "")).strip()
                if not key:
                    return ExecutionResult(False, "Missing key for press action")
                pyautogui.press(key)
                self._settle()
                return ExecutionResult(True, f"Pressed key: {key}")

            if action_name == "hotkey":
                keys = action.get("keys", [])
                if isinstance(keys, str):
                    keys = [keys]
                if not isinstance(keys, list) or not keys:
                    return ExecutionResult(False, "Missing keys for hotkey action")
                pyautogui.hotkey(*[str(k) for k in keys])
                self._settle()
                return ExecutionResult(True, f"Pressed hotkey: {'+'.join(str(k) for k in keys)}")

            if action_name == "scroll":
                amount = int(action.get("amount", -3))
                pyautogui.scroll(amount)
                self._settle()
                return ExecutionResult(True, f"Scrolled by {amount}")

            if action_name == "wait":
                seconds = float(action.get("seconds", 1))
                if seconds < 0:
                    seconds = 0
                time.sleep(seconds)
                return ExecutionResult(True, f"Waited {seconds:.1f}s")

            if action_name == "done":
                return ExecutionResult(True, "Task marked done")

            return ExecutionResult(False, f"Unknown action: {action_name}")

        except Exception as exc:
            logger.error("Action execution error: %s", exc)
            return ExecutionResult(False, str(exc))

    def _settle(self) -> None:
        if self.settle_time > 0:
            time.sleep(self.settle_time)

    @staticmethod
    def _requires_clipboard_paste(text: str) -> bool:
        return any(ord(ch) > 127 for ch in text)

    @staticmethod
    def _paste_text(text: str) -> None:
        if not PYPERCLIP_AVAILABLE:
            pyautogui.write(text, interval=0.02)
            return

        original_clipboard = None
        try:
            original_clipboard = pyperclip.paste()
        except Exception:
            pass

        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

        if original_clipboard is not None:
            try:
                pyperclip.copy(original_clipboard)
            except Exception:
                pass


_executor: ActionExecutor | None = None


def get_executor(settle_time: float = 0.3) -> ActionExecutor:
    global _executor
    if _executor is None:
        _executor = ActionExecutor(settle_time=settle_time)
    return _executor
