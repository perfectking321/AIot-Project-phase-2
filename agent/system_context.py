import threading
import time
import psutil
import pyperclip
import ctypes
import logging
from typing import Dict, Any

logger = logging.getLogger("voxcode.system_context")

class SystemContextProvider:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemContextProvider, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.context: Dict[str, Any] = {}
        self._running = False
        self._thread = None
        self._update_interval = 2.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="SystemContextThread")
        self._thread.start()
        # Initial synchronous update so context is not empty
        self._update_context()
        logger.info("SystemContextProvider started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("SystemContextProvider stopped.")

    def get_context(self) -> Dict[str, Any]:
        with self._lock:
            return self.context.copy()

    def _update_loop(self):
        while self._running:
            self._update_context()
            time.sleep(self._update_interval)

    def _update_context(self):
        try:
            new_context = {
                "active_window": self._get_active_window(),
                "clipboard": self._get_clipboard(),
                "battery": self._get_battery()
            }
            with self._lock:
                self.context = new_context
        except Exception as e:
            logger.error(f"Error updating system context: {e}")

    def _get_active_window(self) -> str:
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except Exception:
            return "Unknown"

    def _get_clipboard(self) -> str:
        try:
            text = pyperclip.paste()
            return text[:500] if text else ""
        except Exception:
            return ""

    def _get_battery(self) -> str:
        try:
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                if battery:
                    return f"{battery.percent}% ({'Plugged in' if battery.power_plugged else 'Discharging'})"
        except Exception:
            pass
        return "Unknown"

# Global accessor
def get_system_context() -> Dict[str, Any]:
    return SystemContextProvider().get_context()
