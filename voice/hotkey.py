"""
VOXCODE Global Hotkey Module
Simple Ctrl+B toggle for recording.
"""

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("voxcode.hotkey")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logger.warning("keyboard library not available")


class GlobalHotkey:
    """
    Simple Ctrl+B toggle hotkey.
    Each press toggles the recording state.
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_stop: Callable[[], None] = None,  # Kept for compatibility
        hotkey: str = "ctrl+b"
    ):
        if not KEYBOARD_AVAILABLE:
            raise ImportError("keyboard library required: pip install keyboard")

        self.on_toggle = on_toggle
        self.hotkey_str = hotkey.lower()
        self._running = False
        self._last_trigger = 0.0
        self._min_interval = 0.6  # Minimum 600ms between triggers
        self._lock = threading.Lock()

        logger.info(f"GlobalHotkey initialized: {self.hotkey_str}")

    def _check_and_trigger(self):
        """Check debounce and trigger callback if allowed."""
        now = time.time()

        with self._lock:
            elapsed = now - self._last_trigger
            if elapsed < self._min_interval:
                logger.debug(f"Debounce: {elapsed:.3f}s < {self._min_interval}s, skipping")
                return False

            self._last_trigger = now
            logger.info(f"Hotkey triggered (elapsed: {elapsed:.2f}s)")

        # Call callback outside lock
        try:
            self.on_toggle()
            return True
        except Exception as e:
            logger.error(f"Callback error: {e}")
            return False

    def start(self):
        """Start listening for Ctrl+B."""
        if self._running:
            return

        self._running = True
        self._last_trigger = 0.0

        # Register the hotkey
        keyboard.add_hotkey(
            self.hotkey_str,
            self._check_and_trigger,
            suppress=False,
            trigger_on_release=False
        )

        logger.info(f"Hotkey active: Press {self.hotkey_str.upper()} to toggle recording")

    def stop(self):
        """Stop listening."""
        if not self._running:
            return

        self._running = False

        try:
            keyboard.unhook_all_hotkeys()
        except:
            pass

        logger.info("Hotkey stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def test_hotkey():
    """Interactive test."""
    print("=" * 50)
    print("HOTKEY TEST - Press Ctrl+B to toggle")
    print("Press Ctrl+C to exit")
    print("=" * 50)

    state = {"recording": False, "count": 0}

    def toggle():
        state["recording"] = not state["recording"]
        state["count"] += 1
        status = "🎤 RECORDING" if state["recording"] else "⏹️ STOPPED"
        print(f"[{time.strftime('%H:%M:%S')}] #{state['count']} {status}")

    hk = GlobalHotkey(toggle)
    hk.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        hk.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    test_hotkey()
