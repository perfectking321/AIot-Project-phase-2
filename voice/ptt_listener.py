"""
Push-to-talk listener.
Hold key to record, release to transcribe and dispatch command text.
"""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("voxcode.ptt")

try:
    import keyboard

    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

from voice.stt import AudioRecorder, SpeechToText


class PTTListener:
    """Monitors a hotkey and emits transcribed commands on release."""

    def __init__(self, on_command: Callable[[str], None], hotkey: str = "ctrl"):
        self.on_command = on_command
        self.hotkey = hotkey
        self._recorder = AudioRecorder()
        self._stt = SpeechToText(preload=True)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        if not KEYBOARD_AVAILABLE:
            raise ImportError("keyboard package not installed. Run: pip install keyboard")

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="ptt-listener")
        self._thread.start()
        logger.info(f"PTT listener started (hold '{self.hotkey}' to record)")

    def _listen_loop(self):
        is_pressed = False
        while self._running:
            try:
                pressed_now = keyboard.is_pressed(self.hotkey)
            except Exception as exc:
                logger.error(f"PTT key polling failed: {exc}")
                time.sleep(0.1)
                continue

            if pressed_now and not is_pressed:
                is_pressed = True
                try:
                    self._recorder.start_recording()
                except Exception as exc:
                    logger.error(f"Failed to start PTT recording: {exc}")
                    is_pressed = False

            elif not pressed_now and is_pressed:
                is_pressed = False
                try:
                    audio = self._recorder.stop_recording()
                except Exception as exc:
                    logger.error(f"Failed to stop PTT recording: {exc}")
                    audio = b""

                if audio and len(audio) > 3200:
                    try:
                        result = self._stt.transcribe(audio)
                        text = result.text.strip()
                        if len(text) > 2:
                            logger.info(f"PTT command: {text}")
                            self.on_command(text)
                    except Exception as exc:
                        logger.error(f"PTT transcription failed: {exc}")

            time.sleep(0.02)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._recorder.cleanup()
