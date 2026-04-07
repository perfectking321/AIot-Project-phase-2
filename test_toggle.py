"""
Test hotkey toggle with simulated recording state.
Run: python test_toggle.py
"""

import logging
import threading
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from voice.hotkey import GlobalHotkey

print("=" * 50)
print("TOGGLE TEST")
print("=" * 50)
print()
print("Press Ctrl+B to toggle recording state")
print("The state should alternate: OFF -> ON -> OFF -> ON ...")
print()
print("Press Ctrl+C to exit")
print("=" * 50)
print()

# Simulated recording state
is_recording = False
lock = threading.Lock()

def handle_toggle():
    global is_recording

    with lock:
        if not is_recording:
            # Start recording
            is_recording = True
            print(f"[{time.strftime('%H:%M:%S')}] 🎤 RECORDING STARTED")
        else:
            # Stop recording
            is_recording = False
            print(f"[{time.strftime('%H:%M:%S')}] ⏹️  RECORDING STOPPED")

# Create hotkey
hk = GlobalHotkey(handle_toggle, hotkey="ctrl+b")
hk.start()

print(f"Initial state: {'RECORDING' if is_recording else 'NOT RECORDING'}")
print()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nExiting...")
finally:
    hk.stop()
    print("Hotkey stopped.")
