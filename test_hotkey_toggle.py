"""
Test the hotkey toggle behavior.
Run: python test_hotkey_toggle.py
"""

import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from voice.hotkey import GlobalHotkey

print("=" * 60)
print("HOTKEY TOGGLE TEST")
print("=" * 60)
print()
print("Press Ctrl+B to START recording")
print("Press Ctrl+B AGAIN to STOP recording")
print()
print("Recording will NOT stop automatically!")
print("It only stops when YOU press Ctrl+B again.")
print()
print("Press Ctrl+C to exit this test.")
print("=" * 60)
print()

recording = False
start_time = None

def on_start():
    global recording, start_time
    recording = True
    start_time = time.time()
    print()
    print(f"[{time.strftime('%H:%M:%S')}] ==================")
    print(f"[{time.strftime('%H:%M:%S')}] >>> RECORDING STARTED")
    print(f"[{time.strftime('%H:%M:%S')}] ==================")
    print(f"[{time.strftime('%H:%M:%S')}] Speak now... (press Ctrl+B to stop)")

def on_stop():
    global recording, start_time
    duration = time.time() - start_time if start_time else 0
    recording = False
    start_time = None
    print()
    print(f"[{time.strftime('%H:%M:%S')}] ==================")
    print(f"[{time.strftime('%H:%M:%S')}] >>> RECORDING STOPPED")
    print(f"[{time.strftime('%H:%M:%S')}] Duration: {duration:.1f} seconds")
    print(f"[{time.strftime('%H:%M:%S')}] ==================")
    print()

hk = GlobalHotkey(on_start, on_stop, "ctrl+b")
hk.start()

try:
    while True:
        if recording and start_time:
            elapsed = time.time() - start_time
            print(f"\r  Recording: {elapsed:.1f}s elapsed...", end="", flush=True)
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\n\nExiting...")
finally:
    hk.stop()
