"""
Full integration test without TUI.
Tests hotkey -> recording -> transcription flow.
"""

import logging
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

from voice.hotkey import GlobalHotkey
from voice.stt import AudioRecorder, SpeechToText

print("=" * 60)
print("FULL INTEGRATION TEST")
print("=" * 60)
print()

# Initialize components
print("Initializing...")
recorder = AudioRecorder()
stt = SpeechToText(preload=True)
print("Ready!")
print()

# State
is_recording = False
lock = threading.Lock()

def toggle_recording():
    global is_recording

    with lock:
        if not is_recording:
            # Start
            is_recording = True
            print(f"\n[{time.strftime('%H:%M:%S')}] 🎤 RECORDING STARTED - Speak now...")
            recorder.start_recording()
        else:
            # Stop
            is_recording = False
            print(f"[{time.strftime('%H:%M:%S')}] ⏹️  RECORDING STOPPED - Processing...")

            audio_data = recorder.stop_recording()
            print(f"Got {len(audio_data)} bytes of audio")

            if len(audio_data) > 5000:
                print("Transcribing...")
                result = stt.transcribe(audio_data)
                if result.text:
                    print(f"\n>>> You said: \"{result.text}\"\n")
                else:
                    print("No speech detected")
            else:
                print("Recording too short")

print("Press Ctrl+B to start/stop recording")
print("Press Ctrl+C to exit")
print("=" * 60)

hk = GlobalHotkey(toggle_recording)
hk.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nExiting...")
finally:
    hk.stop()
    recorder.cleanup()
