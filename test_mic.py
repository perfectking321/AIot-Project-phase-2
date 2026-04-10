#!/usr/bin/env python3
"""
VOXCODE Microphone Test Utility
Test and select the correct microphone.
"""

import sys
import time
import numpy as np

try:
    import pyaudio
except ImportError:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("ERROR: audio backend not installed. Run: pip install pyaudiowpatch (Windows) or pip install pyaudio")
        sys.exit(1)


def list_devices():
    """List all audio input devices."""
    p = pyaudio.PyAudio()
    print("\n=== Available Microphones ===\n")

    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))
            default = " [DEFAULT]" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {info['name']}{default}")

    p.terminate()
    return input_devices


def test_device(device_index: int, duration: float = 3.0):
    """Test a specific audio device."""
    p = pyaudio.PyAudio()
    info = p.get_device_info_by_index(device_index)

    print(f"\n=== Testing: {info['name']} ===")
    print(f"Recording for {duration} seconds - please speak loudly!\n")

    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"ERROR: Could not open device: {e}")
        p.terminate()
        return False

    frames = []
    start_time = time.time()

    print("Recording", end="", flush=True)
    while time.time() - start_time < duration:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            # Show activity
            audio = np.frombuffer(data, dtype=np.int16)
            level = np.abs(audio).mean()
            bars = int(level / 500)
            print(f"\rRecording: {'|' * min(bars, 30):<30} Level: {level:>6.0f}", end="", flush=True)
        except Exception as e:
            print(f"\nError: {e}")
            break

    stream.stop_stream()
    stream.close()

    # Analyze audio
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    mean_level = np.abs(audio_np).mean()
    max_level = np.abs(audio_np).max()

    print(f"\n\nResults:")
    print(f"  Mean level: {mean_level:.0f}")
    print(f"  Max level:  {max_level}")
    print(f"  Total samples: {len(audio_np)}")

    if max_level > 5000:
        print("\n[OK] Good audio levels! This microphone should work well.")
        return True
    elif max_level > 1000:
        print("\n[WARN] Audio level is a bit low. Try speaking louder or closer to mic.")
        return True
    else:
        print("\n[ERROR] Audio level is too low. This microphone may not be working.")
        print("        Try a different device or check Windows sound settings.")
        return False

    p.terminate()


def main():
    print("=" * 50)
    print("  VOXCODE Microphone Test")
    print("=" * 50)

    devices = list_devices()

    if not devices:
        print("\nNo input devices found!")
        return

    print("\nOptions:")
    print("  Enter device number to test")
    print("  Enter 'a' to test all devices")
    print("  Enter 'q' to quit")

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'a':
            working = []
            for idx, info in devices:
                if test_device(idx, 2.0):
                    working.append((idx, info['name']))

            print("\n=== Summary ===")
            if working:
                print("Working microphones:")
                for idx, name in working:
                    print(f"  [{idx}] {name}")
                print(f"\nRecommendation: Use device [{working[0][0]}]")
            else:
                print("No working microphones found!")
        else:
            try:
                idx = int(choice)
                test_device(idx, 3.0)
            except ValueError:
                print("Invalid input")


if __name__ == "__main__":
    main()
