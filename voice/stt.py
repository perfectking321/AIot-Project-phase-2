"""
VOXCODE Speech-to-Text Module
Push-to-talk audio recording with Whisper transcription.
"""

import io
import wave
import threading
import logging
import time
from typing import Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("voxcode.stt")

# Check for pyaudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.debug("pyaudio available")
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("pyaudio not available")

# Check for whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.debug("whisper available")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("whisper not available")

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    text: str
    language: str
    confidence: float
    duration: float


class AudioRecorder:
    """
    Push-to-talk audio recorder.
    Records audio while recording is active.
    """

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        chunk_size: int = 1024,
        device_index: int = None
    ):
        if not PYAUDIO_AVAILABLE:
            raise ImportError("pyaudio required. Install: pip install pyaudio")

        self.sample_rate = sample_rate or config.voice.sample_rate
        self.channels = channels or config.voice.channels
        self.chunk_size = chunk_size
        self.device_index = device_index or config.voice.input_device_index

        self._audio = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._frames: list = []
        self._is_recording = False
        self._lock = threading.Lock()
        self._record_thread: Optional[threading.Thread] = None

        # Get device info
        if self.device_index is not None:
            dev_info = self._audio.get_device_info_by_index(self.device_index)
            logger.info(f"AudioRecorder using device [{self.device_index}]: {dev_info['name']}")
        else:
            dev_info = self._audio.get_default_input_device_info()
            logger.info(f"AudioRecorder using default device: {dev_info['name']}")

        logger.info(f"AudioRecorder init: {self.sample_rate}Hz, {self.channels}ch")

    def start_recording(self) -> None:
        """Start recording audio."""
        with self._lock:
            if self._is_recording:
                logger.warning("Already recording")
                return

            self._frames = []
            logger.info("Opening audio stream...")

            try:
                # Build stream kwargs
                stream_kwargs = {
                    'format': pyaudio.paInt16,
                    'channels': self.channels,
                    'rate': self.sample_rate,
                    'input': True,
                    'frames_per_buffer': self.chunk_size
                }

                # Add device index if specified
                if self.device_index is not None:
                    stream_kwargs['input_device_index'] = self.device_index

                self._stream = self._audio.open(**stream_kwargs)
                self._is_recording = True

                # Start recording in a separate thread
                self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
                self._record_thread.start()

                logger.info("Recording started")
            except Exception as e:
                logger.error(f"Failed to open audio stream: {e}")
                raise

    def _record_loop(self):
        """Recording loop that runs in a separate thread."""
        logger.debug("Recording loop started")
        while self._is_recording and self._stream:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                self._frames.append(data)
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                break
        logger.debug("Recording loop ended")

    def stop_recording(self) -> bytes:
        """Stop recording and return audio data as WAV bytes."""
        with self._lock:
            if not self._is_recording:
                logger.warning("Not recording")
                return b""

            self._is_recording = False

            # Wait for recording thread to finish
            if self._record_thread and self._record_thread.is_alive():
                self._record_thread.join(timeout=1.0)

            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except:
                    pass
                self._stream = None

            frame_count = len(self._frames)
            logger.info(f"Recording stopped: {frame_count} frames")

            if frame_count == 0:
                return b""

            wav_data = self._frames_to_wav(self._frames)
            logger.info(f"WAV data: {len(wav_data)} bytes")
            return wav_data

    def _frames_to_wav(self, frames: list) -> bytes:
        """Convert raw audio frames to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        buffer.seek(0)
        return buffer.read()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def cleanup(self) -> None:
        """Release audio resources."""
        logger.info("Cleaning up audio resources")
        self._is_recording = False
        if self._stream:
            try:
                self._stream.close()
            except:
                pass
        try:
            self._audio.terminate()
        except:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class SpeechToText:
    """Whisper-based speech-to-text transcription."""

    def __init__(self, model_name: str = None, language: str = None, preload: bool = False):
        if not WHISPER_AVAILABLE:
            raise ImportError("openai-whisper required. Install: pip install openai-whisper")

        self.model_name = model_name or config.voice.whisper_model
        self.language = language or config.voice.whisper_language
        self.audio_gain = config.voice.audio_gain
        self._model = None
        self._lock = threading.Lock()

        logger.info(f"SpeechToText init: model={self.model_name}, lang={self.language}, gain={self.audio_gain}")

        # Preload model if requested (for faster first transcription)
        if preload:
            self._load_model()

    def _load_model(self):
        """Load the Whisper model (called at init if preload=True, otherwise lazy-loaded)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(f"Loading Whisper model '{self.model_name}'...")
                    self._model = whisper.load_model(self.model_name)
                    logger.info("Whisper model loaded")

    def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe audio data to text."""
        logger.info(f"Transcribing {len(audio_data)} bytes of audio...")

        self._load_model()

        # Convert WAV bytes to numpy array
        buffer = io.BytesIO(audio_data)
        try:
            with wave.open(buffer, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()

                # Convert to numpy
                audio_np = np.frombuffer(frames, dtype=np.int16)

                # If stereo, convert to mono
                if n_channels == 2:
                    audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Check raw audio level
                raw_level = np.abs(audio_np).mean()
                raw_max = np.abs(audio_np).max()
                logger.info(f"Raw audio level: mean={raw_level:.0f}, max={raw_max}")

                # Auto-gain: if audio is very quiet, boost it more
                effective_gain = self.audio_gain
                if raw_level < 50 and raw_max < 2000:
                    # Very quiet audio - use higher gain
                    effective_gain = max(self.audio_gain, 10.0)
                    logger.info(f"Auto-boosting gain to {effective_gain}x (very quiet audio)")
                elif raw_level < 100 and raw_max < 5000:
                    # Quiet audio - use moderate boost
                    effective_gain = max(self.audio_gain, 8.0)
                    logger.info(f"Auto-boosting gain to {effective_gain}x (quiet audio)")

                # Apply gain to amplify quiet audio
                if effective_gain != 1.0:
                    audio_np = np.clip(audio_np.astype(np.float32) * effective_gain, -32768, 32767).astype(np.int16)
                    boosted_level = np.abs(audio_np).mean()
                    logger.info(f"After {effective_gain}x gain: level={boosted_level:.0f}")

                # Normalize to float32 [-1, 1]
                audio_float = audio_np.astype(np.float32) / 32768.0
                duration = len(audio_float) / sample_rate

                logger.info(f"Audio duration: {duration:.2f}s, samples: {len(audio_float)}")

                # Check if audio has content (not just silence)
                audio_level = np.abs(audio_float).mean()
                logger.info(f"Normalized audio level: {audio_level:.4f}")

                if audio_level < 0.005:
                    logger.warning("Audio appears to be silent or very quiet")
                    # Still try to transcribe, Whisper might pick something up

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return TranscriptionResult(text="", language=self.language, confidence=0.0, duration=0.0)

        # Transcribe with Whisper
        logger.info("Running Whisper transcription...")
        try:
            result = self._model.transcribe(
                audio_float,
                language=self.language,
                fp16=False,  # Use fp32 for CPU compatibility
                verbose=False
            )

            text = result["text"].strip()
            detected_lang = result.get("language", self.language)

            logger.info(f"Transcription result: '{text}' (lang={detected_lang})")

            return TranscriptionResult(
                text=text,
                language=detected_lang,
                confidence=1.0,
                duration=duration
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptionResult(text="", language=self.language, confidence=0.0, duration=0.0)
