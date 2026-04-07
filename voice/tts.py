"""
VOXCODE Text-to-Speech Module
Cross-platform TTS using pyttsx3.
"""

import threading
import queue
from typing import Optional, List
from dataclasses import dataclass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from config import config


@dataclass
class Voice:
    """Represents an available TTS voice."""
    id: str
    name: str
    languages: List[str]
    gender: Optional[str] = None


class TextToSpeech:
    """
    Text-to-speech engine using pyttsx3.
    Supports asynchronous speech with queue management.
    """
    
    def __init__(
        self,
        rate: int = None,
        volume: float = None,
        voice_id: str = None
    ):
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 is required. Install with: pip install pyttsx3")
        
        self._rate = rate or config.voice.tts_rate
        self._volume = volume or config.voice.tts_volume
        self._voice_id = voice_id or config.voice.tts_voice_id
        
        self._engine: Optional[pyttsx3.Engine] = None
        self._speech_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._speaking = False
        self._lock = threading.Lock()
    
    def _init_engine(self) -> pyttsx3.Engine:
        """Initialize the TTS engine."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self._rate)
        engine.setProperty('volume', self._volume)
        
        if self._voice_id:
            engine.setProperty('voice', self._voice_id)
        
        return engine
    
    def speak(self, text: str, block: bool = False) -> None:
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            block: If True, wait for speech to complete
        """
        if not text:
            return
        
        if block:
            self._speak_sync(text)
        else:
            self._speech_queue.put(text)
            self._ensure_worker_running()
    
    def _speak_sync(self, text: str) -> None:
        """Speak text synchronously."""
        engine = self._init_engine()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    
    def _ensure_worker_running(self) -> None:
        """Ensure the background speech worker is running."""
        with self._lock:
            if not self._running:
                self._running = True
                self._worker_thread = threading.Thread(
                    target=self._speech_worker,
                    daemon=True
                )
                self._worker_thread.start()
    
    def _speech_worker(self) -> None:
        """Background worker that processes the speech queue."""
        engine = self._init_engine()
        
        while self._running:
            try:
                text = self._speech_queue.get(timeout=1.0)
                self._speaking = True
                engine.say(text)
                engine.runAndWait()
                self._speaking = False
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
                self._speaking = False
        
        engine.stop()
    
    def stop(self) -> None:
        """Stop all speech and clear the queue."""
        self._running = False
        
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
    
    def wait(self) -> None:
        """Wait for all queued speech to complete."""
        self._speech_queue.join()
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._speaking
    
    @property
    def queue_size(self) -> int:
        """Get number of items in speech queue."""
        return self._speech_queue.qsize()
    
    def get_available_voices(self) -> List[Voice]:
        """Get list of available voices."""
        engine = self._init_engine()
        voices = []
        
        for voice in engine.getProperty('voices'):
            voices.append(Voice(
                id=voice.id,
                name=voice.name,
                languages=voice.languages if hasattr(voice, 'languages') else [],
                gender=voice.gender if hasattr(voice, 'gender') else None
            ))
        
        engine.stop()
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """Change the TTS voice."""
        self._voice_id = voice_id
    
    def set_rate(self, rate: int) -> None:
        """Change speech rate (words per minute)."""
        self._rate = rate
    
    def set_volume(self, volume: float) -> None:
        """Change volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))


def say(text: str, block: bool = True) -> None:
    """Quick function to speak text."""
    tts = TextToSpeech()
    tts.speak(text, block=block)
