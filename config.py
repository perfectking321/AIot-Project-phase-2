"""
VOXCODE Configuration
Global settings for voice, LLM, and UI components.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceConfig:
    """Voice module configuration."""
    # STT Settings (Whisper)
    # Options: tiny (fast, less accurate), base, small (recommended), medium, large (slow, most accurate)
    whisper_model: str = "tiny"  # 'tiny' for speed (~1s), 'small' for accuracy
    whisper_language: str = "en"
    sample_rate: int = 16000
    channels: int = 1

    # Microphone settings
    # Set to None to use default, or specify device index from test_mic.py
    input_device_index: Optional[int] = None
    # Audio amplification (1.0 = no change, 2.0 = double volume, etc.)
    audio_gain: float = 6.0  # Higher gain for quiet microphones

    # Push-to-talk key (virtual key code)
    ptt_key: int = 0x11  # VK_CONTROL (Ctrl key)

    # TTS Settings (pyttsx3)
    tts_rate: int = 175  # Words per minute
    tts_volume: float = 1.0  # 0.0 to 1.0
    tts_voice_id: Optional[str] = None  # None = default system voice


@dataclass
class LLMConfig:
    """LLM/Brain module configuration."""
    # Provider: "ollama" or "groq"
    provider: str = "ollama"  # Switch to "ollama" to use local models

    # Ollama settings (local)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"  # Better instruction following than llama3.2

    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = "llama-3.3-70b-versatile"  # Best for planning/reasoning
    # Alternative models: "llama-3.1-8b-instant" (faster), "mixtral-8x7b-32768"

    # Common settings
    temperature: float = 0.2  # Lower for more consistent planning
    max_tokens: int = 2048
    timeout: int = 60  # seconds

    # System prompt settings
    system_prompt_file: Optional[str] = None

    @property
    def model_name(self) -> str:
        """Get current model based on provider."""
        return self.groq_model if self.provider == "groq" else self.ollama_model


@dataclass
class AgentConfig:
    """Agent module configuration."""
    max_iterations: int = 10
    retry_attempts: int = 3
    action_delay: float = 0.5  # seconds between actions
    screenshot_on_error: bool = True
    safe_mode: bool = True  # Require confirmation for destructive actions

    # Reactive agent settings
    use_reactive_for_complex: bool = True  # Auto-use reactive mode for complex tasks
    reactive_max_iterations: int = 20  # Max steps for reactive agent
    verify_actions: bool = True  # Verify each action succeeded
    screen_monitor_interval: float = 0.5  # seconds between screen captures


@dataclass
class PerceptionConfig:
    """Perception module configuration."""
    # VLM Settings
    vlm_model: str = "qwen2.5vl:7b"  # Vision-Language Model
    vlm_timeout: int = 60
    use_vlm: bool = True

    # Element grounding
    use_showui_api: bool = True  # Free API for grounding
    fallback_to_vlm: bool = True

    # Screen analysis
    analyze_on_capture: bool = True
    cache_screen_state: bool = True


@dataclass
class UIConfig:
    """TUI module configuration."""
    theme: str = "dark"
    show_timestamps: bool = True
    max_history: int = 100
    auto_scroll: bool = True


@dataclass
class MemoryConfig:
    """Memory module configuration."""
    max_episodic_events: int = 1000
    persist_episodic: bool = True
    persist_path: str = "memory/episodic_history.json"
    max_working_observations: int = 20
    vault_dir: str = "memory/vault"
    user_profile_file: str = "memory/vault/USER.yaml"
    history_file: str = "memory/vault/HISTORY.jsonl"
    failures_file: str = "memory/vault/failures.jsonl"


@dataclass
class VoxcodeConfig:
    """Main configuration container."""
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Global settings
    debug: bool = False
    log_file: Optional[str] = "voxcode.log"


# Global config instance
config = VoxcodeConfig()


def load_config_from_env():
    """Load configuration overrides from environment variables."""
    if os.getenv("VOXCODE_DEBUG"):
        config.debug = True
    if os.getenv("OLLAMA_HOST"):
        config.llm.ollama_host = os.getenv("OLLAMA_HOST")
    if os.getenv("VOXCODE_MODEL"):
        config.llm.ollama_model = os.getenv("VOXCODE_MODEL")
    if os.getenv("WHISPER_MODEL"):
        config.voice.whisper_model = os.getenv("WHISPER_MODEL")
    if os.getenv("GROQ_API_KEY"):
        config.llm.groq_api_key = os.getenv("GROQ_API_KEY")
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")


# Auto-load env config on import
load_config_from_env()
