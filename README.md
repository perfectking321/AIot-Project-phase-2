# VOXCODE

**Voice-Controlled Windows Automation Assistant**

VOXCODE is a local, privacy-first voice assistant that automates Windows tasks using natural language commands.

## Features

- **Push-to-Talk Voice Input** - Whisper-powered speech recognition
- **Text-to-Speech** - Audio feedback using pyttsx3
- **Local LLM** - Powered by Ollama (Llama, Mistral, etc.)
- **Windows Automation** - Control apps, type, click, and more
- **Registry-Backed App Launching** - 50+ web/system/app entries for smarter command routing
- **Audit Logging** - Planner/verification/launcher events persisted to `audit_log.jsonl`
- **Unified Debug Trace** - Correlated per-run events + screenshots for planner, pipeline, Qwen, and verifier
- **Terminal UI** - Beautiful TUI built with Textual

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from https://ollama.ai
3. **FFmpeg** (for Whisper) - winget install ffmpeg

### Setup

    # Clone the repository
    git clone https://github.com/yourusername/voxcode.git
    cd voxcode

    # Create virtual environment
    python -m venv venv
    venv\Scripts/activate  # Windows

    # Install dependencies
    pip install -r requirements.txt

    # Pull an Ollama model
    ollama pull llama3.2

## Usage

### Terminal UI (Default)

    python main.py

### Single Command

    python main.py -c "open notepad and type hello world"

### Options

    -c, --command    Execute a single command
    --debug          Enable debug logging
    --model          Specify Ollama model (default: llama3.2)
    --check          Verify dependencies and exit

## Architecture

    voxcode/
    ├── config.py                     # Global configuration
    ├── main.py                       # Entry point
    ├── voice/
    │   ├── stt.py                    # Whisper STT
    │   └── tts.py                    # TTS
    ├── brain/
    │   ├── llm.py                    # LLM clients (Groq/Ollama)
    │   ├── planner.py                # Hierarchical state-based planner
    │   ├── api_registry.py           # API metadata lookup
    │   └── prompts.py                # Prompt templates
    ├── perception/
    │   ├── omniparser.py             # OmniParser integration wrapper
    │   └── screen_state.py           # Semantic state representation
    ├── agent/
    │   ├── eyes.py                   # Screen element perception
    │   ├── hands.py                  # QWEN action decision
    │   ├── verifier.py               # Pixel/state verification
    │   ├── pipeline.py               # Stateful reactive execution
    │   └── reactive_loop.py          # Feedback loop utilities
    ├── tools/
    │   └── execution.py              # Low-level action executor
    ├── memory/
    │   ├── episodic.py               # Episodic memory
    │   ├── working.py                # Working memory
    │   ├── persistent.py             # Vault persistence helpers
    │   └── vault/                    # Persistent profile/history/failures
    └── tui/
        └── app.py                    # Textual application

## Configuration

Environment variables:
- OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
- VOXCODE_MODEL - LLM model name (default: llama3.2)
- GROQ_API_KEY - Groq API key (required when LLM_PROVIDER=groq)
- LLM_PROVIDER - LLM provider: ollama or groq
- WHISPER_MODEL - Whisper model size (tiny/base/small/medium/large)
- VOXCODE_DEBUG - Enable debug mode
- VOXCODE_TRACE_ENABLED - Enable structured trace events (default: true)
- VOXCODE_TRACE_SCREENSHOTS - Save trace screenshots (default: true)
- VOXCODE_TRACE_SCREENSHOT_DIR - Base folder for per-run screenshots (default: screenshots/sessions)

## Debug Artifacts

- `voxcode.log` - Human-readable runtime log stream
- `audit_log.jsonl` - Structured event timeline with `session_id`, `sequence`, planner/pipeline/hands/verifier payloads
- `screenshots/sessions/<session_id>/` - Timestamped visual evidence captured before/after key actions

## Example Commands

- "Open Chrome and go to github.com"
- "Type an email saying I will be late to the meeting"
- "Take a screenshot"
- "Open File Explorer and create a new folder called Projects"
- "Close all Notepad windows"

## Safety

VOXCODE runs in **safe mode** by default:
- Destructive actions require confirmation
- Commands are logged
- Failsafe: move mouse to corner to abort

## License

MIT License
