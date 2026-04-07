# VOXCODE

**Voice-Controlled Windows Automation Assistant**

VOXCODE is a local, privacy-first voice assistant that automates Windows tasks using natural language commands.

## Features

- **Push-to-Talk Voice Input** - Whisper-powered speech recognition
- **Text-to-Speech** - Audio feedback using pyttsx3
- **Local LLM** - Powered by Ollama (Llama, Mistral, etc.)
- **Windows Automation** - Control apps, type, click, and more
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
    venv\Scriptsctivate  # Windows

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
    ├── config.py           # Global configuration
    ├── main.py             # Entry point
    ├── requirements.txt    # Dependencies
    ├── voice/              # Voice I/O module
    │   ├── stt.py          # Speech-to-text (Whisper)
    │   └── tts.py          # Text-to-speech (pyttsx3)
    ├── brain/              # LLM module
    │   ├── llm.py          # Ollama client
    │   └── prompts.py      # System prompts
    ├── agent/              # Automation module
    │   ├── tools.py        # Windows automation tools
    │   ├── planner.py      # Task planning
    │   └── loop.py         # Agent execution loop
    └── tui/                # Terminal UI
        └── app.py          # Textual application

## Configuration

Environment variables:
- OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
- VOXCODE_MODEL - LLM model name (default: llama3.2)
- WHISPER_MODEL - Whisper model size (tiny/base/small/medium/large)
- VOXCODE_DEBUG - Enable debug mode

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
