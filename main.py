#!/usr/bin/env python3
"""
VOXCODE - Voice-Controlled Windows Automation Assistant
Main entry point with comprehensive logging.
"""

import sys
import argparse
import logging
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass


def setup_logging(debug: bool = False):
    """Configure logging - file only for clean TUI."""
    level = logging.DEBUG if debug else logging.INFO

    # File handler only - TUI handles console output
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler("voxcode.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Root logger - file only
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Suppress verbose loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("voxcode.omniparser").setLevel(logging.WARNING)
    logging.getLogger("voxcode.vision").setLevel(logging.WARNING)

    return logging.getLogger("voxcode")


def check_dependencies():
    """Check if required packages are available."""
    logger = logging.getLogger("voxcode.startup")
    missing = []

    logger.info("Checking dependencies...")

    try:
        import textual
        logger.debug("[OK] textual")
    except ImportError:
        missing.append("textual")
        logger.error("[MISSING] textual")

    try:
        import pyaudio
        logger.debug("[OK] pyaudio")
    except ImportError:
        missing.append("pyaudio")
        logger.error("[MISSING] pyaudio")

    try:
        import whisper
        logger.debug("[OK] whisper")
    except ImportError:
        missing.append("openai-whisper")
        logger.error("[MISSING] openai-whisper")

    try:
        import pyautogui
        logger.debug("[OK] pyautogui")
    except ImportError:
        missing.append("pyautogui")
        logger.error("[MISSING] pyautogui")

    try:
        import requests
        logger.debug("[OK] requests")
    except ImportError:
        missing.append("requests")
        logger.error("[MISSING] requests")

    if missing:
        logger.error(f"Missing: {', '.join(missing)}")
        print(f"\n[X] Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    logger.info("All dependencies OK")
    return True


def check_ollama():
    """Check if LLM provider (Ollama or Groq) is available."""
    logger = logging.getLogger("voxcode.startup")
    logger.info("Checking LLM connection...")

    try:
        from brain.llm import get_llm_client
        from config import config

        client = get_llm_client()
        provider = config.llm.provider

        if client.is_available():
            logger.info(f"[OK] LLM connected ({provider}: {config.llm.model_name})")
            print(f"\n[OK] LLM: {provider} - {config.llm.model_name}")
            return True
        else:
            logger.warning(f"[X] {provider} not responding")
            if provider == "ollama":
                print("\n[!] Ollama not available. Make sure it's running:")
                print("    ollama serve")
            else:
                print(f"\n[!] {provider} not available. Check GROQ_API_KEY environment variable")
            return False
    except Exception as e:
        logger.error(f"LLM check failed: {e}")
        return False


def run_tui():
    """Run the TUI application."""
    logger = logging.getLogger("voxcode")
    logger.info("Starting TUI...")

    from tui.app import VoxcodeApp
    app = VoxcodeApp()
    app.run()


def run_cli(command: str):
    """Run a single command in CLI mode using stateful planning + pipeline."""
    logger = logging.getLogger("voxcode")
    logger.info(f"CLI mode: {command}")

    def on_message(msg):
        print(f"  -> {msg}")
        logger.info(f"Agent: {msg}")

    from brain.planner import get_planner
    from agent.pipeline import get_pipeline

    planner = get_planner()
    plan = planner.create_plan(command, "CLI mode - no explicit screen context")
    pipeline = get_pipeline(on_status=on_message)

    result = pipeline.run_task_plan(plan, on_status=on_message)

    print(f"\n[OK] Result: {result}")
    logger.info(f"Result: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="VOXCODE - Voice-Controlled Windows Automation"
    )
    parser.add_argument("-c", "--command", help="Execute a single command")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    # Setup logging first
    logger = setup_logging(debug=args.debug)
    logger.info("=" * 50)
    logger.info("VOXCODE Starting")
    logger.info("=" * 50)

    if args.check:
        deps_ok = check_dependencies()
        ollama_ok = check_ollama()
        if deps_ok and ollama_ok:
            print("\n[OK] All checks passed!")
            sys.exit(0)
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check Ollama (warning only, don't exit)
    check_ollama()

    if args.command:
        run_cli(args.command)
    else:
        run_tui()


if __name__ == "__main__":
    main()
