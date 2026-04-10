#!/usr/bin/env python3
"""
VOXCODE - Voice-Controlled Windows Automation Assistant
Main entry point with comprehensive logging.
"""

import sys
import argparse
import logging
import os
import threading
from queue import Empty, Queue
from logging.handlers import RotatingFileHandler

from config import config

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
    level = logging.DEBUG
    log_file = config.log_file or "voxcode.log"

    # File handler only - TUI handles console output
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=8 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Root logger - file only
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Ensure clean handler setup if logging is initialized multiple times
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
        try:
            existing.close()
        except Exception:
            pass

    root_logger.addHandler(file_handler)

    # Suppress only noisy network loggers; keep VoxCode components visible
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("voxcode.omniparser").setLevel(logging.INFO)
    logging.getLogger("voxcode.vision").setLevel(logging.INFO)

    # Trace the logger initialization itself for deterministic debugging.
    try:
        from agent.trace import get_trace_logger

        trace = get_trace_logger()
        trace.log_event(
            source="main",
            event_type="logging_initialized",
            payload={"debug": debug, "level": logging.getLevelName(level), "log_file": log_file},
        )
    except Exception:
        # Logging must remain robust even if trace initialization fails.
        pass

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
        import pyaudio  # type: ignore
        logger.debug("[OK] pyaudio")
    except ImportError:
        try:
            import pyaudiowpatch as pyaudio  # type: ignore
            logger.debug("[OK] pyaudiowpatch")
        except ImportError:
            missing.append("pyaudio/pyaudiowpatch")
            logger.error("[MISSING] pyaudio and pyaudiowpatch")

    try:
        from faster_whisper import WhisperModel  # noqa: F401
        logger.debug("[OK] faster-whisper")
    except ImportError:
        try:
            import whisper  # noqa: F401
            logger.warning("[WARN] Using openai-whisper (slower). Install faster-whisper for better speed.")
        except ImportError:
            missing.append("faster-whisper or openai-whisper")
            logger.error("[MISSING] faster-whisper and openai-whisper")

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


def check_models():
    """Verify all tiered role models are pulled in Ollama."""
    logger = logging.getLogger("voxcode.startup")

    if config.llm.provider != "ollama":
        return  # Only relevant for Ollama

    required_models = {
        "planner": config.llm.planner_model,
        "executor": config.llm.executor_model,
        "verifier": config.llm.verifier_model,
        "fast": config.llm.fast_model,
    }

    try:
        import requests
        resp = requests.get(f"{config.llm.ollama_host}/api/tags", timeout=5)
        if resp.status_code != 200:
            logger.warning("Could not query Ollama models")
            return

        pulled_models = set()
        for m in resp.json().get("models", []):
            name = m.get("name", "")
            pulled_models.add(name)
            # Also add without tag for matching
            base = name.split(":")[0] if ":" in name else name
            pulled_models.add(base)

        missing = []
        for role, model in required_models.items():
            # Check both exact name and base name
            base = model.split(":")[0] if ":" in model else model
            if model not in pulled_models and base not in pulled_models:
                missing.append((role, model))

        if missing:
            print("\n[!] Missing tiered models:")
            for role, model in missing:
                print(f"    {role}: {model}  →  ollama pull {model}")
            logger.warning(f"Missing models: {missing}")
        else:
            unique_models = set(required_models.values())
            logger.info(f"[OK] All tiered models available: {unique_models}")
            print(f"[OK] Tiered models: {', '.join(sorted(unique_models))}")

    except Exception as e:
        logger.warning(f"Could not verify models: {e}")


def check_admin():
    """Check and display admin privilege status."""
    logger = logging.getLogger("voxcode.startup")
    try:
        from agent.skills.system_skills import IS_ADMIN
        if IS_ADMIN:
            print("[OK] Running with Administrator privileges ✓")
            logger.info("Running with Administrator privileges")
        else:
            print("[!] Running WITHOUT Administrator privileges")
            print("    Some system skills (bluetooth, services) may be limited.")
            print("    To enable full access: right-click → Run as administrator")
            logger.warning("Running without Administrator privileges")
    except Exception as e:
        logger.debug(f"Admin check skipped: {e}")


def run_tui():
    """Run the TUI application."""
    logger = logging.getLogger("voxcode")
    logger.info("Starting TUI...")

    from agent.dispatcher import get_dispatcher
    from tui.app import VoxcodeApp

    dispatcher = get_dispatcher(
        on_message=lambda message: logger.info(f"Dispatcher: {message}"),
        on_state_change=lambda state: logger.debug(f"Dispatcher state: {state}"),
    )
    dispatcher.start()

    command_queue: Queue = Queue()
    stop_event = threading.Event()

    def _queue_worker():
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.2)
            except Empty:
                continue

            if command is None:
                break
            dispatcher.submit(command)

    queue_thread = threading.Thread(target=_queue_worker, daemon=True, name="ptt-command-queue")
    queue_thread.start()

    ptt_listener = None
    if config.voice.enable_ptt_listener:
        try:
            from voice.ptt_listener import PTTListener

            ptt_listener = PTTListener(
                on_command=lambda text: command_queue.put(text),
                hotkey=config.voice.ptt_hotkey,
            )
            ptt_listener.start()
            logger.info("PTT listener started for run_tui")
        except Exception as exc:
            logger.info(f"PTT listener unavailable: {exc}")

    try:
        app = VoxcodeApp(dispatcher=dispatcher)
        app.run()
    finally:
        if ptt_listener:
            ptt_listener.stop()
        stop_event.set()
        command_queue.put(None)
        queue_thread.join(timeout=2.0)
        dispatcher.stop()


def run_cli(command: str):
    """Run a single command in CLI mode via the unified dispatcher."""
    logger = logging.getLogger("voxcode")
    logger.info(f"CLI mode: {command}")

    def on_message(msg):
        print(f"  -> {msg}")
        logger.info(f"Agent: {msg}")

    from agent.dispatcher import get_dispatcher

    dispatcher = get_dispatcher(on_message=on_message)
    result = dispatcher.dispatch(command)

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

    try:
        from agent.system_context import SystemContextProvider
        SystemContextProvider().start()
    except Exception as e:
        logger.error(f"Failed to start SystemContextProvider: {e}")

    try:
        from agent.trace import get_trace_logger

        get_trace_logger().log_event(
            source="main",
            event_type="startup",
            payload={
                "debug": args.debug,
                "mode": "cli" if bool(args.command) else "tui",
                "check_only": args.check,
            },
        )
    except Exception:
        pass

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

    # Verify tiered models are available
    check_models()

    # Show privilege level
    check_admin()

    if args.command:
        run_cli(args.command)
    else:
        run_tui()


if __name__ == "__main__":
    main()
