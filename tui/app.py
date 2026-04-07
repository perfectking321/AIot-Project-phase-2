"""
VOXCODE TUI - Clean Claude Code Style Interface
Ctrl+B toggle for voice recording with clean step-by-step output.
"""

import logging
import asyncio
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, RichLog
from textual.binding import Binding
from textual.reactive import reactive

logger = logging.getLogger("voxcode.tui")


class StatusBar(Static):
    """Status display with current state."""
    recording = reactive(False)
    status_text = reactive("Ready")

    def render(self) -> str:
        if self.recording:
            return "[bold white on red] 🎤 RECORDING [/] Press Ctrl+B to stop"
        return f"[bold cyan]● {self.status_text}[/]"


class VoxcodeApp(App):
    """VOXCODE TUI with clean Claude Code style interface."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 2 1fr auto;
    }
    #status {
        height: 2;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $primary;
    }
    #log {
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    # Remove ctrl+b from Textual bindings - we use global hotkey instead
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def __init__(self):
        super().__init__()
        self.recorder = None
        self.stt = None
        self.vision = None
        self.is_recording = False
        self.global_hotkey = None
        self._toggle_in_progress = False  # Prevent re-entry
        self._current_agent = None  # Track current agent for cleanup

    def _init_voice(self):
        """Initialize voice components."""
        try:
            from voice.stt import AudioRecorder, SpeechToText
            self.recorder = AudioRecorder()
            self.stt = SpeechToText(preload=True)
            logger.info("Voice components initialized")
        except Exception as e:
            logger.error(f"Failed to init voice: {e}")

    def _init_vision(self):
        """Initialize vision."""
        try:
            from agent.vision import ScreenVision
            self.vision = ScreenVision(preload=True)
            logger.info("Vision initialized")
        except Exception as e:
            logger.error(f"Failed to init vision: {e}")

    def _init_global_hotkey(self):
        """Initialize global hotkey listener."""
        try:
            from voice.hotkey import GlobalHotkey
            import time

            self._last_hotkey_time = 0.0

            def on_hotkey_press():
                """Called when Ctrl+B is pressed globally."""
                # Extra debounce at TUI level
                now = time.time()
                if now - self._last_hotkey_time < 0.5:
                    logger.warning(f"TUI debounce: ignoring hotkey (too fast)")
                    return
                self._last_hotkey_time = now

                logger.info(f"Hotkey pressed! Current state: is_recording={self.is_recording}")
                # Schedule toggle on main thread
                self.call_from_thread(self._do_toggle)

            self.global_hotkey = GlobalHotkey(on_hotkey_press, hotkey="ctrl+b")
            self.global_hotkey.start()
            logger.info("Global hotkey started")
        except Exception as e:
            logger.error(f"Failed to init global hotkey: {e}")

    def _do_toggle(self):
        """Actually toggle recording state - called on main thread."""
        # Prevent re-entry from multiple hotkey events
        if self._toggle_in_progress:
            logger.warning("Toggle already in progress, ignoring")
            return

        self._toggle_in_progress = True
        logger.info(f"_do_toggle: is_recording={self.is_recording}")

        try:
            if not self.is_recording:
                # START recording
                if not self.recorder:
                    self.log_error("Voice system not available")
                    return

                self.is_recording = True
                logger.info("Setting is_recording = True")

                # Update UI
                try:
                    status = self.query_one("#status", StatusBar)
                    status.recording = True
                except Exception as e:
                    logger.error(f"UI update error: {e}")

                self.log_status("Recording... speak now")

                # Start audio capture
                try:
                    self.recorder.start_recording()
                    logger.info("recorder.start_recording() called")
                except Exception as e:
                    logger.error(f"Start recording error: {e}")
                    self.is_recording = False
                    try:
                        status = self.query_one("#status", StatusBar)
                        status.recording = False
                    except:
                        pass

            else:
                # STOP recording
                logger.info("Stopping recording")
                self.is_recording = False

                # Update UI
                try:
                    status = self.query_one("#status", StatusBar)
                    status.recording = False
                except Exception as e:
                    logger.error(f"UI update error: {e}")

                self.log_status("Processing...")

                # Get audio
                try:
                    audio_data = self.recorder.stop_recording()
                    logger.info(f"Got {len(audio_data)} bytes")

                    if audio_data and len(audio_data) > 5000:
                        self.run_worker(self._process_audio(audio_data))
                    else:
                        self.log_warning("Recording too short")
                        self.set_status("Ready")
                except Exception as e:
                    logger.error(f"Stop recording error: {e}")
                    self.log_error(f"Error: {e}")
                    self.set_status("Ready")

            logger.info(f"_do_toggle done: is_recording={self.is_recording}")
        finally:
            self._toggle_in_progress = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status")
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "VOXCODE"
        self.sub_title = "Voice-Controlled Automation"

        self.log_system("━" * 50)
        self.log_system("  VOXCODE - Voice Controlled Windows Automation")
        self.log_system("━" * 50)
        self.log_info("")
        self.log_status("Loading models...")

        await asyncio.to_thread(self._init_voice)
        await asyncio.to_thread(self._init_vision)
        self._init_global_hotkey()

        self.log_info("")
        if self.recorder:
            self.log_success("✓ Voice system ready")
        if self.vision:
            self.log_success("✓ Vision system ready")
        if self.global_hotkey:
            self.log_success("✓ Global hotkey active (Ctrl+B)")

        self.log_info("")
        self.log_info("[bold]Press Ctrl+B[/] to start/stop recording")
        self.log_info("")
        self.set_status("Ready")

    def on_unmount(self) -> None:
        """Cleanup when app exits."""
        # Stop the global hotkey
        if self.global_hotkey:
            self.global_hotkey.stop()

        # Stop any running agent
        if self._current_agent:
            self._current_agent.stop()
            logger.info("Agent stopped on exit")

        # Cleanup recorder
        if self.recorder:
            try:
                self.recorder.cleanup()
            except:
                pass

    # ==================== CLEAN LOGGING ====================

    def _log(self, msg: str, style: str = "white"):
        """Log with style."""
        try:
            log_widget = self.query_one("#log", RichLog)
            log_widget.write(f"[{style}]{msg}[/]")
        except:
            pass

    def log_info(self, msg: str):
        self._log(msg, "white")

    def log_success(self, msg: str):
        self._log(msg, "green")

    def log_error(self, msg: str):
        self._log(f"✗ {msg}", "red")

    def log_warning(self, msg: str):
        self._log(f"⚠ {msg}", "yellow")

    def log_system(self, msg: str):
        self._log(msg, "cyan")

    def log_status(self, msg: str):
        self._log(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/] {msg}", "bold blue")

    def log_user(self, msg: str):
        self._log(f"\n[bold yellow]You:[/] {msg}", "yellow")

    def log_step(self, step_num: int, total: int, msg: str, status: str = "running"):
        """Log a step in Claude Code style."""
        icons = {"running": "○", "done": "●", "failed": "✗"}
        icon = icons.get(status, "○")
        color = {"running": "blue", "done": "green", "failed": "red"}.get(status, "blue")
        self._log(f"  [{color}]{icon}[/] Step {step_num}/{total}: {msg}", color)

    def set_status(self, text: str):
        try:
            status = self.query_one("#status", StatusBar)
            status.status_text = text
        except:
            pass

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self.log_system("Log cleared")

    async def _process_audio(self, audio_data: bytes) -> None:
        """Process recorded audio."""
        try:
            self.log_status("Transcribing...")
            result = await asyncio.to_thread(self.stt.transcribe, audio_data)

            if result.text:
                self.log_user(result.text)
                await self._execute_command(result.text)
            else:
                self.log_warning("No speech detected")
                self.set_status("Ready")

        except Exception as e:
            self.log_error(f"Transcription failed: {e}")
            self.set_status("Ready")

    async def _execute_command(self, command: str) -> None:
        """Execute command - routes to Browser-Use or Fast Agent based on task type."""
        self.set_status(f"Analyzing command...")
        self.log_info("")

        # Check if this is a browser task
        from agent.skills.normal_chrome_agent import NormalChromeAgent
        from agent.skills.browser_agent import BrowserUseAgent

        if BrowserUseAgent.is_browser_task(command):
            await self._execute_browser_command(command)
        else:
            await self._execute_desktop_command(command)

    async def _execute_browser_command(self, command: str) -> None:
        """Execute browser task using normal Chrome browser (no special setup required)."""
        self.log_status("Browser task detected → Using normal Chrome")
        self.log_info("[dim]Controlling your existing Chrome browser with Windows automation[/]")
        self.log_info("[dim]No special setup required - works with normal Chrome[/]")
        self.log_info("")

        try:
            from agent.skills.normal_chrome_agent import NormalChromeAgent
            from agent.tools import WindowsTools

            def on_status(msg: str):
                self.set_status(msg)

            def on_step(step_num: int, msg: str, status: str):
                self.log_agent_step(step_num, msg, status)

            # Create tools instance
            tools = WindowsTools(vision_instance=self.vision)

            # Create Normal Chrome agent
            agent = NormalChromeAgent(
                on_status=on_status,
                on_step=on_step,
                tools=tools
            )

            # Store reference for cleanup
            self._current_agent = agent

            # Execute the browser task
            result = await agent.execute_task(command)

            # Clear agent reference
            self._current_agent = None

            # Show result
            self.log_info("")
            if result.success:
                self.log_success(f"✓ {result.message}")
            else:
                self.log_warning(f"⚠ {result.message}")

        except Exception as e:
            self.log_error(f"Browser execution failed: {e}")
            logger.error(f"Browser execution error: {e}", exc_info=True)

        self._current_agent = None
        self.set_status("Ready")

    async def _execute_desktop_command(self, command: str) -> None:
        """Execute desktop task using Fast Visual Agent (Screenshot + Groq)."""
        self.set_status(f"Thinking...")
        self.log_status("Desktop task → Using Fast Visual Agent")
        self.log_info("[dim]Screenshot → Analyze → Act → Repeat[/]")
        self.log_info("")

        try:
            from agent.fast_agent import FastVisualAgent
            from agent.tools import WindowsTools

            # Create tools instance
            tools = WindowsTools(vision_instance=self.vision)

            # Step counter for display
            self._current_step = 0

            def on_status(msg: str):
                self.set_status(msg)

            def on_step(step_num: int, msg: str, status: str):
                self._current_step = step_num
                self.log_agent_step(step_num, msg, status)

            # Create Fast Agent (OmniParser + Groq = fast!)
            agent = FastVisualAgent(
                vision=self.vision,
                tools=tools,
                on_status=on_status,
                on_step=on_step
            )

            # Store reference for cleanup on exit
            self._current_agent = agent

            # Execute
            result = await asyncio.to_thread(agent.process_command, command)

            # Clear agent reference
            self._current_agent = None

            # Show result
            self.log_info("")
            if "Successfully" in result:
                self.log_success(f"✓ {result}")
            else:
                self.log_warning(f"⚠ {result}")

        except Exception as e:
            self.log_error(f"Execution failed: {e}")
            logger.error(f"Execution error: {e}", exc_info=True)

        self._current_agent = None
        self.set_status("Ready")

    def log_agent_step(self, step_num: int, msg: str, status: str = "running"):
        """Log an autonomous agent step."""
        icons = {"running": "◌", "done": "●", "failed": "✗", "thinking": "◐"}
        icon = icons.get(status, "○")
        colors = {"running": "blue", "done": "green", "failed": "red", "thinking": "yellow"}
        color = colors.get(status, "blue")
        self._log(f"  [{color}]{icon}[/] Cycle {step_num}: {msg}", color)


def run_app():
    """Run the VOXCODE TUI."""
    # Suppress verbose logging to console
    logging.getLogger("voxcode.omniparser").setLevel(logging.WARNING)
    logging.getLogger("voxcode.vision").setLevel(logging.WARNING)
    logging.getLogger("voxcode.tools").setLevel(logging.WARNING)
    logging.getLogger("voxcode.planner").setLevel(logging.WARNING)

    app = VoxcodeApp()
    app.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("voxcode.log", encoding='utf-8')]
    )
    run_app()
