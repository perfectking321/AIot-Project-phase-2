"""
VOXCODE Browser Agent
Uses browser-use library for reliable web automation.
This handles all browser-based tasks (YouTube, Google, websites, etc.)
"""

import asyncio
import logging
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("voxcode.browser_agent")


class GroqLLMWrapper:
    """
    Simple wrapper to make VOXCODE's Groq client compatible with browser-use.
    This allows us to reuse the existing Groq API key and configuration.
    """

    def __init__(self):
        from brain.llm import get_llm_client
        self.client = get_llm_client()

        # Required attributes for browser-use compatibility
        self.provider = "groq"
        self.model_name = "llama-3.3-70b-versatile"
        self.model = "llama-3.3-70b-versatile"  # Alternative attribute name
        self.temperature = 0.2

    def invoke(self, messages):
        """Convert browser-use format to our format."""
        if isinstance(messages, list):
            # Convert list of messages to single prompt
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    role = msg.get("role", "user")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                    else:
                        prompt_parts.append(str(content))
                else:
                    prompt_parts.append(str(msg))
            prompt = "\n".join(prompt_parts)
        else:
            prompt = str(messages)

        result = self.client.generate(prompt)
        return MockResponse(result.content)

    async def ainvoke(self, messages, output_format=None, **kwargs):
        """Async version of invoke for browser-use compatibility."""
        # Ignore extra parameters like output_format and session_id that browser-use might pass
        return self.invoke(messages)

    def predict(self, text):
        """Alternative method name that browser-use might use."""
        result = self.client.generate(text)
        return result.content

    def generate(self, messages, **kwargs):
        """Generate method for browser-use compatibility."""
        if isinstance(messages, list):
            # Convert list of messages to single prompt
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    role = msg.get("role", "user")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                    else:
                        prompt_parts.append(str(content))
                else:
                    prompt_parts.append(str(msg))
            prompt = "\n".join(prompt_parts)
        else:
            prompt = str(messages)

        result = self.client.generate(prompt)
        return MockResponse(result.content)

    async def agenerate(self, messages, **kwargs):
        """Async generate method for browser-use compatibility."""
        return self.generate(messages, **kwargs)

    async def apredict(self, text):
        """Async predict method."""
        result = self.client.generate(text)
        return result.content

    def bind(self, **kwargs):
        """Bind method for langchain compatibility."""
        return self

    def with_config(self, config):
        """With config method for langchain compatibility."""
        return self

    def __call__(self, prompt):
        """Make the object callable."""
        result = self.client.generate(prompt)
        return MockResponse(result.content)


class MockResponse:
    """Mock response object for browser-use compatibility."""

    def __init__(self, content):
        self.content = content
        self.text = content  # Alternative attribute name
        # Add usage attribute for browser-use compatibility
        self.usage = MockUsage()

    def __str__(self):
        return self.content

    @property
    def choices(self):
        """For OpenAI-style responses."""
        return [MockChoice(self.content)]


class MockUsage:
    """Mock usage object for API compatibility."""
    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 50
        self.total_tokens = 150


class MockChoice:
    """Mock choice object for OpenAI-style compatibility."""

    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message object for OpenAI-style compatibility."""

    def __init__(self, content):
        self.content = content


class BrowserTaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BrowserTaskResult:
    """Result from a browser task."""
    success: bool
    message: str
    data: Optional[dict] = None


class BrowserUseAgent:
    """
    Browser automation agent using browser-use library.

    This provides reliable browser control via:
    - Playwright for DOM access (not just screenshots)
    - LLM for task understanding and planning
    - Automatic element detection and interaction
    """

    # Keywords that indicate a browser task
    BROWSER_KEYWORDS = [
        "youtube", "google", "search", "browse", "website", "web",
        "chrome", "firefox", "edge", "browser", "url", "http",
        "video", "watch", "play video", "open site", "go to",
        "navigate to", "look up", "find online", "search for",
        "twitter", "facebook", "instagram", "reddit", "github",
        "amazon", "ebay", "netflix", "spotify", "new tab",
        "open tab", "tab"
    ]

    def __init__(
        self,
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None,
        headless: bool = False
    ):
        """
        Initialize browser agent.

        Args:
            on_status: Callback for status updates
            on_step: Callback for step updates (step_num, message, status)
            headless: Run browser in headless mode (default False for visibility)
        """
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)
        self.headless = headless
        self._stop_requested = False
        self._step_count = 0

        logger.info("BrowserUseAgent initialized")

    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        logger.info("Browser agent stop requested")

    @staticmethod
    def is_browser_task(command: str) -> bool:
        """
        Check if a command is a browser-related task.

        Args:
            command: The user's command

        Returns:
            True if this should be handled by browser agent
        """
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in BrowserUseAgent.BROWSER_KEYWORDS)

    async def execute_task(self, task: str) -> BrowserTaskResult:
        """
        Execute a browser task using browser-use.

        Args:
            task: Natural language task description

        Returns:
            BrowserTaskResult with success status and message
        """
        self._stop_requested = False
        self._step_count = 0

        logger.info(f"Browser task: {task}")
        self.on_status("Starting browser...")
        self.on_step(1, "Initializing browser-use agent", "running")

        try:
            # Import browser-use components
            from browser_use import Agent
            from config import config

            # Use the same LLM client as VOXCODE (Groq)
            llm = self._create_groq_llm()

            self.on_step(1, "Browser agent ready", "done")
            self._step_count = 2
            self.on_step(2, f"Executing: {task[:50]}...", "running")
            self.on_status("Executing browser task...")

            # Create and run the browser-use agent
            agent = Agent(
                task=task,
                llm=llm,
                use_vision=True,  # Use vision for better understanding
            )

            # Run the task
            result = await agent.run()

            self.on_step(self._step_count, "Task completed", "done")
            self.on_status("Browser task complete")

            return BrowserTaskResult(
                success=True,
                message=f"Successfully completed: {task}",
                data={"result": str(result)}
            )

        except ImportError as e:
            error_msg = str(e)
            if "browser_use" in error_msg or "browser-use" in error_msg:
                install_msg = "browser-use not installed. Run: pip install browser-use playwright && playwright install chromium"
                logger.error(install_msg)
                self.on_step(self._step_count or 1, install_msg, "failed")
                return BrowserTaskResult(
                    success=False,
                    message=install_msg
                )
            elif "langchain_anthropic" in error_msg:
                # Ignore this error since we're using Groq now
                pass
            else:
                logger.error(f"Import error: {e}")
                self.on_step(self._step_count or 1, f"Import error: {e}", "failed")
                return BrowserTaskResult(
                    success=False,
                    message=f"Import error: {e}"
                )

        except Exception as e:
            logger.error(f"Browser task failed: {e}", exc_info=True)
            self.on_step(self._step_count or 1, f"Error: {e}", "failed")
            self.on_status("Browser task failed")

            return BrowserTaskResult(
                success=False,
                message=f"Browser task failed: {e}"
            )

    def _create_groq_llm(self):
        """Create a Groq LLM wrapper for browser-use."""
        return GroqLLMWrapper()

    def _get_anthropic_key(self) -> str:
        """Get Anthropic API key from environment or config."""
        import os

        # Try environment variable first
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key

        # Try to read from a config file
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for line in f:
                        if line.startswith("ANTHROPIC_API_KEY="):
                            return line.split("=", 1)[1].strip()
        except:
            pass

        # Return empty - will fail gracefully
        logger.warning("No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable.")
        return ""

    def execute_sync(self, task: str) -> BrowserTaskResult:
        """
        Synchronous wrapper for execute_task.

        Args:
            task: Natural language task description

        Returns:
            BrowserTaskResult
        """
        return asyncio.run(self.execute_task(task))


# Convenience function for direct use
async def run_browser_task(
    task: str,
    on_status: Callable[[str], None] = None,
    on_step: Callable[[int, str, str], None] = None
) -> BrowserTaskResult:
    """
    Run a browser task using browser-use.

    Args:
        task: Natural language description of what to do
        on_status: Optional status callback
        on_step: Optional step callback

    Returns:
        BrowserTaskResult

    Example:
        result = await run_browser_task(
            "Go to YouTube and search for Taylor Swift, click the first video"
        )
    """
    agent = BrowserUseAgent(on_status=on_status, on_step=on_step)
    return await agent.execute_task(task)
