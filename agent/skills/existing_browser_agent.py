"""
VOXCODE Existing Browser Agent
Uses Playwright to connect to existing Chrome browser with remote debugging.
"""

import asyncio
import logging
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("voxcode.existing_browser_agent")


@dataclass
class BrowserTaskResult:
    """Result from a browser task."""
    success: bool
    message: str
    data: Optional[dict] = None


class ExistingBrowserAgent:
    """
    Browser automation agent that connects to your existing Chrome browser.
    Requires Chrome to be started with --remote-debugging-port=9222
    """

    def __init__(
        self,
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None
    ):
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)
        self._stop_requested = False
        self._step_count = 0

        logger.info("ExistingBrowserAgent initialized")

    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        logger.info("Existing browser agent stop requested")

    async def execute_task(self, task: str) -> BrowserTaskResult:
        """
        Execute a browser task using existing Chrome browser.

        Args:
            task: Natural language task description

        Returns:
            BrowserTaskResult with success status and message
        """
        self._stop_requested = False
        self._step_count = 0

        logger.info(f"Existing browser task: {task}")
        self.on_status("Connecting to existing Chrome...")
        self.on_step(1, "Looking for Chrome with remote debugging", "running")

        try:
            import requests
            from playwright.async_api import async_playwright

            # Check if Chrome is running with remote debugging
            chrome_port = None
            self.on_status("Checking for Chrome remote debugging...")

            for port in [9222, 9223, 9224, 9225]:
                try:
                    logger.info(f"Checking Chrome on port {port}")
                    response = requests.get(f"http://localhost:{port}/json/version", timeout=3)
                    if response.status_code == 200:
                        chrome_port = port
                        logger.info(f"Found Chrome on port {port}")
                        break
                except Exception as e:
                    logger.debug(f"Port {port} check failed: {e}")
                    continue

            if not chrome_port:
                self.on_step(1, "Chrome remote debugging not found", "failed")
                error_msg = ("Chrome not running with remote debugging. "
                           "Please close Chrome and run: chrome.exe --remote-debugging-port=9222")
                logger.error(error_msg)
                return BrowserTaskResult(
                    success=False,
                    message=error_msg
                )

            self.on_step(1, f"Found Chrome on port {chrome_port}", "done")
            self._step_count = 2
            self.on_step(2, "Connecting to your Chrome browser", "running")

            # Connect to existing Chrome with timeout
            playwright = await async_playwright().start()

            try:
                logger.info(f"Connecting to Chrome at http://localhost:{chrome_port}")

                # Connect to existing browser with timeout
                connect_task = playwright.chromium.connect_over_cdp(f"http://localhost:{chrome_port}")
                browser = await asyncio.wait_for(connect_task, timeout=10.0)

                # Get existing contexts and pages
                contexts = browser.contexts
                if not contexts:
                    logger.info("No contexts found, creating new one")
                    context = await browser.new_context()
                else:
                    context = contexts[0]  # Use first context
                    logger.info(f"Using existing context with {len(context.pages)} pages")

                pages = context.pages
                if not pages:
                    logger.info("No pages found, creating new one")
                    page = await context.new_page()
                elif "new tab" in task.lower():
                    logger.info("Creating new tab as requested")
                    page = await context.new_page()
                else:
                    page = pages[0]  # Use first page
                    logger.info("Using existing page")

                self.on_step(2, "Connected to Chrome browser", "done")
                self._step_count = 3
                self.on_step(3, f"Executing: {task[:50]}...", "running")

                # Parse and execute the task with timeout
                execute_task = self._execute_browser_action(page, task)
                await asyncio.wait_for(execute_task, timeout=30.0)

                self.on_step(3, "Task completed", "done")
                self.on_status("Browser task complete")

                return BrowserTaskResult(
                    success=True,
                    message=f"Successfully completed: {task}"
                )

            except asyncio.TimeoutError:
                error_msg = "Task timed out - Chrome may be unresponsive"
                logger.error(error_msg)
                self.on_step(self._step_count, error_msg, "failed")
                return BrowserTaskResult(success=False, message=error_msg)

            finally:
                try:
                    await playwright.stop()
                except:
                    pass

        except ImportError:
            msg = "Playwright not installed. Run: pip install playwright && playwright install chromium"
            logger.error(msg)
            self.on_step(self._step_count or 1, msg, "failed")
            return BrowserTaskResult(success=False, message=msg)

        except Exception as e:
            logger.error(f"Existing browser task failed: {e}", exc_info=True)
            self.on_step(self._step_count or 1, f"Error: {e}", "failed")
            return BrowserTaskResult(success=False, message=f"Task failed: {e}")

    async def _execute_browser_action(self, page, task: str):
        """Execute the browser action based on the task."""
        task_lower = task.lower()

        try:
            logger.info(f"Executing browser action for: {task}")

            # Handle different types of tasks
            if "new tab" in task_lower:
                logger.info("Handling new tab request")
                # New tab is handled in execute_task by creating new page
                if "google" in task_lower:
                    logger.info("Navigating to Google")
                    await page.goto("https://www.google.com", wait_until="networkidle")
                elif "youtube" in task_lower:
                    logger.info("Navigating to YouTube")
                    await page.goto("https://www.youtube.com", wait_until="networkidle")
                else:
                    logger.info("Opening blank tab")
                    await page.goto("about:blank")

            elif "youtube" in task_lower:
                logger.info("Handling YouTube request")
                await page.goto("https://www.youtube.com", wait_until="networkidle")

                if "search" in task_lower:
                    # Find search terms
                    search_term = self._extract_search_term(task)
                    if search_term:
                        logger.info(f"Searching YouTube for: {search_term}")
                        search_box = page.locator('input[name="search_query"]')
                        await search_box.click()
                        await search_box.fill(search_term)
                        await search_box.press("Enter")

                        # Wait for results
                        await page.wait_for_load_state("networkidle")

                        if "first video" in task_lower or "click first" in task_lower:
                            # Click first video result
                            first_video = page.locator('#contents ytd-video-renderer').first
                            await first_video.click()
                            await page.wait_for_load_state("networkidle")

                            if "fullscreen" in task_lower:
                                # Press 'f' for fullscreen
                                await page.keyboard.press("f")

            elif "google" in task_lower:
                logger.info("Handling Google request")
                await page.goto("https://www.google.com", wait_until="networkidle")

                if "search" in task_lower:
                    search_term = self._extract_search_term(task)
                    if search_term:
                        logger.info(f"Searching Google for: {search_term}")
                        search_box = page.locator('input[name="q"]')
                        await search_box.click()
                        await search_box.fill(search_term)
                        await search_box.press("Enter")
                        await page.wait_for_load_state("networkidle")

            elif any(site in task_lower for site in ["facebook", "twitter", "github"]):
                # Navigate to specific sites
                if "facebook" in task_lower:
                    logger.info("Navigating to Facebook")
                    await page.goto("https://www.facebook.com", wait_until="networkidle")
                elif "twitter" in task_lower:
                    logger.info("Navigating to Twitter")
                    await page.goto("https://www.twitter.com", wait_until="networkidle")
                elif "github" in task_lower:
                    logger.info("Navigating to GitHub")
                    await page.goto("https://www.github.com", wait_until="networkidle")

            else:
                # Generic case - just open a blank tab or navigate to Google
                logger.info("Generic browser action - opening Google")
                await page.goto("https://www.google.com", wait_until="networkidle")

            logger.info("Browser action completed successfully")

        except Exception as e:
            logger.error(f"Browser action failed: {e}")
            raise

    def _extract_search_term(self, task: str) -> str:
        """Extract search term from task description."""
        task_lower = task.lower()

        # Common patterns for search terms
        patterns = [
            "search for ",
            "search about ",
            "find ",
            "look for ",
            "look up ",
        ]

        for pattern in patterns:
            if pattern in task_lower:
                # Extract everything after the pattern
                start_idx = task_lower.find(pattern) + len(pattern)
                rest = task[start_idx:].strip()

                # Remove trailing instructions
                stop_words = [" and ", " then ", " click ", " open "]
                for stop in stop_words:
                    if stop in rest.lower():
                        rest = rest[:rest.lower().find(stop)]

                return rest.strip().strip('"').strip("'")

        # Fallback: look for quoted terms
        if '"' in task:
            parts = task.split('"')
            if len(parts) >= 3:
                return parts[1]

        return ""


# Convenience function
async def run_existing_browser_task(
    task: str,
    on_status: Callable[[str], None] = None,
    on_step: Callable[[int, str, str], None] = None
) -> BrowserTaskResult:
    """
    Run a browser task using existing Chrome browser.

    Example:
        result = await run_existing_browser_task(
            "Open a new tab and go to YouTube, search for cats"
        )
    """
    agent = ExistingBrowserAgent(on_status=on_status, on_step=on_step)
    return await agent.execute_task(task)