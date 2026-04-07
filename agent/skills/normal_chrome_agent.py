"""
VOXCODE Normal Chrome Agent
Uses Windows automation to control normal Chrome browser without remote debugging.
Works with your existing Chrome browser just like desktop automation.
"""

import asyncio
import logging
import time
from typing import Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger("voxcode.normal_chrome_agent")


@dataclass
class BrowserTaskResult:
    """Result from a browser task."""
    success: bool
    message: str
    data: Optional[dict] = None


class NormalChromeAgent:
    """
    Browser automation agent that works with normal Chrome using Windows automation.
    No special setup required - works with any running Chrome browser.
    """

    def __init__(
        self,
        on_status: Callable[[str], None] = None,
        on_step: Callable[[int, str, str], None] = None,
        tools=None
    ):
        self.on_status = on_status or (lambda x: None)
        self.on_step = on_step or (lambda n, msg, status: None)
        self.tools = tools
        self._stop_requested = False
        self._step_count = 0

        logger.info("NormalChromeAgent initialized")

    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
        logger.info("Normal Chrome agent stop requested")

    async def execute_task(self, task: str) -> BrowserTaskResult:
        """
        Execute a browser task using normal Chrome with Windows automation.

        Args:
            task: Natural language task description

        Returns:
            BrowserTaskResult with success status and message
        """
        self._stop_requested = False
        self._step_count = 0

        logger.info(f"Normal Chrome task: {task}")
        self.on_status("Looking for Chrome browser...")
        self.on_step(1, "Finding Chrome window", "running")

        if not self.tools:
            return BrowserTaskResult(
                success=False,
                message="No tools available for browser automation"
            )

        try:
            # Check if Chrome is already running
            chrome_found = await self._find_or_open_chrome()

            if not chrome_found:
                self.on_step(1, "Chrome not found, opening it", "running")
                await self._open_chrome()
                time.sleep(2)  # Wait for Chrome to load

            self.on_step(1, "Chrome browser ready", "done")
            self._step_count = 2
            self.on_step(2, f"Executing: {task[:50]}...", "running")

            # Parse and execute the task
            success = await self._execute_browser_action(task)

            if success:
                self.on_step(2, "Task completed", "done")
                self.on_status("Browser task complete")
                return BrowserTaskResult(
                    success=True,
                    message=f"Successfully completed: {task}"
                )
            else:
                self.on_step(2, "Task failed", "failed")
                return BrowserTaskResult(
                    success=False,
                    message="Browser task could not be completed"
                )

        except Exception as e:
            logger.error(f"Normal Chrome task failed: {e}", exc_info=True)
            self.on_step(self._step_count or 1, f"Error: {e}", "failed")
            return BrowserTaskResult(success=False, message=f"Task failed: {e}")

    async def _find_or_open_chrome(self) -> bool:
        """Find existing Chrome window or return False if not found."""
        try:
            import pygetwindow as gw

            # Look for Chrome windows
            chrome_windows = []
            for window in gw.getAllWindows():
                if window.title and ("chrome" in window.title.lower() or "google chrome" in window.title.lower()):
                    chrome_windows.append(window)

            if chrome_windows:
                # Focus on the first Chrome window
                chrome_windows[0].activate()
                logger.info(f"Found and focused Chrome window: {chrome_windows[0].title}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error finding Chrome: {e}")
            return False

    async def _open_chrome(self):
        """Open Chrome browser."""
        try:
            # Use tools to open Chrome
            result = self.tools.open_application("chrome")
            if result.success:
                logger.info("Chrome opened successfully")
                # Wait for Chrome to fully load
                time.sleep(3)
            else:
                logger.error(f"Failed to open Chrome: {result}")
        except Exception as e:
            logger.error(f"Error opening Chrome: {e}")

    async def _execute_browser_action(self, task: str) -> bool:
        """Execute the browser action based on the task with intelligent task parsing."""
        task_lower = task.lower()

        try:
            logger.info(f"Executing browser action for: {task}")

            # Parse the task to understand what the user wants
            wants_new_tab = "new tab" in task_lower
            wants_youtube = "youtube" in task_lower
            wants_google = "google" in task_lower and not wants_youtube
            wants_search = "search" in task_lower or "find" in task_lower or "look for" in task_lower

            # Extract search terms
            search_term = self._extract_search_term(task) if wants_search else None

            logger.info(f"Task analysis: new_tab={wants_new_tab}, youtube={wants_youtube}, google={wants_google}, search={wants_search}, search_term='{search_term}'")

            # Step 1: Handle new tab if requested
            if wants_new_tab:
                logger.info("Opening new tab")
                result = self.tools.hotkey(keys=["ctrl", "t"])
                if not result.success:
                    logger.error("Failed to open new tab")
                    return False
                time.sleep(1)

            # Step 2: Navigate to appropriate site
            if wants_youtube:
                logger.info("Navigating to YouTube")
                success = await self._navigate_to_url("youtube.com")
                if not success:
                    return False

                # If we want to search on YouTube, do it
                if wants_search and search_term:
                    logger.info(f"Performing YouTube search for: {search_term}")
                    await self._youtube_search(search_term, task_lower)

            elif wants_google:
                logger.info("Navigating to Google")
                success = await self._navigate_to_url("google.com")
                if not success:
                    return False

                # If we want to search on Google, do it
                if wants_search and search_term:
                    logger.info(f"Performing Google search for: {search_term}")
                    await self._google_search(search_term)

            elif wants_search and search_term and not wants_youtube and not wants_google:
                # User wants to search but didn't specify where - default to current page or YouTube for videos
                if "video" in task_lower or "cat" in task_lower or "movie" in task_lower:
                    logger.info("Search term suggests videos - going to YouTube")
                    success = await self._navigate_to_url("youtube.com")
                    if success:
                        await self._youtube_search(search_term, task_lower)
                else:
                    logger.info("General search - going to Google")
                    success = await self._navigate_to_url("google.com")
                    if success:
                        await self._google_search(search_term)

            elif any(site in task_lower for site in ["facebook", "twitter", "github", "netflix", "amazon"]):
                # Navigate to specific sites
                if "facebook" in task_lower:
                    await self._navigate_to_url("facebook.com")
                elif "twitter" in task_lower:
                    await self._navigate_to_url("twitter.com")
                elif "github" in task_lower:
                    await self._navigate_to_url("github.com")
                elif "netflix" in task_lower:
                    await self._navigate_to_url("netflix.com")
                elif "amazon" in task_lower:
                    await self._navigate_to_url("amazon.com")

            elif wants_new_tab and not wants_youtube and not wants_google:
                # Just opened new tab, go to a default useful page
                logger.info("New tab opened, going to Google as default")
                await self._navigate_to_url("google.com")

            else:
                # Default case - try to be intelligent about what user wants
                if search_term:
                    # User has search term but no specific site - use Google
                    logger.info("Default search action - going to Google")
                    success = await self._navigate_to_url("google.com")
                    if success:
                        await self._google_search(search_term)
                else:
                    # No clear intent - go to Google
                    logger.info("Default action - navigating to Google")
                    await self._navigate_to_url("google.com")

            logger.info("Browser action completed successfully")
            return True

        except Exception as e:
            logger.error(f"Browser action failed: {e}")
            return False

    async def _navigate_to_url(self, url: str):
        """Navigate to a URL using address bar."""
        try:
            # Focus address bar (Ctrl+L)
            result = self.tools.hotkey(keys=["ctrl", "l"])
            if not result.success:
                logger.error("Failed to focus address bar")
                return False

            time.sleep(0.5)

            # Type the URL
            result = self.tools.type_text(text=url)
            if not result.success:
                logger.error(f"Failed to type URL: {url}")
                return False

            time.sleep(0.3)

            # Press Enter
            result = self.tools.press_key(key="enter")
            if not result.success:
                logger.error("Failed to press Enter")
                return False

            # Wait for page to load
            time.sleep(2)
            logger.info(f"Navigated to {url}")
            return True

        except Exception as e:
            logger.error(f"Navigation to {url} failed: {e}")
            return False

    async def _youtube_search(self, search_term: str, task_lower: str):
        """Search on YouTube with multiple fallback methods."""
        try:
            # Wait for YouTube to load
            time.sleep(3)

            logger.info(f"Attempting YouTube search for: {search_term}")

            # Method 1: Try clicking search box directly with vision
            search_clicked = False
            if self.tools and hasattr(self.tools, 'click_text'):
                # Try different search box indicators
                search_attempts = [
                    "Search",
                    "search",
                    "Search YouTube",
                ]

                for search_text in search_attempts:
                    logger.info(f"Trying to click: {search_text}")
                    result = self.tools.click_text(text=search_text)
                    if result.success:
                        logger.info(f"Successfully clicked search using: {search_text}")
                        search_clicked = True
                        time.sleep(0.5)
                        break

            # Method 2: Use keyboard shortcut to focus search (/)
            if not search_clicked:
                logger.info("Trying keyboard shortcut to focus search")
                self.tools.press_key(key="/")
                time.sleep(0.5)
                search_clicked = True

            # Method 3: Use Tab navigation as fallback
            if not search_clicked:
                logger.info("Using Tab navigation to find search box")
                for i in range(5):  # Try up to 5 tabs
                    self.tools.press_key(key="tab")
                    time.sleep(0.3)

            # Clear any existing text and type search term
            logger.info(f"Typing search term: {search_term}")

            # Clear existing content
            self.tools.hotkey(keys=["ctrl", "a"])  # Select all
            time.sleep(0.2)

            # Type search term
            self.tools.type_text(text=search_term)
            time.sleep(0.8)

            # Press Enter to search
            logger.info("Executing search")
            self.tools.press_key(key="enter")
            time.sleep(4)  # Wait for search results

            # Handle clicking first video if requested
            if "first video" in task_lower or "click first" in task_lower or "play first" in task_lower:
                logger.info("Attempting to click first video")
                await self._click_first_youtube_video()

                # Handle fullscreen request
                if "fullscreen" in task_lower or "full screen" in task_lower:
                    logger.info("Enabling fullscreen")
                    time.sleep(2)  # Wait for video to start
                    self.tools.press_key(key="f")

            logger.info(f"YouTube search completed for: {search_term}")

        except Exception as e:
            logger.error(f"YouTube search failed: {e}")

    async def _click_first_youtube_video(self):
        """Click the first video result with multiple methods."""
        try:
            clicked = False

            # Method 1: Try clicking common video elements
            video_click_attempts = [
                "thumbnail",
                "video",
                "play",
            ]

            for attempt in video_click_attempts:
                logger.info(f"Trying to click first video using: {attempt}")
                result = self.tools.click_text(text=attempt)
                if result.success:
                    logger.info(f"Successfully clicked video using: {attempt}")
                    clicked = True
                    break
                time.sleep(0.5)

            # Method 2: Use coordinates if available (fallback)
            if not clicked:
                logger.info("Using coordinate-based click for first video")
                # Click in the general area where first video usually appears
                # This is approximate but often works
                import pyautogui
                screen_width, screen_height = pyautogui.size()

                # Approximate position of first video (left side, upper area)
                x = int(screen_width * 0.25)  # 25% from left
                y = int(screen_height * 0.35)  # 35% from top

                result = self.tools.click(x=x, y=y)
                if result.success:
                    logger.info("Clicked first video using coordinates")
                    clicked = True

            if not clicked:
                logger.warning("Could not click first video, but search was completed")

        except Exception as e:
            logger.error(f"Failed to click first video: {e}")

    async def _google_search(self, search_term: str):
        """Search on Google."""
        try:
            # Wait for Google to load
            time.sleep(2)

            # Click on search box or just start typing (Google usually auto-focuses)
            self.tools.type_text(text=search_term)
            time.sleep(0.5)

            # Press Enter to search
            self.tools.press_key(key="enter")
            time.sleep(2)  # Wait for search results

            logger.info(f"Google search completed for: {search_term}")

        except Exception as e:
            logger.error(f"Google search failed: {e}")

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
async def run_normal_chrome_task(
    task: str,
    tools,
    on_status: Callable[[str], None] = None,
    on_step: Callable[[int, str, str], None] = None
) -> BrowserTaskResult:
    """
    Run a browser task using normal Chrome browser.

    Example:
        result = await run_normal_chrome_task(
            "Open a new tab and go to YouTube, search for cats",
            tools=windows_tools
        )
    """
    agent = NormalChromeAgent(on_status=on_status, on_step=on_step, tools=tools)
    return await agent.execute_task(task)