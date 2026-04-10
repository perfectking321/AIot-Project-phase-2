"""
VOXCODE Browser Session Manager
Persistent singleton Playwright browser — launches its own visible Chromium.

NO remote debugging port needed. Playwright manages the browser lifecycle.
The browser stays open for the entire VOXCODE session (~50ms per action).

Usage:
    session = BrowserSessionManager.get_instance()
    page = await session.get_active_page()  # returns (browser, page)
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger("voxcode.browser_session")

try:
    from playwright.async_api import async_playwright, Playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. BrowserSessionManager disabled.")


class BrowserSessionManager:
    """
    Singleton that owns a persistent Playwright browser instance.

    This launches a REAL, visible Chromium browser (not headless).
    No --remote-debugging-port needed. Playwright controls it directly.

    The browser stays alive for the whole VOXCODE session.
    Every skill call reuses the same browser/page — ~50ms per action.
    """

    _instance: Optional["BrowserSessionManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def get_instance(cls) -> "BrowserSessionManager":
        return cls()

    def _init_once(self):
        if self._initialized:
            return
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._connected = False
        self._initialized = True

    # ── Connection management ──

    async def _ensure_connected(self):
        """Launch Playwright browser if not already running."""
        self._init_once()

        if self._connected and self._browser and self._browser.is_connected():
            return

        # If a stale connection exists, clean it up
        await self._cleanup()

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=[
                    "--start-maximized",
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ]
            )
            # Create a persistent context with a reasonable viewport
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                no_viewport=False,
            )
            # Open a blank page so there's always at least one
            page = await self._context.new_page()
            await page.goto("about:blank")

            self._connected = True
            logger.info("BrowserSessionManager: launched Playwright Chromium browser")
        except Exception as e:
            await self._cleanup()
            raise RuntimeError(f"Cannot launch Playwright browser: {e}")

    async def _cleanup(self):
        """Release playwright resources."""
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._playwright = None
        self._context = None
        self._connected = False

    async def get_active_page(self) -> Tuple["Browser", "Page"]:
        """
        Return (browser, active_page).

        The caller must NOT close the browser. The page reference is valid
        until the next navigation or until the browser disconnects.
        """
        await self._ensure_connected()

        pages = self._context.pages
        if not pages:
            # Create a new page if somehow all were closed
            page = await self._context.new_page()
            await page.goto("about:blank")
            pages = [page]

        # Return last page (most recently active)
        return self._browser, pages[-1]

    async def new_page(self) -> "Page":
        """Create a new tab/page and return it."""
        await self._ensure_connected()
        page = await self._context.new_page()
        return page

    async def get_all_pages(self):
        """Return all open pages."""
        await self._ensure_connected()
        return self._context.pages

    async def disconnect(self):
        """Explicitly disconnect (e.g. on shutdown)."""
        await self._cleanup()
        logger.info("BrowserSessionManager: disconnected")

    # ── Accessibility tree shortcut ──

    async def get_accessibility_snapshot(self, page: "Page" = None):
        """
        Return the Chrome Accessibility Tree snapshot for the active page.

        Uses CDP directly since page.accessibility was removed in Playwright 1.58+.
        Returns a simplified tree: {role, name, children: [...]}
        """
        if page is None:
            _, page = await self.get_active_page()
        try:
            # Use CDP session to get the accessibility tree
            cdp = await page.context.new_cdp_session(page)
            result = await cdp.send("Accessibility.getFullAXTree")
            await cdp.detach()

            # Convert raw AX nodes into a simplified tree
            nodes = result.get("nodes", [])
            if not nodes:
                return None

            # Build lookup by nodeId
            node_map = {}
            for node in nodes:
                node_id = node.get("nodeId", "")
                role = node.get("role", {}).get("value", "")
                name = node.get("name", {}).get("value", "")
                ignored = node.get("ignored", False)
                child_ids = node.get("childIds", [])

                if ignored:
                    continue

                node_map[node_id] = {
                    "role": role,
                    "name": name,
                    "children": [],
                    "_child_ids": child_ids,
                }

            # Build tree structure
            root = None
            for node_id, node_data in node_map.items():
                for child_id in node_data.get("_child_ids", []):
                    if child_id in node_map:
                        node_data["children"].append(node_map[child_id])
                if root is None:
                    root = node_data  # First non-ignored node is root

            # Clean up internal fields
            def clean(n):
                n.pop("_child_ids", None)
                for c in n.get("children", []):
                    clean(c)
            if root:
                clean(root)

            return root
        except Exception as e:
            logger.error(f"Accessibility snapshot failed: {e}")
            return None

    @staticmethod
    def is_available() -> bool:
        """Check if Playwright is installed and usable."""
        return PLAYWRIGHT_AVAILABLE


def run_async(coro):
    """Run an async coroutine from synchronous code safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
