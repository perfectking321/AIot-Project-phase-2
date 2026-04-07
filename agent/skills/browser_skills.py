"""
VOXCODE Browser Skills
Skills for browser navigation and web searching.
"""

import time
import logging
from typing import Optional

from agent.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger("voxcode.skills.browser")


class NavigateToUrlSkill(Skill):
    """Skill to navigate to a URL in a browser."""

    name = "navigate_url"
    description = "Navigate to a URL in the browser"
    params = ["url", "browser"]
    preconditions = []
    postconditions = ["page_loaded"]

    def execute(
        self,
        url: str = None,
        browser: str = "chrome",
        new_tab: bool = False,
        **kwargs
    ) -> SkillResult:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to
            browser: Which browser to use
            new_tab: Open in new tab

        Returns:
            SkillResult
        """
        if not url:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="url is required"
            )

        tools = self._get_tools()

        # Ensure browser is open
        from agent.skills.app_skills import OpenAppSkill
        open_app = OpenAppSkill(tools=self._tools)
        result = open_app.execute(app_name=browser)

        time.sleep(1.0)

        # Open new tab if requested
        if new_tab:
            tools.hotkey("ctrl", "t")
            time.sleep(0.5)

        # Focus address bar
        tools.hotkey("ctrl", "l")
        time.sleep(0.3)

        # Type URL
        tools.type_text(url)
        time.sleep(0.2)

        # Navigate
        tools.press_key("enter")
        time.sleep(2.0)  # Wait for page load

        return SkillResult(
            status=SkillStatus.SUCCESS,
            message=f"Navigated to {url}",
            data={"url": url, "browser": browser}
        )


class SearchWebSkill(Skill):
    """Skill to search the web."""

    name = "search_web"
    description = "Search the web using a search engine"
    params = ["query", "engine"]
    preconditions = []
    postconditions = ["search_results_visible"]

    # Search engine URL patterns
    SEARCH_ENGINES = {
        "google": "https://www.google.com/search?q=",
        "bing": "https://www.bing.com/search?q=",
        "duckduckgo": "https://duckduckgo.com/?q=",
        "youtube": "https://www.youtube.com/results?search_query=",
    }

    def execute(
        self,
        query: str = None,
        engine: str = "google",
        browser: str = "chrome",
        **kwargs
    ) -> SkillResult:
        """
        Search the web.

        Args:
            query: Search query
            engine: Search engine to use
            browser: Browser to use

        Returns:
            SkillResult
        """
        if not query:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="query is required"
            )

        tools = self._get_tools()

        # Get search URL
        engine_lower = engine.lower()
        if engine_lower in self.SEARCH_ENGINES:
            base_url = self.SEARCH_ENGINES[engine_lower]
        else:
            base_url = self.SEARCH_ENGINES["google"]

        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote_plus(query)
        search_url = base_url + encoded_query

        # Navigate to search
        nav_skill = NavigateToUrlSkill(tools=self._tools)
        result = nav_skill.execute(url=search_url, browser=browser)

        if result.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Searched for: {query}",
                data={"query": query, "engine": engine, "url": search_url}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Failed to search: {result.message}"
        )


class ClickLinkSkill(Skill):
    """Skill to click a link on a webpage."""

    name = "click_link"
    description = "Click a link on the current webpage"
    params = ["link_text"]
    preconditions = ["page_loaded"]
    postconditions = ["link_clicked"]

    def execute(self, link_text: str = None, **kwargs) -> SkillResult:
        """
        Click a link on the page.

        Args:
            link_text: Text of the link to click

        Returns:
            SkillResult
        """
        if not link_text:
            return SkillResult(
                status=SkillStatus.FAILED,
                message="link_text is required"
            )

        tools = self._get_tools()
        result = tools.click_text(link_text)

        if result.success:
            time.sleep(1.0)  # Wait for navigation
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Clicked link: {link_text}",
                data={"link_text": link_text}
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message=f"Link not found: {link_text}"
        )


class NewTabSkill(Skill):
    """Skill to open a new browser tab."""

    name = "new_tab"
    description = "Open a new browser tab"
    params = []
    preconditions = ["browser_open"]
    postconditions = ["new_tab_open"]

    def execute(self, **kwargs) -> SkillResult:
        """Open a new tab."""
        tools = self._get_tools()
        result = tools.hotkey("ctrl", "t")

        if result.success:
            time.sleep(0.5)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message="Opened new tab"
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message="Failed to open new tab"
        )


class CloseTabSkill(Skill):
    """Skill to close the current browser tab."""

    name = "close_tab"
    description = "Close the current browser tab"
    params = []
    preconditions = ["browser_open"]
    postconditions = ["tab_closed"]

    def execute(self, **kwargs) -> SkillResult:
        """Close current tab."""
        tools = self._get_tools()
        result = tools.hotkey("ctrl", "w")

        if result.success:
            time.sleep(0.3)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message="Closed tab"
            )

        return SkillResult(
            status=SkillStatus.FAILED,
            message="Failed to close tab"
        )
