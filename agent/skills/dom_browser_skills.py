"""
VOXCODE DOM Browser Skills — v2 (Modern Agent Architecture)

Replaces the old "connect → act → disconnect" pattern with a persistent
BrowserSessionManager and structured DOM perception.

Skills provided:
- DOMObserveSkill       — inject JS, return interactive + visible elements with ref IDs
- DOMClickRefSkill      — click element by ref_id (xPath-based, no coordinates)
- DOMTypeRefSkill       — type text into an input identified by ref_id
- DOMNavigateSkill      — navigate to a URL
- DOMReadPageSkill      — read page title, URL, and full text content
- DOMExtractSkill       — structured data extraction from current page
- DOMSearchExtractSkill — search Google and return top results as JSON
- DOMScrollSkill        — scroll page or element
- DOMWaitSkill          — wait for element or network idle

All skills use the BrowserSessionManager singleton for persistent connections.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, List, Any

from agent.skills.base import Skill, SkillResult, SkillStatus

logger = logging.getLogger("voxcode.skills.dom_browser")

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. DOM browser skills disabled.")

# ── Helpers ──

def _get_session_manager():
    """Lazy import to avoid circular imports."""
    from agent.browser_session import BrowserSessionManager
    return BrowserSessionManager.get_instance()


def _run_async(coro):
    """Run an async coroutine from sync skill context."""
    from agent.browser_session import run_async
    return run_async(coro)


# ── JavaScript for DOM element extraction ──

BUILD_DOM_TREE_JS = """
(() => {
    const INTERACTIVE_ROLES = new Set([
        'button', 'link', 'textbox', 'searchbox', 'combobox',
        'checkbox', 'radio', 'tab', 'menuitem', 'option',
        'switch', 'slider', 'spinbutton', 'listbox', 'menu',
        'menubar', 'tablist', 'treeitem'
    ]);

    const INTERACTIVE_TAGS = new Set([
        'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA',
        'LABEL', 'SUMMARY', 'DETAILS'
    ]);

    function isVisible(el) {
        if (!el || !el.getBoundingClientRect) return false;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none') return false;
        if (style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) <= 0) return false;
        return true;
    }

    function isInteractive(el) {
        if (INTERACTIVE_TAGS.has(el.tagName)) return true;
        const role = el.getAttribute('role');
        if (role && INTERACTIVE_ROLES.has(role.toLowerCase())) return true;
        if (el.onclick || el.getAttribute('onclick')) return true;
        if (el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1') return true;
        if (el.contentEditable === 'true') return true;
        const style = window.getComputedStyle(el);
        if (style.cursor === 'pointer') return true;
        return false;
    }

    function getXPath(el) {
        const segments = [];
        let current = el;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let index = 0;
            let sibling = current.previousSibling;
            while (sibling) {
                if (sibling.nodeType === Node.ELEMENT_NODE &&
                    sibling.tagName === current.tagName) {
                    index++;
                }
                sibling = sibling.previousSibling;
            }
            const tagName = current.tagName.toLowerCase();
            const xpathIndex = index > 0 ? `[${index + 1}]` : '';
            segments.unshift(`${tagName}${xpathIndex}`);
            current = current.parentNode;
        }
        return '/' + segments.join('/');
    }

    function getElementText(el) {
        // Prefer aria-label, then placeholder, then innerText (truncated)
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel.substring(0, 80);
        const placeholder = el.getAttribute('placeholder');
        if (placeholder) return placeholder.substring(0, 80);
        const title = el.getAttribute('title');
        if (title) return title.substring(0, 80);
        const alt = el.getAttribute('alt');
        if (alt) return alt.substring(0, 80);
        const text = (el.innerText || el.textContent || '').trim();
        return text.substring(0, 80);
    }

    const elements = [];
    const allEls = document.querySelectorAll('*');
    let refId = 0;

    for (const el of allEls) {
        if (!isVisible(el)) continue;
        if (!isInteractive(el)) continue;

        const rect = el.getBoundingClientRect();
        elements.push({
            ref_id: refId,
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            type: el.getAttribute('type') || '',
            name: getElementText(el),
            value: el.value || '',
            xpath: getXPath(el),
            rect: {
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height)
            },
            href: el.href || '',
            aria_label: el.getAttribute('aria-label') || '',
            placeholder: el.getAttribute('placeholder') || '',
            checked: el.checked || false,
            disabled: el.disabled || false
        });
        refId++;

        if (refId >= 150) break;  // Token budget: max 150 elements
    }

    return elements;
})()
"""


class _ElementCache:
    """Cache the last observe() result so click_ref/type_ref can use it."""
    elements: List[Dict[str, Any]] = []
    timestamp: float = 0.0
    page_url: str = ""

    @classmethod
    def update(cls, elements: List[Dict], url: str):
        import time
        cls.elements = elements
        cls.timestamp = time.time()
        cls.page_url = url

    @classmethod
    def get_by_ref(cls, ref_id: int) -> Optional[Dict]:
        for el in cls.elements:
            if el.get("ref_id") == ref_id:
                return el
        return None

    @classmethod
    def is_stale(cls, max_age: float = 30.0) -> bool:
        import time
        return (time.time() - cls.timestamp) > max_age


# ═══════════════════════════════════════════════════════════════
# SKILL: Observe (list interactive elements)
# ═══════════════════════════════════════════════════════════════

class DOMObserveSkill(Skill):
    """
    Inject JS into the page and return a list of interactive + visible elements.
    Each element gets a ref_id that can be used by click_ref / type_ref.
    """

    name = "dom_observe"
    description = "List all interactive elements on the current page with ref IDs"
    params = []
    preconditions = []
    postconditions = ["elements_observed"]

    def execute(self, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        async def _observe():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            elements = await page.evaluate(BUILD_DOM_TREE_JS)
            url = page.url
            return elements, url

        try:
            elements, url = _run_async(_observe())
            _ElementCache.update(elements, url)

            # Build compact summary for LLM
            summary_lines = []
            for el in elements[:50]:  # Show top 50 in summary
                name = el.get("name", "").strip()
                tag = el.get("tag", "")
                role = el.get("role", "")
                ref = el.get("ref_id", "")
                label = name or role or tag
                summary_lines.append(f"  [{ref}] <{tag}> {label}")

            summary = "\n".join(summary_lines)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Found {len(elements)} interactive elements on {url}",
                data={
                    "elements": elements,
                    "count": len(elements),
                    "url": url,
                    "summary": summary,
                }
            )
        except Exception as e:
            logger.error(f"DOMObserveSkill error: {e}")
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Click by ref_id
# ═══════════════════════════════════════════════════════════════

class DOMClickRefSkill(Skill):
    """Click an element by its ref_id from the last observe() call."""

    name = "dom_click_ref"
    description = "Click an element by ref_id (from observe) or by visible text"
    params = ["ref_id", "text"]
    preconditions = []
    postconditions = ["element_clicked"]

    def execute(self, ref_id: int = None, text: str = None, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        if ref_id is None and text is None:
            return SkillResult(status=SkillStatus.FAILED, message="ref_id or text required")

        async def _click():
            session = _get_session_manager()
            browser, page = await session.get_active_page()

            if ref_id is not None:
                el = _ElementCache.get_by_ref(ref_id)
                if el and el.get("xpath"):
                    xpath = el["xpath"]
                    try:
                        await page.click(f"xpath={xpath}", timeout=5000)
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                        return f"Clicked ref {ref_id}: {el.get('name', '')}"
                    except Exception:
                        # Fallback: click by coordinates from cached rect
                        rect = el.get("rect", {})
                        if rect:
                            x = rect["x"] + rect["width"] // 2
                            y = rect["y"] + rect["height"] // 2
                            await page.mouse.click(x, y)
                            await page.wait_for_load_state("domcontentloaded", timeout=5000)
                            return f"Clicked ref {ref_id} by coordinates ({x}, {y})"
                        raise

            if text:
                await page.click(f"text={text}", timeout=5000)
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                return f"Clicked text: {text}"

            return "Nothing to click"

        try:
            msg = _run_async(_click())
            return SkillResult(status=SkillStatus.SUCCESS, message=msg)
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Type text by ref_id
# ═══════════════════════════════════════════════════════════════

class DOMTypeRefSkill(Skill):
    """Type text into an input field identified by ref_id."""

    name = "dom_type_ref"
    description = "Type text into a field by ref_id (from observe) or selector"
    params = ["ref_id", "text", "selector"]
    preconditions = []
    postconditions = ["text_typed"]

    def execute(self, ref_id: int = None, text: str = "",
                selector: str = None, press_enter: bool = False, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        if not text:
            return SkillResult(status=SkillStatus.FAILED, message="text is required")

        async def _type():
            session = _get_session_manager()
            browser, page = await session.get_active_page()

            if ref_id is not None:
                el = _ElementCache.get_by_ref(ref_id)
                if el and el.get("xpath"):
                    xpath = el["xpath"]
                    try:
                        await page.fill(f"xpath={xpath}", text, timeout=5000)
                    except Exception:
                        # Fallback: click then type
                        await page.click(f"xpath={xpath}", timeout=5000)
                        await page.keyboard.type(text, delay=30)
                    if press_enter:
                        await page.keyboard.press("Enter")
                    return f"Typed into ref {ref_id}"
            elif selector:
                await page.fill(selector, text, timeout=5000)
                if press_enter:
                    await page.keyboard.press("Enter")
                return f"Typed into {selector}"
            else:
                # Type into currently focused element
                await page.keyboard.type(text, delay=30)
                if press_enter:
                    await page.keyboard.press("Enter")
                return "Typed into focused element"

        try:
            msg = _run_async(_type())
            return SkillResult(status=SkillStatus.SUCCESS, message=msg)
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Navigate
# ═══════════════════════════════════════════════════════════════

class DOMNavigateSkill(Skill):
    """Navigate to a URL in the active tab."""

    name = "dom_navigate"
    description = "Navigate to a specific URL in the browser"
    params = ["url"]
    preconditions = []
    postconditions = ["page_navigated"]

    def execute(self, url: str = None, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")
        if not url:
            return SkillResult(status=SkillStatus.FAILED, message="url is required")

        async def _navigate():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return page.url

        try:
            final_url = _run_async(_navigate())
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Navigated to {final_url}",
                data={"url": final_url}
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Read Page
# ═══════════════════════════════════════════════════════════════

class DOMReadPageSkill(Skill):
    """Read the current page's title, URL, and text content."""

    name = "dom_read_page"
    description = "Read the current browser page content (title, URL, body text)"
    params = []
    preconditions = []
    postconditions = ["page_content_read"]

    def execute(self, max_chars: int = 3000, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        async def _read():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            title = await page.title()
            url = page.url
            body_text = await page.inner_text("body")
            body_text = body_text[:max_chars].strip()
            return {
                "title": title,
                "url": url,
                "body_text": body_text,
                "char_count": len(body_text)
            }

        try:
            data = _run_async(_read())
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Read page: '{data['title']}' at {data['url']}",
                data=data
            )
        except Exception as e:
            logger.error(f"DOMReadPageSkill error: {e}")
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Extract (structured data from page — Stagehand pattern)
# ═══════════════════════════════════════════════════════════════

class DOMExtractSkill(Skill):
    """Run custom JavaScript in the browser and return the result."""

    name = "dom_extract"
    description = "Run JavaScript in the browser page context and return the result"
    params = ["js_code"]
    preconditions = []
    postconditions = ["data_extracted"]

    def execute(self, js_code: str = None, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")
        if not js_code:
            return SkillResult(status=SkillStatus.FAILED, message="js_code is required")

        async def _extract():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            result = await page.evaluate(js_code)
            return result

        try:
            result = _run_async(_extract())
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message="JavaScript executed successfully",
                data={"result": result}
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Search Google & extract results
# ═══════════════════════════════════════════════════════════════

class DOMSearchExtractSkill(Skill):
    """Search Google and return top results as structured JSON."""

    name = "dom_search_extract"
    description = "Search Google and return top results as structured data"
    params = ["query"]
    preconditions = []
    postconditions = ["search_results_extracted"]

    GOOGLE_RESULTS_JS = """
    (() => {
        const results = [];
        const cards = document.querySelectorAll('.g, [data-hveid]');
        for (const card of cards) {
            const titleEl = card.querySelector('h3');
            const linkEl = card.querySelector('a');
            const snippetEl = card.querySelector('.VwiC3b, .s3v9rd, [data-sncf]');
            if (titleEl && linkEl) {
                results.push({
                    title: titleEl.textContent.trim(),
                    link: linkEl.href,
                    snippet: snippetEl ? snippetEl.textContent.trim() : ''
                });
            }
            if (results.length >= 8) break;
        }
        return results;
    })()
    """

    def execute(self, query: str = None, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")
        if not query:
            return SkillResult(status=SkillStatus.FAILED, message="query is required")

        import urllib.parse
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"

        async def _search_and_extract():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(2000)
            results = await page.evaluate(self.GOOGLE_RESULTS_JS)
            return results

        try:
            results = _run_async(_search_and_extract())
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Found {len(results)} results for: {query}",
                data={"query": query, "results": results}
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Scroll
# ═══════════════════════════════════════════════════════════════

class DOMScrollSkill(Skill):
    """Scroll the page or a specific element."""

    name = "dom_scroll"
    description = "Scroll the page up/down by a specified amount"
    params = ["direction", "amount"]
    preconditions = []
    postconditions = ["page_scrolled"]

    def execute(self, direction: str = "down", amount: int = 500, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        async def _scroll():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            dy = amount if direction == "down" else -amount
            await page.mouse.wheel(0, dy)
            await page.wait_for_timeout(300)
            return f"Scrolled {direction} by {amount}px"

        try:
            msg = _run_async(_scroll())
            return SkillResult(status=SkillStatus.SUCCESS, message=msg)
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Wait
# ═══════════════════════════════════════════════════════════════

class DOMWaitSkill(Skill):
    """Wait for an element or network idle on the current page."""

    name = "dom_wait"
    description = "Wait for an element to appear or page network to be idle"
    params = ["selector"]
    preconditions = []
    postconditions = ["element_visible"]

    def execute(self, selector: str = None, timeout: int = 10000,
                wait_for: str = "selector", **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        async def _wait():
            session = _get_session_manager()
            browser, page = await session.get_active_page()
            if wait_for == "selector" and selector:
                await page.wait_for_selector(selector, timeout=timeout)
            else:
                await page.wait_for_load_state(wait_for, timeout=timeout)
            return True

        try:
            _run_async(_wait())
            return SkillResult(
                status=SkillStatus.SUCCESS,
                message=f"Wait condition met: {selector or wait_for}"
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ═══════════════════════════════════════════════════════════════
# SKILL: Accessibility Tree (Playwright MCP pattern)
# ═══════════════════════════════════════════════════════════════

class DOMAccessibilitySkill(Skill):
    """Read the Chrome Accessibility Tree for the current page."""

    name = "dom_accessibility"
    description = "Get the Chrome Accessibility Tree (roles, names, states)"
    params = []
    preconditions = []
    postconditions = ["accessibility_read"]

    def execute(self, **kwargs) -> SkillResult:
        if not PLAYWRIGHT_AVAILABLE:
            return SkillResult(status=SkillStatus.FAILED, message="Playwright not installed")

        async def _get_tree():
            session = _get_session_manager()
            snapshot = await session.get_accessibility_snapshot()
            return snapshot

        try:
            tree = _run_async(_get_tree())
            if tree:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message="Accessibility tree retrieved",
                    data={"tree": tree}
                )
            return SkillResult(status=SkillStatus.FAILED, message="Empty accessibility tree")
        except Exception as e:
            return SkillResult(status=SkillStatus.FAILED, message=str(e))


# ── Backward compatibility re-exports ──
# Some existing code (dispatcher.py) imports these names
DOMFillSkill = DOMTypeRefSkill
DOMClickSkill = DOMClickRefSkill

# Module-level flags for backward compatibility
DOM_AVAILABLE = False
if PLAYWRIGHT_AVAILABLE:
    from agent.browser_session import BrowserSessionManager
    DOM_AVAILABLE = BrowserSessionManager.is_available()
