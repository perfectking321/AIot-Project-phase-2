"""
VOXCODE Eyes - OmniParser with Florence-2 icon captioning enabled.
Returns filtered, labeled screen elements for Qwen consumption.
Implements token compression via proximity filtering.
"""
import time
import logging
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("voxcode.eyes")


@dataclass
class ScreenElement:
    """A detected screen element with all relevant properties."""
    id: int
    label: str
    center: Tuple[int, int]  # (x, y)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    element_type: str  # 'text', 'icon', 'button', 'input', 'link', 'tab'

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "center": self.center,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "type": self.element_type
        }


class Eyes:
    """
    Wraps OmniParser. Florence-2 captioning ENABLED so icons get real names
    instead of 'icon'. Implements element filtering by proximity to reduce
    Qwen prompt tokens (token compression formula).

    Key optimizations:
    - 100ms cache to avoid double-scanning same frame
    - Proximity filtering reduces ~800 tokens to ~200 tokens
    - Florence-2 provides semantic icon descriptions
    """

    def __init__(self, use_caption_model: bool = True, preload: bool = True):
        """
        Initialize Eyes with OmniParser backend.

        Args:
            use_caption_model: Enable Florence-2 for icon captioning (slower but better labels)
            preload: Load models immediately vs lazy loading
        """
        from agent.omniparser import OmniParser

        # Enable Florence-2 caption model for icon identification
        self.parser = OmniParser(
            use_caption_model=use_caption_model,
            preload=preload
        )

        # Cache for speculative pre-scanning
        self._last_parsed: Optional[List[ScreenElement]] = None
        self._last_parse_time: float = 0
        self.cache_ttl: float = 0.1  # 100ms cache - avoid double-scanning same frame

        # Screen dimensions (will be set on first scan)
        self._screen_width: int = 1920
        self._screen_height: int = 1080

        logger.info(f"Eyes initialized (caption_model={use_caption_model})")

    def scan(self, force: bool = False) -> List[ScreenElement]:
        """
        Scan screen. Returns all elements as ScreenElement list.
        Cached for cache_ttl seconds to support speculative pre-scanning.

        Args:
            force: Bypass cache and force fresh scan

        Returns:
            List of ScreenElement objects
        """
        now = time.time()

        # Return cached if valid and not forced
        if not force and self._last_parsed and (now - self._last_parse_time) < self.cache_ttl:
            logger.debug(f"Eyes: returning cached scan ({len(self._last_parsed)} elements)")
            return self._last_parsed

        # Parse screen using OmniParser
        start_time = time.time()
        parsed = self.parser.parse_screen(save_labeled=False)

        # Convert to ScreenElement list
        elements: List[ScreenElement] = []
        for e in parsed.elements:
            elements.append(ScreenElement(
                id=e.id,
                label=e.label,
                center=e.center,
                bbox=e.bbox,
                confidence=e.confidence,
                element_type=e.element_type
            ))

        # Update screen dimensions from first element if available
        if elements and parsed.elements:
            # Get max coords to estimate screen size
            max_x = max(e.bbox[2] for e in parsed.elements)
            max_y = max(e.bbox[3] for e in parsed.elements)
            if max_x > 0:
                self._screen_width = max_x
            if max_y > 0:
                self._screen_height = max_y

        # Update cache
        self._last_parsed = elements
        self._last_parse_time = now

        scan_time = (time.time() - start_time) * 1000
        logger.info(f"Eyes: scanned {len(elements)} elements in {scan_time:.0f}ms")

        return elements

    def filter_near(
        self,
        elements: List[ScreenElement],
        cx: int,
        cy: int,
        radius: int = 300
    ) -> List[ScreenElement]:
        """
        Token compression: return only elements within radius pixels of (cx, cy).
        Reduces Qwen prompt from ~800 tokens (50 elements) to ~200 tokens (10-15 elements).
        If no elements in radius, return closest 10 overall.

        Args:
            elements: All screen elements
            cx, cy: Center point for filtering
            radius: Pixel radius for inclusion

        Returns:
            Filtered list of nearby elements
        """
        # Find elements within radius (Manhattan distance for speed)
        nearby = [
            e for e in elements
            if abs(e.center[0] - cx) <= radius and abs(e.center[1] - cy) <= radius
        ]

        # If we have enough nearby elements, return them (capped at 15)
        if len(nearby) >= 3:
            # Sort by confidence, take top 15
            nearby.sort(key=lambda e: e.confidence, reverse=True)
            return nearby[:15]

        # Fallback: sort all by distance, return closest 10
        elements_sorted = sorted(
            elements,
            key=lambda e: ((e.center[0] - cx) ** 2 + (e.center[1] - cy) ** 2) ** 0.5
        )
        return elements_sorted[:10]

    def filter_by_region(
        self,
        elements: List[ScreenElement],
        region: str = "full"
    ) -> List[ScreenElement]:
        """
        Filter elements by screen region.

        Args:
            elements: All screen elements
            region: 'top', 'bottom', 'left', 'right', 'center', 'full'

        Returns:
            Elements in specified region
        """
        if region == "full":
            return elements

        w, h = self._screen_width, self._screen_height

        if region == "top":
            return [e for e in elements if e.center[1] < h * 0.3]
        elif region == "bottom":
            return [e for e in elements if e.center[1] > h * 0.7]
        elif region == "left":
            return [e for e in elements if e.center[0] < w * 0.3]
        elif region == "right":
            return [e for e in elements if e.center[0] > w * 0.7]
        elif region == "center":
            return [
                e for e in elements
                if w * 0.2 < e.center[0] < w * 0.8 and h * 0.2 < e.center[1] < h * 0.8
            ]

        return elements

    def find_element_by_text(
        self,
        elements: List[ScreenElement],
        search_text: str,
        fuzzy: bool = True
    ) -> Optional[ScreenElement]:
        """
        Find element by label text.

        Args:
            elements: List of elements to search
            search_text: Text to find
            fuzzy: Allow partial matches

        Returns:
            Best matching element or None
        """
        search_lower = search_text.lower().strip()
        matches = []

        for elem in elements:
            label_lower = elem.label.lower().strip()

            if fuzzy:
                # Check if search text is in label or vice versa
                if search_lower in label_lower or label_lower in search_lower:
                    matches.append(elem)
            else:
                if search_lower == label_lower:
                    matches.append(elem)

        # Sort by confidence and return best match
        if matches:
            matches.sort(key=lambda e: e.confidence, reverse=True)
            return matches[0]

        return None

    def elements_to_prompt_str(self, elements: List[ScreenElement]) -> str:
        """
        Format elements for Qwen prompt. Minimal tokens.

        Args:
            elements: List of screen elements

        Returns:
            Formatted string for LLM consumption (~16 tokens per element)
        """
        lines = []
        for e in elements:
            # Format: "- type 'label' at (x, y) conf:0.XX"
            # This is ~16 tokens per element
            lines.append(
                f"[{e.id}] {e.element_type} '{e.label}' at {e.center} (conf:{e.confidence:.2f})"
            )
        return "\n".join(lines)

    def elements_to_compact_str(self, elements: List[ScreenElement]) -> str:
        """
        Ultra-compact format for very large element lists.
        ~10 tokens per element.

        Args:
            elements: List of screen elements

        Returns:
            Compact string format
        """
        lines = []
        for e in elements:
            # Format: "[id] type:label@(x,y)"
            short_label = e.label[:25] + "..." if len(e.label) > 25 else e.label
            lines.append(f"[{e.id}] {e.element_type}:'{short_label}'@{e.center}")
        return "\n".join(lines)

    def get_screen_summary(self, elements: List[ScreenElement]) -> str:
        """
        Get a brief summary of what's on screen.
        Useful for initial context.

        Args:
            elements: List of screen elements

        Returns:
            Brief summary string
        """
        # Count by type
        type_counts = {}
        for e in elements:
            type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1

        # Get some notable labels
        notable = [e.label for e in elements if len(e.label) > 3][:5]

        summary_parts = [f"{count} {etype}s" for etype, count in type_counts.items()]
        summary = f"Screen has {len(elements)} elements: {', '.join(summary_parts)}"

        if notable:
            summary += f". Notable items: {', '.join(notable[:3])}"

        return summary


# Singleton instance
_eyes: Optional[Eyes] = None


def get_eyes(use_caption_model: bool = True, preload: bool = True) -> Eyes:
    """
    Get or create the global Eyes instance.

    Args:
        use_caption_model: Enable Florence-2 for icon captioning
        preload: Load models immediately

    Returns:
        Global Eyes instance
    """
    global _eyes
    if _eyes is None:
        _eyes = Eyes(use_caption_model=use_caption_model, preload=preload)
    return _eyes


def reset_eyes():
    """Reset the global Eyes instance (useful for testing)."""
    global _eyes
    _eyes = None
