"""
VOXCODE Screen Vision Module
Find UI elements on screen using EasyOCR.
"""

import os
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("voxcode.vision")

try:
    import pyautogui
    from PIL import Image
    import numpy as np
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Try to import EasyOCR
try:
    import easyocr
    OCR_AVAILABLE = True
    logger.info("EasyOCR available")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("EasyOCR not available - install with: pip install easyocr")


@dataclass
class ScreenElement:
    """A UI element found on screen."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class ScreenVision:
    """Screen analysis and element finding using EasyOCR."""

    def __init__(self, preload: bool = False):
        self._screenshot_dir = "screenshots"
        os.makedirs(self._screenshot_dir, exist_ok=True)
        self._reader = None  # Lazy load EasyOCR

        # Preload OCR model if requested (for faster first use)
        if preload:
            self._get_reader()

    def _get_reader(self):
        """Load EasyOCR reader (called at init if preload=True, otherwise lazy-loaded)."""
        if self._reader is None and OCR_AVAILABLE:
            logger.info("Loading EasyOCR model (first time may take a moment)...")
            self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR model loaded")
        return self._reader

    def take_screenshot(self, region: Tuple[int, int, int, int] = None) -> Optional[Image.Image]:
        """Take a screenshot of the screen or a region."""
        if not PYAUTOGUI_AVAILABLE:
            logger.error("pyautogui not available")
            return None

        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            return screenshot
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def save_screenshot(self, filename: str = None) -> Optional[str]:
        """Take and save a screenshot."""
        screenshot = self.take_screenshot()
        if not screenshot:
            return None

        if not filename:
            filename = f"screen_{int(time.time())}.png"

        filepath = os.path.join(self._screenshot_dir, filename)
        screenshot.save(filepath)
        logger.info(f"Screenshot saved: {filepath}")
        return filepath

    def find_text_on_screen(self, search_text: str, exact: bool = False) -> List[ScreenElement]:
        """
        Find text on screen using OCR.
        Returns list of matching elements with their positions.
        """
        if not OCR_AVAILABLE:
            logger.error("OCR not available - install easyocr")
            return []

        screenshot = self.take_screenshot()
        if not screenshot:
            return []

        reader = self._get_reader()
        if not reader:
            return []

        try:
            # Convert PIL image to numpy array
            img_array = np.array(screenshot)

            # Run OCR
            logger.info(f"Running OCR to find: '{search_text}'")
            results = reader.readtext(img_array)

            elements = []
            search_lower = search_text.lower()

            for (bbox, text, conf) in results:
                if not text.strip():
                    continue

                text_clean = text.strip().strip('"\'')
                text_lower = text_clean.lower()

                # Check for match
                if exact:
                    match = text_lower == search_lower
                    match_score = 1.0 if match else 0.0
                else:
                    # Partial match with scoring
                    if text_lower == search_lower:
                        match = True
                        match_score = 2.0  # Exact match gets highest priority
                    elif search_lower in text_lower:
                        match = True
                        match_score = 1.5  # Search term is part of found text
                    elif text_lower in search_lower:
                        match = True
                        match_score = 0.5  # Found text is part of search term (less likely what we want)
                    else:
                        match = False
                        match_score = 0.0

                if match:
                    # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))

                    element = ScreenElement(
                        text=text_clean,
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=conf * match_score  # Weight by match quality
                    )
                    elements.append(element)
                    logger.info(f"Found '{text_clean}' at ({element.x}, {element.y}) conf={conf:.2f} match_score={match_score}")

            # Sort by weighted confidence (match quality * OCR confidence)
            elements.sort(key=lambda e: e.confidence, reverse=True)
            return elements

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def find_all_text(self) -> List[ScreenElement]:
        """Get all text visible on screen."""
        if not OCR_AVAILABLE:
            return []

        screenshot = self.take_screenshot()
        if not screenshot:
            return []

        reader = self._get_reader()
        if not reader:
            return []

        try:
            img_array = np.array(screenshot)
            results = reader.readtext(img_array)

            elements = []
            for (bbox, text, conf) in results:
                if not text.strip() or conf < 0.3:
                    continue

                x1 = int(min(p[0] for p in bbox))
                y1 = int(min(p[1] for p in bbox))
                x2 = int(max(p[0] for p in bbox))
                y2 = int(max(p[1] for p in bbox))

                element = ScreenElement(
                    text=text,
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=conf
                )
                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def click_text(self, search_text: str) -> bool:
        """Find text on screen and click it."""
        elements = self.find_text_on_screen(search_text)

        if not elements:
            logger.warning(f"Text not found: {search_text}")
            return False

        # Click the best match
        element = elements[0]
        cx, cy = element.center
        logger.info(f"Clicking '{element.text}' at ({cx}, {cy})")

        try:
            pyautogui.click(cx, cy)
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False


# Global instance
vision = ScreenVision()
