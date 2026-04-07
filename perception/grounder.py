"""
VOXCODE Element Grounder
Precise element location using ShowUI API or local grounding.
"""

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("voxcode.perception.grounder")

try:
    from PIL import Image
    import requests
    import io
    import base64
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class GroundingResult:
    """Result of element grounding."""
    found: bool
    x: Optional[int] = None
    y: Optional[int] = None
    confidence: float = 0.0
    description: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Get center point."""
        if self.x is not None and self.y is not None:
            return (self.x, self.y)
        return None


class ElementGrounder:
    """
    Precise element grounding - finds exact coordinates for UI elements.

    Uses ShowUI API (free HuggingFace Spaces) or falls back to local VLM.
    """

    # ShowUI HuggingFace Spaces API endpoint
    SHOWUI_API = "https://showlab-showui.hf.space/api/predict"

    def __init__(self, use_showui_api: bool = True, fallback_to_vlm: bool = True):
        """
        Initialize grounder.

        Args:
            use_showui_api: Try ShowUI API first (free, no GPU needed)
            fallback_to_vlm: Fall back to local VLM if ShowUI fails
        """
        self.use_showui_api = use_showui_api
        self.fallback_to_vlm = fallback_to_vlm
        self._vlm = None

    def _get_vlm(self):
        """Lazy load VLM for fallback."""
        if self._vlm is None:
            from perception.vlm import get_vlm
            self._vlm = get_vlm()
        return self._vlm

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode image to base64."""
        buffer = io.BytesIO()
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def ground_element(
        self,
        screenshot: Image.Image,
        element_description: str
    ) -> GroundingResult:
        """
        Find precise coordinates of an element on screen.

        Args:
            screenshot: PIL Image of the screen
            element_description: What to find (e.g., "the send button", "search box")

        Returns:
            GroundingResult with coordinates if found
        """
        # Try ShowUI API first
        if self.use_showui_api:
            result = self._ground_with_showui(screenshot, element_description)
            if result.found:
                return result
            logger.info("ShowUI API didn't find element, trying fallback...")

        # Fallback to VLM-based grounding
        if self.fallback_to_vlm:
            return self._ground_with_vlm(screenshot, element_description)

        return GroundingResult(
            found=False,
            description=f"Could not locate: {element_description}"
        )

    def _ground_with_showui(
        self,
        screenshot: Image.Image,
        description: str
    ) -> GroundingResult:
        """Use ShowUI API for grounding."""
        try:
            # Prepare image
            img_base64 = self._encode_image_base64(screenshot)

            # ShowUI API expects specific format
            # Note: This is the Gradio API format
            payload = {
                "data": [
                    f"data:image/png;base64,{img_base64}",  # Image
                    description,  # Query
                ]
            }

            response = requests.post(
                self.SHOWUI_API,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Parse ShowUI response
                # Format varies, but typically returns coordinates
                if "data" in data and len(data["data"]) > 0:
                    result_text = str(data["data"][0])

                    # Try to extract coordinates
                    coords = self._parse_coordinates(result_text, screenshot.size)
                    if coords:
                        return GroundingResult(
                            found=True,
                            x=coords[0],
                            y=coords[1],
                            confidence=0.8,
                            description=f"Found via ShowUI: {description}"
                        )

            logger.warning(f"ShowUI API response unclear: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("ShowUI API timeout")
        except Exception as e:
            logger.error(f"ShowUI API error: {e}")

        return GroundingResult(found=False, description="ShowUI API failed")

    def _ground_with_vlm(
        self,
        screenshot: Image.Image,
        description: str
    ) -> GroundingResult:
        """Use local VLM for grounding with coordinate estimation."""
        vlm = self._get_vlm()

        # Ask VLM for element location
        prompt = f"""Find the element: "{description}"

Look at the screenshot and tell me:
1. Can you see this element? (yes/no)
2. If yes, describe its EXACT position in percentages:
   - Horizontal: X% from left (0% = left edge, 100% = right edge)
   - Vertical: Y% from top (0% = top edge, 100% = bottom edge)
3. What does it look like?

Example response format:
FOUND: yes
POSITION: X=75%, Y=85%
DESCRIPTION: Blue send button in bottom right corner

If not found, respond:
FOUND: no
REASON: [why you can't see it]"""

        response = vlm._make_request(prompt, [screenshot])

        if response.success:
            # Parse response for coordinates
            coords = self._parse_vlm_coordinates(response.content, screenshot.size)
            if coords:
                return GroundingResult(
                    found=True,
                    x=coords[0],
                    y=coords[1],
                    confidence=0.6,  # Lower confidence for VLM-estimated
                    description=f"VLM estimated: {description}"
                )

        return GroundingResult(
            found=False,
            description=f"VLM could not locate: {description}"
        )

    def _parse_coordinates(self, text: str, image_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Parse coordinates from various response formats."""
        import re

        width, height = image_size

        # Try to find pixel coordinates (x, y) or [x, y]
        pixel_pattern = r'[\(\[]?\s*(\d+)\s*[,;]\s*(\d+)\s*[\)\]]?'
        match = re.search(pixel_pattern, text)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            # Validate coordinates
            if 0 <= x <= width and 0 <= y <= height:
                return (x, y)

        # Try percentage format
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(percent_pattern, text)
        if len(matches) >= 2:
            x_percent = float(matches[0])
            y_percent = float(matches[1])
            x = int(width * x_percent / 100)
            y = int(height * y_percent / 100)
            return (x, y)

        return None

    def _parse_vlm_coordinates(self, text: str, image_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Parse coordinates from VLM response."""
        import re

        text_lower = text.lower()

        # Check if found
        if "found: no" in text_lower or "not found" in text_lower or "cannot see" in text_lower:
            return None

        width, height = image_size

        # Look for percentage format: X=75%, Y=85%
        x_match = re.search(r'x\s*[=:]\s*(\d+(?:\.\d+)?)\s*%', text_lower)
        y_match = re.search(r'y\s*[=:]\s*(\d+(?:\.\d+)?)\s*%', text_lower)

        if x_match and y_match:
            x_percent = float(x_match.group(1))
            y_percent = float(y_match.group(1))
            x = int(width * x_percent / 100)
            y = int(height * y_percent / 100)
            return (x, y)

        # Try general coordinate parsing
        return self._parse_coordinates(text, image_size)

    def ground_multiple(
        self,
        screenshot: Image.Image,
        descriptions: list
    ) -> Dict[str, GroundingResult]:
        """
        Find multiple elements on screen.

        Args:
            screenshot: PIL Image
            descriptions: List of element descriptions

        Returns:
            Dict mapping description to GroundingResult
        """
        results = {}
        for desc in descriptions:
            results[desc] = self.ground_element(screenshot, desc)
        return results


# Global instance
_grounder_instance: Optional[ElementGrounder] = None


def get_grounder() -> ElementGrounder:
    """Get or create global grounder instance."""
    global _grounder_instance

    if _grounder_instance is None:
        _grounder_instance = ElementGrounder()

    return _grounder_instance
