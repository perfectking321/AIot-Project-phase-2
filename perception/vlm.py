"""
VOXCODE Vision Language Model Integration
Semantic screen understanding using Qwen2.5-VL or MiniCPM-V via Ollama.
"""

import base64
import logging
import io
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("voxcode.perception.vlm")

try:
    from PIL import Image
    import requests
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import config


@dataclass
class VLMResponse:
    """Response from Vision Language Model."""
    content: str
    model: str
    success: bool = True
    error: Optional[str] = None


class VisionLanguageModel:
    """
    Vision-Language Model for semantic screen understanding.

    Uses multimodal models via Ollama to understand what's on screen,
    not just detect elements but understand their meaning and state.
    """

    # Available VLM models in Ollama (ordered by preference)
    SUPPORTED_MODELS = [
        "qwen2.5vl:7b",      # Best for GUI, 7B params
        "qwen2.5vl:3b",      # Lighter version
        "minicpm-v:8b",      # Good alternative
        "llava:7b",          # Fallback
        "llava:13b",         # Larger fallback
    ]

    # Prompt for semantic screen understanding
    SCREEN_UNDERSTANDING_PROMPT = """Analyze this screenshot and describe:

1. **Active Application**: What app/window is currently in focus?
2. **App State**: What state is the application in? (e.g., "chat open with John", "search results showing", "login page")
3. **Visible Apps**: List all visible applications/windows
4. **Ready Actions**: What actions can be taken right now? (e.g., "can type in message field", "can click send button")
5. **Key UI Elements**: Important buttons, text fields, or interactive elements visible

Be specific and concise. Focus on what a user would need to know to interact with this screen."""

    ELEMENT_GROUNDING_PROMPT = """Look at this screenshot. I need to find: "{element_description}"

If you can see this element:
1. Describe exactly where it is (e.g., "bottom right corner", "in the chat input area")
2. What does it look like?
3. Is it clickable/interactable?

If you cannot find it, say "NOT FOUND" and describe what you see instead."""

    STATE_CHECK_PROMPT = """Look at this screenshot and answer these specific questions:

1. Is "{app_name}" visible on screen? (yes/no)
2. If yes, what state is it in?
3. Is there a text input field ready for typing? (yes/no)
4. What was the last action that appears to have been taken?

Answer concisely."""

    def __init__(
        self,
        model: str = None,
        host: str = None,
        timeout: int = 60
    ):
        """
        Initialize VLM.

        Args:
            model: Model name (auto-detects if None)
            host: Ollama host URL
            timeout: Request timeout in seconds
        """
        self.host = (host or config.llm.ollama_host).rstrip("/")
        self.timeout = timeout
        self.model = model
        self._available_model = None

        if not PIL_AVAILABLE:
            logger.error("PIL not available - VLM requires PIL for image processing")

    def _detect_available_model(self) -> Optional[str]:
        """Detect which VLM model is available in Ollama."""
        if self._available_model:
            return self._available_model

        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] + ":" + m.get("name", "").split(":")[-1]
                              if ":" in m.get("name", "") else m.get("name", "")
                              for m in models]

                # Check for supported models
                for supported in self.SUPPORTED_MODELS:
                    base_name = supported.split(":")[0]
                    for available in model_names:
                        if base_name in available.lower():
                            self._available_model = available
                            logger.info(f"Found VLM model: {available}")
                            return available

                logger.warning(f"No VLM model found. Available: {model_names}")
                logger.info(f"Install one with: ollama pull qwen2.5vl:7b")

        except Exception as e:
            logger.error(f"Failed to detect VLM models: {e}")

        return None

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _make_request(self, prompt: str, images: List[Image.Image]) -> VLMResponse:
        """Make a request to Ollama with images."""
        model = self.model or self._detect_available_model()

        if not model:
            return VLMResponse(
                content="",
                model="none",
                success=False,
                error="No VLM model available. Install with: ollama pull qwen2.5vl:7b"
            )

        try:
            # Encode images
            encoded_images = [self._encode_image(img) for img in images]

            payload = {
                "model": model,
                "prompt": prompt,
                "images": encoded_images,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024
                }
            }

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            return VLMResponse(
                content=data.get("response", ""),
                model=model,
                success=True
            )

        except requests.exceptions.Timeout:
            return VLMResponse(
                content="",
                model=model,
                success=False,
                error="Request timed out"
            )
        except Exception as e:
            logger.error(f"VLM request failed: {e}")
            return VLMResponse(
                content="",
                model=model,
                success=False,
                error=str(e)
            )

    def understand_screen(self, screenshot: Image.Image) -> VLMResponse:
        """
        Get semantic understanding of the current screen.

        Args:
            screenshot: PIL Image of the screen

        Returns:
            VLMResponse with detailed screen description
        """
        return self._make_request(self.SCREEN_UNDERSTANDING_PROMPT, [screenshot])

    def find_element(self, screenshot: Image.Image, description: str) -> VLMResponse:
        """
        Find a specific element on screen by description.

        Args:
            screenshot: PIL Image of the screen
            description: What to look for (e.g., "the send button", "message input field")

        Returns:
            VLMResponse with element location description
        """
        prompt = self.ELEMENT_GROUNDING_PROMPT.format(element_description=description)
        return self._make_request(prompt, [screenshot])

    def check_app_state(self, screenshot: Image.Image, app_name: str) -> VLMResponse:
        """
        Check the state of a specific application.

        Args:
            screenshot: PIL Image of the screen
            app_name: Name of app to check (e.g., "WhatsApp", "Chrome")

        Returns:
            VLMResponse with app state information
        """
        prompt = self.STATE_CHECK_PROMPT.format(app_name=app_name)
        return self._make_request(prompt, [screenshot])

    def verify_action(
        self,
        before: Image.Image,
        after: Image.Image,
        action_description: str
    ) -> VLMResponse:
        """
        Verify if an action was successful by comparing before/after screenshots.

        Args:
            before: Screenshot before action
            after: Screenshot after action
            action_description: What action was attempted

        Returns:
            VLMResponse with verification result
        """
        prompt = f"""Compare these two screenshots. The action attempted was: "{action_description}"

First image: BEFORE the action
Second image: AFTER the action

Answer:
1. Did the screen change? (yes/no)
2. Was the action successful? (yes/no/unclear)
3. What changed on the screen?
4. Is there any error message visible?"""

        return self._make_request(prompt, [before, after])

    def custom_query(self, screenshot: Image.Image, query: str) -> VLMResponse:
        """
        Ask a custom question about the screen.

        Args:
            screenshot: PIL Image of the screen
            query: Custom question to ask

        Returns:
            VLMResponse with answer
        """
        return self._make_request(query, [screenshot])

    def is_available(self) -> bool:
        """Check if VLM is available."""
        return self._detect_available_model() is not None


# Global VLM instance
_vlm_instance: Optional[VisionLanguageModel] = None


def get_vlm(preload: bool = False) -> VisionLanguageModel:
    """Get or create global VLM instance."""
    global _vlm_instance

    if _vlm_instance is None:
        _vlm_instance = VisionLanguageModel()
        if preload:
            _vlm_instance._detect_available_model()

    return _vlm_instance
