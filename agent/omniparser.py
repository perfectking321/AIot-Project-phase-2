"""
VOXCODE OmniParser Integration
Screen parsing using Microsoft OmniParser for intelligent UI element detection.
"""

import os
import io
import base64
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("voxcode.omniparser")

# Check dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - OmniParser requires torch")

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available - install with: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Try importing transformers for icon captioning
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available for icon captioning")


@dataclass
class UIElement:
    """A detected UI element with semantic information."""
    id: int
    element_type: str  # 'text', 'icon', 'button', 'input', etc.
    label: str  # The text or description of the element
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    is_interactable: bool = True

    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            "id": self.id,
            "type": self.element_type,
            "label": self.label,
            "center": self.center,
            "bbox": self.bbox,
            "interactable": self.is_interactable
        }


@dataclass
class ParsedScreen:
    """Result of parsing a screen."""
    elements: List[UIElement] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    labeled_image_path: Optional[str] = None
    parse_time: float = 0.0

    def find_by_label(self, search_text: str, fuzzy: bool = True) -> List[UIElement]:
        """Find elements matching a label."""
        search_lower = search_text.lower()
        matches = []

        for elem in self.elements:
            label_lower = elem.label.lower()

            if fuzzy:
                if search_lower in label_lower or label_lower in search_lower:
                    matches.append(elem)
            else:
                if search_lower == label_lower:
                    matches.append(elem)

        # Sort by confidence
        matches.sort(key=lambda e: e.confidence, reverse=True)
        return matches

    def find_by_type(self, element_type: str) -> List[UIElement]:
        """Find elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]

    def get_interactable(self) -> List[UIElement]:
        """Get all interactable elements."""
        return [e for e in self.elements if e.is_interactable]

    def to_prompt_format(self) -> str:
        """Format elements for LLM prompt."""
        lines = ["## Screen Elements (click by ID or label):\n"]

        for elem in self.elements:
            if elem.is_interactable:
                lines.append(f"[{elem.id}] {elem.element_type}: \"{elem.label}\" at {elem.center}")

        return "\n".join(lines)

    def to_json(self) -> List[Dict]:
        """Convert to JSON-serializable format."""
        return [e.to_dict() for e in self.elements]


class OmniParser:
    """
    Screen parser using OmniParser approach.
    Combines YOLO detection + OCR + icon captioning for comprehensive UI understanding.
    """

    # Model paths (will be downloaded on first use)
    WEIGHTS_DIR = Path("weights")
    ICON_DETECT_MODEL = "icon_detect/model.pt"
    ICON_CAPTION_MODEL = "icon_caption_florence"

    def __init__(
        self,
        weights_dir: str = None,
        use_gpu: bool = None,
        preload: bool = False,
        use_caption_model: bool = False  # Disable by default for speed
    ):
        """
        Initialize OmniParser.

        Args:
            weights_dir: Directory containing model weights
            use_gpu: Force GPU usage (auto-detect if None)
            preload: Load models immediately instead of lazy loading
            use_caption_model: Whether to load icon caption model (slower but more descriptive)
        """
        self.weights_dir = Path(weights_dir) if weights_dir else self.WEIGHTS_DIR
        self.device = self._detect_device(use_gpu)
        self.use_caption_model = use_caption_model

        # Models (lazy loaded)
        self._yolo_model = None
        self._caption_model = None
        self._caption_processor = None
        self._ocr_reader = None

        # Screenshot directory
        self._screenshot_dir = Path("screenshots")
        self._screenshot_dir.mkdir(exist_ok=True)

        # Detection thresholds
        self.box_threshold = 0.05
        self.iou_threshold = 0.7

        logger.info(f"OmniParser initialized (device: {self.device})")

        if preload:
            self._load_models()

    def _detect_device(self, use_gpu: bool = None) -> str:
        """Detect best available device."""
        if use_gpu is False:
            return 'cpu'

        if TORCH_AVAILABLE and torch.cuda.is_available():
            return 'cuda'

        return 'cpu'

    def _load_models(self):
        """Load all models."""
        self._get_yolo_model()
        self._get_ocr_reader()
        if self.use_caption_model:
            self._get_caption_model()

    def _get_yolo_model(self):
        """Load YOLO model for icon/element detection."""
        if self._yolo_model is not None:
            return self._yolo_model

        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - using OCR only mode")
            return None

        model_path = self.weights_dir / self.ICON_DETECT_MODEL

        if not model_path.exists():
            logger.info("Downloading OmniParser icon detection model...")
            self._download_weights()

        if model_path.exists():
            try:
                self._yolo_model = YOLO(str(model_path))
                logger.info("YOLO icon detection model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
        else:
            logger.warning("YOLO model not found - using OCR only mode")

        return self._yolo_model

    def _get_ocr_reader(self):
        """Load EasyOCR reader."""
        if self._ocr_reader is not None:
            return self._ocr_reader

        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not available")
            return None

        try:
            logger.info("Loading EasyOCR...")
            self._ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), verbose=False)
            logger.info("EasyOCR loaded")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")

        return self._ocr_reader

    def _get_caption_model(self):
        """Load Florence-2 caption model for icon description."""
        if self._caption_model is not None:
            return self._caption_model, self._caption_processor

        if not TRANSFORMERS_AVAILABLE:
            return None, None

        model_path = self.weights_dir / self.ICON_CAPTION_MODEL

        if not model_path.exists():
            # Try using base Florence model
            model_path = "microsoft/Florence-2-base"

        try:
            logger.info("Loading Florence-2 caption model...")
            self._caption_processor = AutoProcessor.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._caption_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            logger.info("Florence-2 caption model loaded")
        except Exception as e:
            logger.warning(f"Could not load caption model: {e}")

        return self._caption_model, self._caption_processor

    def _download_weights(self):
        """Download OmniParser weights from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download

            self.weights_dir.mkdir(parents=True, exist_ok=True)

            # Download icon detection model
            logger.info("Downloading OmniParser v2 weights...")

            files = [
                "icon_detect/model.pt",
                "icon_detect/model.yaml",
                "icon_detect/train_args.yaml"
            ]

            for f in files:
                try:
                    hf_hub_download(
                        repo_id="microsoft/OmniParser-v2.0",
                        filename=f,
                        local_dir=str(self.weights_dir)
                    )
                except Exception as e:
                    logger.warning(f"Could not download {f}: {e}")

            logger.info("Weights downloaded successfully")

        except ImportError:
            logger.warning("huggingface_hub not available - install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")

    def take_screenshot(self) -> Optional[Image.Image]:
        """Capture current screen."""
        if not PYAUTOGUI_AVAILABLE:
            logger.error("pyautogui not available")
            return None

        try:
            return pyautogui.screenshot()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def parse_screen(self, image: Image.Image = None, save_labeled: bool = False) -> ParsedScreen:
        """
        Parse the screen and detect all UI elements.

        Args:
            image: PIL Image to parse (takes screenshot if None)
            save_labeled: Save a labeled version of the image

        Returns:
            ParsedScreen with detected elements
        """
        start_time = time.time()
        result = ParsedScreen()

        # Take screenshot if not provided
        if image is None:
            image = self.take_screenshot()
            if image is None:
                return result

        img_array = np.array(image)
        img_width, img_height = image.size

        elements = []
        element_id = 0

        # 1. Run OCR to find text elements
        ocr_reader = self._get_ocr_reader()
        if ocr_reader:
            try:
                ocr_results = ocr_reader.readtext(img_array)

                for (bbox, text, conf) in ocr_results:
                    if not text.strip() or conf < 0.3:
                        continue

                    # Convert bbox format
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))

                    # Determine element type based on text and position
                    elem_type = self._classify_text_element(text, (x1, y1, x2, y2), img_width, img_height)

                    element = UIElement(
                        id=element_id,
                        element_type=elem_type,
                        label=text.strip(),
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        is_interactable=elem_type in ['button', 'link', 'input', 'tab', 'menu_item', 'text']
                    )
                    elements.append(element)
                    element_id += 1

            except Exception as e:
                logger.error(f"OCR failed: {e}")

        # 2. Run YOLO to find icons and non-text elements
        yolo_model = self._get_yolo_model()
        if yolo_model:
            try:
                yolo_results = yolo_model(image, conf=self.box_threshold, verbose=False)

                for r in yolo_results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()

                        # Skip if overlaps significantly with existing OCR elements
                        if self._overlaps_existing(elements, (x1, y1, x2, y2)):
                            continue

                        # Crop and get icon description
                        icon_desc = self._get_icon_description(image, (int(x1), int(y1), int(x2), int(y2)))

                        element = UIElement(
                            id=element_id,
                            element_type='icon',
                            label=icon_desc,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf,
                            is_interactable=True
                        )
                        elements.append(element)
                        element_id += 1

            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        # Sort elements by position (top-left to bottom-right)
        elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

        # Reassign IDs after sorting
        for i, elem in enumerate(elements):
            elem.id = i

        result.elements = elements
        result.parse_time = time.time() - start_time

        # Save labeled image if requested
        if save_labeled:
            result.labeled_image_path = self._save_labeled_image(image, elements)

        logger.info(f"Parsed screen: {len(elements)} elements in {result.parse_time:.2f}s")

        return result

    def _classify_text_element(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int
    ) -> str:
        """Classify a text element based on its content and position."""
        text_lower = text.lower().strip()
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Button-like keywords
        button_keywords = [
            'submit', 'send', 'ok', 'cancel', 'save', 'delete', 'close', 'open',
            'search', 'login', 'sign in', 'sign up', 'register', 'next', 'back',
            'continue', 'done', 'apply', 'confirm', 'yes', 'no', 'accept', 'decline',
            'upload', 'download', 'install', 'update', 'refresh', 'retry'
        ]

        # Input field indicators
        input_keywords = [
            'type', 'enter', 'search', 'write', 'email', 'password', 'username',
            'name', 'message', 'comment', 'reply', 'placeholder'
        ]

        # Tab/menu keywords
        tab_keywords = ['home', 'settings', 'profile', 'chats', 'calls', 'status', 'notifications']

        # Check for button
        if any(kw in text_lower for kw in button_keywords):
            return 'button'

        # Check for input placeholder
        if any(kw in text_lower for kw in input_keywords):
            return 'input'

        # Check for tabs (usually at top of window)
        if y1 < img_height * 0.15 and any(kw in text_lower for kw in tab_keywords):
            return 'tab'

        # Check if it looks like a link (short text, single line)
        if width < img_width * 0.3 and height < 50:
            return 'link'

        # Default to text
        return 'text'

    def _overlaps_existing(
        self,
        elements: List[UIElement],
        bbox: Tuple[float, float, float, float],
        threshold: float = 0.5
    ) -> bool:
        """Check if bbox overlaps significantly with existing elements."""
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)

        if box_area == 0:
            return True

        for elem in elements:
            ex1, ey1, ex2, ey2 = elem.bbox

            # Calculate intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection / box_area > threshold:
                    return True

        return False

    def _get_icon_description(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
        """Get a description for an icon/image element."""
        if not self.use_caption_model:
            return "icon"

        model, processor = self._get_caption_model()
        if model is None:
            return "icon"

        try:
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2)).resize((64, 64))

            inputs = processor(images=cropped, text="<CAPTION>", return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20,
                    num_beams=1
                )

            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()

        except Exception as e:
            logger.debug(f"Icon captioning failed: {e}")
            return "icon"

    def _save_labeled_image(self, image: Image.Image, elements: List[UIElement]) -> str:
        """Save image with labeled bounding boxes."""
        from PIL import ImageDraw, ImageFont

        labeled = image.copy()
        draw = ImageDraw.Draw(labeled)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        colors = {
            'button': 'red',
            'input': 'blue',
            'link': 'green',
            'icon': 'orange',
            'tab': 'purple',
            'text': 'gray',
            'menu_item': 'cyan'
        }

        for elem in elements:
            color = colors.get(elem.element_type, 'white')
            x1, y1, x2, y2 = elem.bbox

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label
            label = f"[{elem.id}] {elem.label[:20]}"
            draw.text((x1, y1 - 15), label, fill=color, font=font)

        # Save
        filename = f"labeled_{int(time.time())}.png"
        filepath = self._screenshot_dir / filename
        labeled.save(filepath)

        return str(filepath)

    def find_element(self, search_text: str, parsed: ParsedScreen = None) -> Optional[UIElement]:
        """
        Find a UI element by label text.

        Args:
            search_text: Text to search for
            parsed: Pre-parsed screen (parses current screen if None)

        Returns:
            Best matching UIElement or None
        """
        if parsed is None:
            parsed = self.parse_screen()

        matches = parsed.find_by_label(search_text)
        return matches[0] if matches else None

    def click_element(self, element: UIElement) -> bool:
        """Click on a UI element."""
        if not PYAUTOGUI_AVAILABLE:
            return False

        try:
            cx, cy = element.center
            logger.info(f"Clicking [{element.id}] '{element.label}' at ({cx}, {cy})")
            pyautogui.click(cx, cy)
            time.sleep(0.3)
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False


# Global instance (lazy initialized)
_omniparser_instance = None

def get_omniparser(preload: bool = False) -> OmniParser:
    """Get or create the global OmniParser instance."""
    global _omniparser_instance
    if _omniparser_instance is None:
        _omniparser_instance = OmniParser(preload=preload)
    return _omniparser_instance
