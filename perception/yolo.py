"""
VOXCODE YOLO Perception
YOLOv8n fine-tuned on GUI elements → numbered bounding boxes.
Weights path: weights/icon_detect/model.pt (already present via OmniParser)
"""
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger("voxcode.perception.yolo")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed: pip install ultralytics")

WEIGHTS_PATH = Path("weights/icon_detect/model.pt")
BOX_THRESHOLD = 0.05
IOU_THRESHOLD = 0.7


@dataclass
class DetectedElement:
    id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def crop_region(self) -> Tuple[int, int, int, int]:
        return self.bbox


class YOLOPerception:
    """Detects interactable UI elements using YOLOv8n GUI fine-tuned model."""

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            logger.error("ultralytics not available")
            return
        if not WEIGHTS_PATH.exists():
            logger.warning(f"YOLO weights not found at {WEIGHTS_PATH}. Falling back to OCR-only mode.")
            return
        try:
            self._model = YOLO(str(WEIGHTS_PATH))
            logger.info(f"YOLO model loaded from {WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")

    def detect(self, image: Image.Image) -> List[DetectedElement]:
        """Detect UI elements. Returns list sorted top-left to bottom-right."""
        if self._model is None:
            return []

        img_array = np.array(image)
        elements = []

        try:
            results = self._model(image, conf=BOX_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            element_id = 0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    elements.append(DetectedElement(
                        id=element_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf
                    ))
                    element_id += 1

            # Sort top-left to bottom-right
            elements.sort(key=lambda e: (e.bbox[1] // 50, e.bbox[0]))
            for i, e in enumerate(elements):
                e.id = i

            logger.info(f"YOLO detected {len(elements)} elements")
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")

        return elements

    @property
    def available(self) -> bool:
        return self._model is not None


_instance: Optional[YOLOPerception] = None

def get_yolo() -> YOLOPerception:
    global _instance
    if _instance is None:
        _instance = YOLOPerception()
    return _instance
