"""
VOXCODE Verifier - Fast pixel-diff screen change detection + audit logging.
No LLM calls. No Groq on failure. Just log and retry via Qwen.

Verification is done by comparing screen regions before/after action.
If pixels didn't change significantly, action likely failed.
"""
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from brain.llm import OllamaClient, get_model_for_role
from config import config

logger = logging.getLogger("voxcode.verifier")

# Audit log path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = PROJECT_ROOT / "audit_log.jsonl"


class Verifier:
    """
    Pixel diff check: compare region around action point before/after.
    If unchanged -> action likely failed -> log to audit file.

    Fast: ~5ms per check (just numpy array comparison).
    No API calls on failure - just logs and signals retry.
    """

    def __init__(
        self,
        change_threshold: float = 5.0,
        check_radius: int = 100,
        audit_log_path: Optional[Path] = None
    ):
        """
        Initialize Verifier.

        Args:
            change_threshold: Minimum mean pixel difference to consider "changed"
                            5.0 means avg pixel shift of 5/255 across region
            check_radius: Radius in pixels around action point to capture
            audit_log_path: Path for audit log (default: audit_log.jsonl)
        """
        self.threshold = change_threshold
        self.radius = check_radius
        self.audit_path = audit_log_path or AUDIT_LOG_PATH

        # Statistics
        self.total_checks = 0
        self.changes_detected = 0
        self.no_changes_detected = 0

        self._semantic_client = get_model_for_role("verifier")

        logger.info(
            f"Verifier initialized (threshold={change_threshold}, radius={check_radius})"
        )

    def capture_region(self, cx: int, cy: int) -> Optional[np.ndarray]:
        """
        Capture screen region around point.

        Args:
            cx, cy: Center coordinates

        Returns:
            numpy array of pixels or None on error
        """
        try:
            import pyautogui
            from PIL import Image

            # Calculate region bounds (ensure non-negative)
            x1 = max(0, cx - self.radius)
            y1 = max(0, cy - self.radius)
            width = self.radius * 2
            height = self.radius * 2

            # Capture region
            shot = pyautogui.screenshot(region=(x1, y1, width, height))
            return np.array(shot)

        except Exception as e:
            logger.warning(f"Region capture failed: {e}")
            return None

    def capture_full_screen(self) -> Optional[np.ndarray]:
        """
        Capture full screen for broad change detection.

        Returns:
            numpy array of full screen or None
        """
        try:
            import pyautogui
            shot = pyautogui.screenshot()
            return np.array(shot)
        except Exception as e:
            logger.warning(f"Full screen capture failed: {e}")
            return None

    def did_screen_change(
        self,
        before: Optional[np.ndarray],
        after: Optional[np.ndarray],
        update_stats: bool = True
    ) -> bool:
        """
        Pixel diff formula: mean absolute difference of region.
        Returns True if screen changed (action likely succeeded).

        Args:
            before: Screenshot before action
            after: Screenshot after action
            update_stats: Whether to update statistics (default True)

        Returns:
            True if significant change detected
        """
        if update_stats:
            self.total_checks += 1

        # If we can't compare, assume changed (safer)
        if before is None or after is None:
            logger.debug("Cannot compare - missing screenshot, assuming changed")
            if update_stats:
                self.changes_detected += 1
            return True

        # If shapes don't match, definitely changed
        if before.shape != after.shape:
            logger.debug("Shape mismatch - screen changed")
            if update_stats:
                self.changes_detected += 1
            return True

        # Calculate mean absolute difference
        diff = np.abs(before.astype(float) - after.astype(float)).mean()

        changed = diff > self.threshold

        if changed:
            if update_stats:
                self.changes_detected += 1
            logger.debug(f"Pixel diff: {diff:.2f} > {self.threshold} (CHANGED)")
        else:
            if update_stats:
                self.no_changes_detected += 1
            logger.debug(f"Pixel diff: {diff:.2f} <= {self.threshold} (NO CHANGE)")

        return changed

    def get_change_percentage(
        self,
        before: Optional[np.ndarray],
        after: Optional[np.ndarray]
    ) -> float:
        """
        Get percentage of pixels that changed significantly.

        Args:
            before, after: Screenshots to compare

        Returns:
            Percentage of changed pixels (0-100)
        """
        if before is None or after is None or before.shape != after.shape:
            return 100.0  # Assume full change if can't compare

        # Per-pixel threshold
        pixel_threshold = 20  # Consider pixel changed if diff > 20/255

        diff = np.abs(before.astype(float) - after.astype(float))
        changed_pixels = (diff > pixel_threshold).sum()
        total_pixels = diff.size

        return (changed_pixels / total_pixels) * 100

    def audit_log(self, entry: Dict[str, Any]) -> None:
        """
        Append audit entry to JSONL file. Never raises.
        Each line = one action attempt with full context.

        Args:
            entry: Dictionary with action details
        """
        try:
            entry["timestamp"] = datetime.now().isoformat()
            entry["verifier_stats"] = {
                "total_checks": self.total_checks,
                "changes_detected": self.changes_detected,
                "no_changes": self.no_changes_detected
            }

            # Ensure directory exists
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")

    @staticmethod
    def _elements_to_semantic_text(elements_seen: List[Any]) -> str:
        """Render OmniParser elements into compact text for semantic checks."""
        lines: List[str] = []
        for idx, elem in enumerate(elements_seen or []):
            if isinstance(elem, dict):
                label = elem.get("label", "")
                elem_type = elem.get("type", elem.get("element_type", "element"))
                center = elem.get("center", ("?", "?"))
            else:
                label = getattr(elem, "label", "")
                elem_type = getattr(elem, "element_type", "element")
                center = getattr(elem, "center", ("?", "?"))
            lines.append(f"[{idx}] {elem_type} '{label}' at {center}")
        return "\n".join(lines)

    def semantic_verify(self, verify_condition: str, elements_seen: List[Any]) -> bool:
        """Check a verify_condition against current OmniParser elements using Ollama."""
        condition = (verify_condition or "").strip()
        if not condition:
            return True

        elements_text = self._elements_to_semantic_text(elements_seen)
        prompt = f"""Screen elements:
{elements_text}

Condition to check: "{condition}"

Is this condition met? Reply only: YES or NO"""

        try:
            response = self._semantic_client.generate(prompt)
            answer = (response.content or "").strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.warning(f"Semantic verify failed: {e}")
            return False

    def verify_action(
        self,
        subgoal: str,
        decision: Dict[str, Any],
        elements_seen: List[Any],
        before_region: Optional[np.ndarray],
        after_region: Optional[np.ndarray],
        verify_condition: str = "",
        retry_count: int = 0,
        before_screenshot: Optional[str] = None,
        after_screenshot: Optional[str] = None,
        active_window: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Full verify cycle. Returns True if action succeeded.
        Logs to audit regardless of outcome.

        Args:
            subgoal: The task being performed
            decision: The action that was executed
            elements_seen: Screen elements at time of decision
            before_region: Screenshot before action
            after_region: Screenshot after action
            retry_count: Current retry attempt number

        Returns:
            True if screen changed (action succeeded), False otherwise
        """
        condition = (verify_condition or subgoal or "").strip()
        changed = bool(self.semantic_verify(condition, elements_seen))

        coord_x = decision.get("x")
        coord_y = decision.get("y")
        if coord_x is not None:
            coord_x = int(coord_x)
        if coord_y is not None:
            coord_y = int(coord_y)

        # Build audit entry
        audit_entry = {
            "subgoal": subgoal,
            "decision": decision,
            "elements_count": len(elements_seen) if elements_seen else 0,
            "semantic_condition": condition,
            "semantic_verified": bool(changed),
            "retry_count": int(retry_count),
            "action": decision.get("action"),
            "active_window": active_window,
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot,
            "coords": {
                "x": coord_x,
                "y": coord_y
            } if decision.get("action") == "click" else None
        }

        if extra_context:
            audit_entry["extra_context"] = extra_context

        self.audit_log(audit_entry)

        if not changed:
            logger.warning(
                f"Action may have failed (verify condition not met): "
                f"{decision} for subgoal '{subgoal}'"
            )

        return changed

    def wait_for_screen_settle(
        self,
        max_wait: float = 2.0,
        check_interval: float = 0.1,
        settle_threshold: float = 0.5
    ) -> bool:
        """
        Wait for screen to stop changing (e.g., after page load).

        Args:
            max_wait: Maximum time to wait in seconds
            check_interval: Time between checks
            settle_threshold: Max change percentage to consider "settled"

        Returns:
            True if screen settled, False if timed out
        """
        start_time = time.time()
        last_capture = self.capture_full_screen()

        while time.time() - start_time < max_wait:
            time.sleep(check_interval)
            current = self.capture_full_screen()

            change_pct = self.get_change_percentage(last_capture, current)

            if change_pct < settle_threshold:
                logger.debug(f"Screen settled after {time.time()-start_time:.1f}s")
                return True

            last_capture = current

        logger.debug(f"Screen settle timeout after {max_wait}s")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "total_checks": self.total_checks,
            "changes_detected": self.changes_detected,
            "no_changes_detected": self.no_changes_detected,
            "success_rate": (
                self.changes_detected / self.total_checks * 100
                if self.total_checks > 0 else 0
            )
        }


# Singleton instance
_verifier: Optional[Verifier] = None


def get_verifier(
    change_threshold: float = 5.0,
    check_radius: int = 100
) -> Verifier:
    """
    Get or create the global Verifier instance.

    Args:
        change_threshold: Pixel difference threshold
        check_radius: Capture region radius

    Returns:
        Global Verifier instance
    """
    global _verifier
    if _verifier is None:
        _verifier = Verifier(
            change_threshold=change_threshold,
            check_radius=check_radius
        )
    return _verifier


def reset_verifier():
    """Reset the global Verifier instance (useful for testing)."""
    global _verifier
    _verifier = None
