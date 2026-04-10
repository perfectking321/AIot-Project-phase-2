"""
VOXCODE Trace Utilities
Structured debug events and screenshot capture for end-to-end troubleshooting.
"""

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import config

logger = logging.getLogger("voxcode.trace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_jsonable(value: Any, max_text: int = 1000) -> Any:
    """Safely convert values to JSON-serializable structures."""
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) <= max_text:
            return value
        return f"{value[:max_text]}... <truncated {len(value) - max_text} chars>"

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, max_text=max_text) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item, max_text=max_text)
            for key, item in value.items()
        }

    return str(value)


def _safe_slug(text: str) -> str:
    """Create a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip()).strip("_")
    if not slug:
        return "event"
    return slug[:80]


class TraceLogger:
    """Centralized structured tracing for VoxCode sessions."""

    def __init__(self):
        now = datetime.utcnow()
        self.session_id = now.strftime("run_%Y%m%d_%H%M%S_%f")
        self._sequence = 0
        self._lock = threading.Lock()

        screenshot_root = Path(getattr(config.agent, "trace_screenshot_dir", "screenshots/sessions"))
        if not screenshot_root.is_absolute():
            screenshot_root = PROJECT_ROOT / screenshot_root
        self.screenshot_dir = screenshot_root / self.session_id

        self.audit_path = PROJECT_ROOT / "audit_log.jsonl"

        if getattr(config.agent, "trace_screenshots", True):
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trace logger initialized: session_id=%s, screenshots=%s",
            self.session_id,
            self.screenshot_dir,
        )

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def _relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path).replace("\\", "/")

    def log_event(self, source: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Write one structured trace event to audit_log.jsonl."""
        if not getattr(config.agent, "trace_enabled", True):
            return

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "sequence": self._next_sequence(),
            "source": source,
            "event_type": event_type,
            "payload": _to_jsonable(payload or {}),
        }

        try:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Trace event write failed: %s", exc)

    def log_exception(self, source: str, event_type: str, exc: Exception, payload: Optional[Dict[str, Any]] = None) -> None:
        """Log an exception as a structured event."""
        error_payload = dict(payload or {})
        error_payload["error_type"] = type(exc).__name__
        error_payload["error"] = str(exc)
        self.log_event(source=source, event_type=event_type, payload=error_payload)

    def start_run(self, source: str, goal: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the start of a command/task run."""
        payload = {"goal": goal}
        if metadata:
            payload.update(metadata)
        self.log_event(source=source, event_type="run_started", payload=payload)

    def capture_screenshot(
        self,
        source: str,
        tag: str,
        *,
        step_num: Optional[int] = None,
        retry_count: Optional[int] = None,
        image=None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Capture a screenshot and log it as trace evidence."""
        if not getattr(config.agent, "trace_enabled", True):
            return None

        if not getattr(config.agent, "trace_screenshots", True):
            return None

        try:
            if image is None:
                import pyautogui
                image = pyautogui.screenshot()

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")
            parts = [timestamp]
            if step_num is not None:
                parts.append(f"s{step_num:02d}")
            if retry_count is not None:
                parts.append(f"r{retry_count}")
            parts.append(_safe_slug(tag))
            filename = "_".join(parts) + ".png"

            path = self.screenshot_dir / filename
            image.save(path)

            rel = self._relative_path(path)
            payload = {
                "tag": tag,
                "path": rel,
            }
            if step_num is not None:
                payload["step_num"] = step_num
            if retry_count is not None:
                payload["retry_count"] = retry_count
            if extra_payload:
                payload.update(extra_payload)

            self.log_event(source=source, event_type="screenshot_captured", payload=payload)
            logger.info("Trace screenshot saved: %s", rel)
            return rel

        except Exception as exc:
            self.log_exception(
                source=source,
                event_type="screenshot_capture_failed",
                exc=exc,
                payload={"tag": tag, "step_num": step_num, "retry_count": retry_count},
            )
            logger.warning("Screenshot capture failed for %s: %s", tag, exc)
            return None

    def get_active_window_title(self) -> str:
        """Best-effort active window title capture for trace context."""
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and getattr(win, "title", ""):
                return win.title
        except Exception:
            pass
        return "Unknown"


_trace_logger: Optional[TraceLogger] = None


def get_trace_logger() -> TraceLogger:
    """Get the global TraceLogger instance."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = TraceLogger()
    return _trace_logger
