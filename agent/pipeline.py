"""
VOXCODE Pipeline - 3-model orchestration with pipeline parallelism.

Flow per subgoal:
  Thread A: OmniParser scan (runs ahead speculatively)
  Thread B: Qwen decide + execute + pixel verify
  Overlap: while B executes action, A scans next subgoal's screen

Latency target: ~242ms per cycle on consumer GPU (RTX 3060)

Key optimizations:
1. Amdahl: Qwen INT4 = 3.73x faster inference
2. Pipeline parallelism: OmniParser scan overlaps with Qwen inference
3. Speculative pre-scan: scan N+1 during action execution of N
4. Token compression: filter to ~10 nearest elements (~200 tokens vs 800)
5. Pixel diff verify: 5ms check, no API call on failure
"""
import time
import logging
import threading
import re
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional, List, Callable, Tuple, Any, Dict

from config import config
from agent.eyes import get_eyes, Eyes, ScreenElement
from agent.hands import get_hands, Hands, SubgoalTracker
from agent.trace import get_trace_logger
from agent.tools import WindowsTools
from agent.verifier import get_verifier, Verifier
from brain.planner import TaskPlan, Subtask

logger = logging.getLogger("voxcode.pipeline")

# Constants
MAX_RETRIES = 3
ACTION_SETTLE_TIME = 0.3  # seconds to wait for UI to update after action
DEFAULT_SCREEN_CENTER = (960, 540)


class Pipeline:
    """
    Orchestrates Eyes -> Hands -> Verifier with pipeline parallelism.

    The pipeline runs each subgoal through:
    1. SCAN: OmniParser detects UI elements
    2. DECIDE: Qwen LLM picks an action
    3. EXECUTE: pyautogui performs the action
    4. VERIFY: pixel-diff checks if screen changed

    Parallelism: While step N executes, we speculatively scan for step N+1.
    """

    def __init__(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[int, str, str], None]] = None,
        use_caption_model: bool = False,  # Disable for speed
        preload_models: bool = True
    ):
        """
        Initialize Pipeline with all components.

        Args:
            on_status: Callback for status updates (msg)
            on_step: Callback for step updates (step_num, msg, status)
            use_caption_model: Enable Florence-2 for icon captioning
            preload_models: Load models immediately
        """
        # Callbacks
        self.on_status = on_status or (lambda msg: None)
        self.on_step = on_step or (lambda step, msg, status: None)

        # Components (lazy loaded)
        self._eyes: Optional[Eyes] = None
        self._hands: Optional[Hands] = None
        self._verifier: Optional[Verifier] = None
        self._tools: Optional[WindowsTools] = None
        self._use_caption = use_caption_model

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")

        # Control
        self._stop = False
        self._lock = threading.Lock()

        # Track last action coords for proximity-based filtering
        self._last_action_x = DEFAULT_SCREEN_CENTER[0]
        self._last_action_y = DEFAULT_SCREEN_CENTER[1]

        # Statistics
        self.total_subgoals = 0
        self.successful_subgoals = 0
        self.failed_subgoals = 0
        self.total_retries = 0
        self._trace = get_trace_logger()

        if preload_models:
            self._load_components()

        logger.info("Pipeline initialized")

    def _load_components(self):
        """Load all components (Eyes, Hands, Verifier)."""
        logger.info("Loading pipeline components...")
        self._get_eyes()
        self._get_hands()
        self._get_verifier()
        logger.info("Pipeline components loaded")

    def _get_eyes(self) -> Eyes:
        """Get or create Eyes instance."""
        if self._eyes is None:
            self._eyes = get_eyes(use_caption_model=self._use_caption, preload=True)
        return self._eyes

    def _get_hands(self) -> Hands:
        """Get or create Hands instance."""
        if self._hands is None:
            self._hands = get_hands()
        return self._hands

    def _get_verifier(self) -> Verifier:
        """Get or create Verifier instance."""
        if self._verifier is None:
            self._verifier = get_verifier()
        return self._verifier

    def _get_tools(self) -> WindowsTools:
        """Get or create tools instance for deterministic utility actions."""
        if self._tools is None:
            self._tools = WindowsTools(use_omniparser=False)
        return self._tools

    def stop(self):
        """Signal pipeline to stop."""
        with self._lock:
            self._stop = True
        logger.info("Pipeline stop requested")

    def is_stopped(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop

    def reset(self):
        """Reset pipeline state for new task."""
        with self._lock:
            self._stop = False
        self._last_action_x = DEFAULT_SCREEN_CENTER[0]
        self._last_action_y = DEFAULT_SCREEN_CENTER[1]
        self.total_subgoals = 0
        self.successful_subgoals = 0
        self.failed_subgoals = 0
        self.total_retries = 0

    def _element_preview(self, elements: List[ScreenElement]) -> List[Dict[str, Any]]:
        """Serialize a bounded element list for trace logs."""
        limit = max(1, int(getattr(config.agent, "trace_element_preview_limit", 25)))
        preview = []
        for element in elements[:limit]:
            preview.append(
                {
                    "id": element.id,
                    "label": element.label,
                    "center": element.center,
                    "bbox": element.bbox,
                    "confidence": round(float(element.confidence), 4),
                    "type": element.element_type,
                }
            )

        if len(elements) > limit:
            preview.append({"truncated": len(elements) - limit})

        return preview

    def _capture_step_screenshot(
        self,
        *,
        step_num: int,
        retry_count: int,
        tag: str,
        subtask: str,
    ) -> Optional[str]:
        """Capture per-attempt screenshots for deterministic debugging."""
        return self._trace.capture_screenshot(
            source="pipeline",
            tag=tag,
            step_num=step_num,
            retry_count=retry_count,
            extra_payload={"subtask": subtask},
        )

    def _elements_text(self, elements: List[ScreenElement]) -> str:
        """Build normalized text corpus from detected elements."""
        return " ".join((e.label or "").lower() for e in elements)

    def _postconditions_met(self, elements: List[ScreenElement], postconditions: List[str]) -> bool:
        """
        Semantic postcondition check against currently visible elements.
        """
        if not postconditions:
            return True
        for condition in postconditions:
            cond = (condition or "").strip()
            if not cond:
                continue
            if not self._semantic_verify_condition(cond, elements):
                return False
        return True

    def _expected_state_text(self, subtask: Subtask) -> str:
        """Compact expected-state text for the decision model."""
        if getattr(subtask, "verify_condition", ""):
            return subtask.verify_condition
        if subtask.output_state:
            return subtask.output_state
        if subtask.postconditions:
            return ", ".join(subtask.postconditions)
        return ""

    def _subtask_verify_condition(self, subtask: Subtask) -> str:
        """Resolve the semantic verify condition for a subtask."""
        verify_condition = (getattr(subtask, "verify_condition", "") or "").strip()
        if verify_condition:
            return verify_condition
        if subtask.postconditions:
            return subtask.postconditions[0]
        if subtask.output_state:
            return subtask.output_state
        return subtask.description

    def _semantic_verify_condition(self, verify_condition: str, elements: Optional[List[ScreenElement]] = None) -> bool:
        """Semantic state gate using verifier + OmniParser elements."""
        if not verify_condition:
            return True
        eyes = self._get_eyes()
        verifier = self._get_verifier()
        current_elements = elements if elements is not None else eyes.scan(force=True)
        return verifier.semantic_verify(verify_condition, current_elements)

    def _extract_app_target(self, subtask: Subtask) -> str:
        """Extract app target from planner params/description."""
        params = subtask.params or {}
        for key in ("app_name", "path_or_name", "target", "application", "app"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        match = re.search(r"(?:open|launch|start|ensure|close|exit|quit)\s+([a-zA-Z0-9 ._:-]+)", subtask.description, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .")
            candidate = re.split(r"\s+(?:to|and|then|for|is|are|with)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            return candidate

        return subtask.description.strip()

    @staticmethod
    def _looks_like_close_intent(text: str) -> bool:
        """Detect close/exit intent phrases."""
        return bool(re.search(r"\b(close|exit|quit|minimi[sz]e)\b", text or "", flags=re.IGNORECASE))

    def _window_matches_target(self, target: str, active_window_title: str) -> bool:
        """Check whether active window title likely corresponds to the target app."""
        window = (active_window_title or "").lower()
        if not window:
            return False

        target_lower = (target or "").lower().strip()
        stem = Path(target_lower).stem if target_lower else ""
        aliases = {
            target_lower,
            stem,
            target_lower.replace(".exe", "").strip(),
        }

        if "chrome" in target_lower:
            aliases.update({"chrome", "google chrome"})
        if "notepad" in target_lower:
            aliases.update({"notepad"})
        if "edge" in target_lower:
            aliases.update({"edge", "microsoft edge"})

        return any(alias and alias in window for alias in aliases)

    def _run_ensure_app_open_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
    ) -> Tuple[bool, List[ScreenElement]]:
        """Deterministically execute app-open subtasks through WindowsTools."""
        target = self._extract_app_target(subtask)
        verify_condition = self._subtask_verify_condition(subtask)
        if self._looks_like_close_intent(subtask.description) or self._looks_like_close_intent(target):
            self._trace.log_event(
                source="pipeline",
                event_type="ensure_app_open_redirected_to_close",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "target": target,
                },
            )
            return self._run_close_window_subtask(
                subtask=subtask,
                step_num=step_num,
                total_steps=total_steps,
            )

        tools = self._get_tools()

        retry_count = 0
        while retry_count < MAX_RETRIES and not self.is_stopped():
            before_active = tools.get_active_window()
            before_window = (
                before_active.data.get("title")
                if before_active.success and before_active.data
                else "Unknown"
            )
            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="ensure_app_open_before",
                subtask=subtask.description,
            )

            self._trace.log_event(
                source="pipeline",
                event_type="ensure_app_open_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "target": target,
                    "before_window": before_window,
                    "before_screenshot": before_screenshot,
                },
            )

            open_result = tools.open_application(target)
            time.sleep(max(0.8, ACTION_SETTLE_TIME))

            after_active = tools.get_active_window()
            after_window = (
                after_active.data.get("title")
                if after_active.success and after_active.data
                else "Unknown"
            )

            if open_result.success and not self._window_matches_target(target, after_window):
                focus_result = tools.focus_window(target)
                self._trace.log_event(
                    source="pipeline",
                    event_type="ensure_app_open_focus_attempt",
                    payload={
                        "step_num": step_num,
                        "subtask": subtask.description,
                        "retry_count": retry_count,
                        "target": target,
                        "focus_success": focus_result.success,
                        "focus_message": focus_result.message,
                    },
                )
                time.sleep(0.3)
                after_active = tools.get_active_window()
                after_window = (
                    after_active.data.get("title")
                    if after_active.success and after_active.data
                    else "Unknown"
                )

            verified = bool(open_result.success) and self._window_matches_target(target, after_window)
            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="ensure_app_open_after",
                subtask=subtask.description,
            )

            self._trace.log_event(
                source="pipeline",
                event_type="ensure_app_open_result",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "target": target,
                    "open_success": open_result.success,
                    "open_message": open_result.message,
                    "after_window": after_window,
                    "verified": verified,
                    "after_screenshot": after_screenshot,
                },
            )

            if verified:
                time.sleep(ACTION_SETTLE_TIME)
                semantic_elements = self._get_eyes().scan(force=True)
                verified = self._semantic_verify_condition(verify_condition, semantic_elements)

            if verified:
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                return True, []

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self.on_step(step_num, f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})", "running")
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1

        return False, []

    def _run_close_window_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
    ) -> Tuple[bool, List[ScreenElement]]:
        """Deterministically close the active window and verify focus change."""
        tools = self._get_tools()
        target = self._extract_app_target(subtask)

        retry_count = 0
        while retry_count < MAX_RETRIES and not self.is_stopped():
            before_active = tools.get_active_window()
            before_window = (
                before_active.data.get("title")
                if before_active.success and before_active.data
                else "Unknown"
            )

            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="close_window_before",
                subtask=subtask.description,
            )

            self._trace.log_event(
                source="pipeline",
                event_type="close_window_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "target": target,
                    "before_window": before_window,
                    "before_screenshot": before_screenshot,
                },
            )

            close_result = tools.close_window()
            time.sleep(max(0.8, ACTION_SETTLE_TIME))

            after_active = tools.get_active_window()
            after_window = (
                after_active.data.get("title")
                if after_active.success and after_active.data
                else "Unknown"
            )

            still_target = self._window_matches_target(target, after_window)
            verified = bool(close_result.success) and (before_window != after_window or not still_target)

            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="close_window_after",
                subtask=subtask.description,
            )

            self._trace.log_event(
                source="pipeline",
                event_type="close_window_result",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "target": target,
                    "close_success": close_result.success,
                    "close_message": close_result.message,
                    "after_window": after_window,
                    "verified": verified,
                    "after_screenshot": after_screenshot,
                },
            )

            if verified:
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                return True, []

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self.on_step(step_num, f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})", "running")
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1

        return False, []

    def _extract_url_target(self, subtask: Subtask) -> str:
        """Extract URL from planner params/description for deterministic navigation."""
        params = subtask.params or {}
        for key in ("url", "target_url", "destination"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                url = value.strip()
                if not re.match(r"^https?://", url, flags=re.IGNORECASE):
                    url = f"https://{url}"
                return url

        match = re.search(
            r"(https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|io|dev|ai|co)(?:/\S*)?)",
            subtask.description,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""

        url = match.group(1).rstrip(".,")
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"https://{url}"
        return url

    def _run_navigate_to_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
    ) -> Tuple[bool, List[ScreenElement]]:
        """Deterministically navigate via address bar for browser tasks."""
        url = self._extract_url_target(subtask)
        verify_condition = self._subtask_verify_condition(subtask)
        if not url:
            return False, []

        tools = self._get_tools()
        eyes = self._get_eyes()

        retry_count = 0
        host = re.sub(r"^https?://", "", url.lower()).split("/")[0]
        host_parts = [part for part in host.split(".") if part]
        if host_parts and host_parts[0] == "www" and len(host_parts) > 1:
            host_key = host_parts[1]
        else:
            host_key = host_parts[0] if host_parts else ""

        while retry_count < MAX_RETRIES and not self.is_stopped():
            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="navigate_before",
                subtask=subtask.description,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="navigate_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "url": url,
                    "before_screenshot": before_screenshot,
                },
            )

            focus_result = tools.hotkey("ctrl", "l")
            type_result = tools.type_text(url)
            enter_result = tools.press_key("enter")
            time.sleep(max(1.0, ACTION_SETTLE_TIME))

            current_elements = eyes.scan(force=True)
            corpus = self._elements_text(current_elements)
            active_window = tools.get_active_window()
            active_title = (
                active_window.data.get("title", "")
                if active_window.success and active_window.data
                else ""
            ).lower()

            verified = bool(focus_result.success and type_result.success and enter_result.success) and (
                (host_key and host_key in corpus)
                or (host_key and host_key in active_title)
            )

            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="navigate_after",
                subtask=subtask.description,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="navigate_result",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "url": url,
                    "focus_success": focus_result.success,
                    "type_success": type_result.success,
                    "enter_success": enter_result.success,
                    "verified": verified,
                    "active_window": active_title,
                    "after_screenshot": after_screenshot,
                },
            )

            if verified:
                time.sleep(ACTION_SETTLE_TIME)
                semantic_elements = eyes.scan(force=True)
                verified = self._semantic_verify_condition(verify_condition, semantic_elements)

            if verified:
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                return True, []

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self.on_step(step_num, f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})", "running")
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1

        return False, []

    def _run_system_control_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
    ) -> Tuple[bool, List[ScreenElement]]:
        """Execute deterministic local system controls (volume/brightness)."""
        tools = self._get_tools()
        verify_condition = self._subtask_verify_condition(subtask)
        command = ""
        if isinstance(subtask.params, dict):
            command = str(subtask.params.get("command", "") or "").strip()
        if not command:
            command = subtask.description

        retry_count = 0
        while retry_count < MAX_RETRIES and not self.is_stopped():
            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="system_control_before",
                subtask=subtask.description,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="system_control_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "command": command,
                    "before_screenshot": before_screenshot,
                },
            )

            result = tools.control_system_setting(command)
            verified = bool(result.success)
            if verified:
                time.sleep(ACTION_SETTLE_TIME)
                semantic_elements = self._get_eyes().scan(force=True)
                verified = self._semantic_verify_condition(verify_condition, semantic_elements)

            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="system_control_after",
                subtask=subtask.description,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="system_control_result",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "command": command,
                    "result_success": result.success,
                    "result_message": result.message,
                    "verified": verified,
                    "after_screenshot": after_screenshot,
                },
            )

            if verified:
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                return True, []

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self.on_step(step_num, f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})", "running")
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1

        return False, []

    def _deterministic_decision_for_subtask(self, subtask: Subtask) -> Optional[Dict[str, Any]]:
        """Return a deterministic decision for strongly-typed subtasks when possible."""
        params = subtask.params if isinstance(subtask.params, dict) else {}
        action_type = (subtask.action_type or "").strip().lower()

        if action_type == "type_text":
            text = params.get("text")
            if isinstance(text, str) and text.strip():
                return {"action": "type", "text": text.strip()}

        if action_type == "wait":
            seconds = params.get("seconds", params.get("duration", 1))
            if isinstance(seconds, (int, float)):
                return {"action": "wait", "seconds": max(0.2, min(float(seconds), 15.0))}

        if action_type == "click_target":
            x = params.get("x")
            y = params.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return {"action": "click", "x": int(x), "y": int(y)}

        return None

    def _extract_type_payload(self, subtask: Subtask) -> str:
        """Extract textual payload for type_text conversion."""
        params = subtask.params if isinstance(subtask.params, dict) else {}
        text = params.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        desc = subtask.description or ""
        quoted = re.search(r"['\"]([^'\"]+)['\"]", desc)
        if quoted:
            return quoted.group(1).strip()

        match = re.search(r"\b(?:type|write|enter(?: text)?)\b\s+(.+)", desc, flags=re.IGNORECASE)
        if not match:
            return ""

        candidate = match.group(1).strip()
        candidate = re.split(r"\s+(?:in|into|on|to|inside)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        return candidate.strip(" .\"'")

    def _normalize_unknown_subtask(self, subtask: Subtask) -> bool:
        """Convert unknown subtasks into safe action types when possible."""
        action_type = (subtask.action_type or "").strip().lower()
        if action_type and action_type != "unknown":
            return True

        description = (subtask.description or "").strip()
        desc_lower = description.lower()
        params = subtask.params if isinstance(subtask.params, dict) else {}
        subtask.params = params

        if re.search(r"\b(close|exit|quit|minimi[sz]e)\b", desc_lower):
            subtask.action_type = "click_target"
            params.setdefault("target", self._extract_app_target(subtask) or "active window")
        elif re.search(r"\b(open|launch|start)\b", desc_lower):
            subtask.action_type = "launch_app"
            target = self._extract_app_target(subtask)
            if target:
                params.setdefault("app", target)
        else:
            url = self._extract_url_target(subtask)
            if url:
                subtask.action_type = "navigate_to"
                params.setdefault("url", url)
            elif re.search(r"\b(navigate|go to|visit|browse|open url)\b", desc_lower):
                subtask.action_type = "navigate_to"
            elif re.search(r"\b(type|write|enter text|enter)\b", desc_lower):
                subtask.action_type = "type_text"
                text = self._extract_type_payload(subtask)
                if text:
                    params.setdefault("text", text)
            elif re.search(r"\b(search|find|look up)\b", desc_lower):
                subtask.action_type = "search"
                params.setdefault("query", description)
            elif re.search(r"\b(wait|pause|hold)\b", desc_lower):
                subtask.action_type = "verify"
                match = re.search(r"(\d+(?:\.\d+)?)\s*second", desc_lower)
                if match:
                    params.setdefault("seconds", float(match.group(1)))
                else:
                    params.setdefault("seconds", 1.0)
            elif re.search(r"\b(mute|unmute|volume|brightness)\b", desc_lower):
                subtask.action_type = "click_target"
                params.setdefault("target", description)

        normalized = (subtask.action_type or "").strip().lower()
        if normalized and normalized != "unknown":
            self._trace.log_event(
                source="pipeline",
                event_type="unknown_subtask_normalized",
                payload={
                    "subtask": description,
                    "normalized_action_type": normalized,
                    "params": params,
                },
            )
            return True

        self._trace.log_event(
            source="pipeline",
            event_type="unknown_subtask_blocked",
            payload={
                "subtask": description,
                "reason": "no_safe_normalization",
            },
        )
        return False

    def run_subgoal(
        self,
        subgoal: str,
        step_num: int,
        total_steps: int,
        prefetched_elements: Optional[List[ScreenElement]] = None,
        tracker: Optional[SubgoalTracker] = None,
    ) -> Tuple[bool, List[ScreenElement]]:
        """
        Execute one subgoal with full pipeline optimization.

        Args:
            subgoal: The task to accomplish
            step_num: Current step number (1-indexed)
            total_steps: Total number of steps
            prefetched_elements: Elements from speculative pre-scan

        Returns:
            (success, elements_for_next_subgoal)
        """
        if self.is_stopped():
            return False, []

        self.total_subgoals += 1
        self.on_step(step_num, f"{subgoal}", "running")

        eyes = self._get_eyes()
        hands = self._get_hands()
        verifier = self._get_verifier()

        # ── EYES: get elements (use prefetched if available) ──────────────
        if prefetched_elements is not None and len(prefetched_elements) > 0:
            elements = prefetched_elements
            logger.info(f"Using prefetched elements ({len(elements)} total)")
        else:
            start_scan = time.time()
            elements = eyes.scan(force=True)
            scan_time = (time.time() - start_scan) * 1000
            logger.info(f"Fresh scan: {len(elements)} elements in {scan_time:.0f}ms")

        # Token compression: filter to elements near last action point
        filtered = eyes.filter_near(
            elements,
            cx=self._last_action_x,
            cy=self._last_action_y,
            radius=400  # Wider radius for better coverage
        )
        elements_str = eyes.elements_to_prompt_str(filtered)
        logger.info(f"Filtered to {len(filtered)} elements (was {len(elements)})")

        retry_count = 0
        success = False
        next_elements: List[ScreenElement] = []

        while retry_count < MAX_RETRIES and not self.is_stopped():
            active_window = self._trace.get_active_window_title()
            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="subgoal_before_decision",
                subtask=subgoal,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="subgoal_attempt_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subgoal": subgoal,
                    "retry_count": retry_count,
                    "active_window": active_window,
                    "elements_total": len(elements),
                    "elements_filtered": len(filtered),
                    "before_screenshot": before_screenshot,
                    "filtered_elements_preview": self._element_preview(filtered),
                },
            )

            # ── HANDS: Qwen decides action ────────────────────────────────
            start_decide = time.time()
            decision = hands.decide(subgoal, elements_str, expected_state="", tracker=tracker)
            decide_time = (time.time() - start_decide) * 1000
            logger.info(f"Qwen decision ({decide_time:.0f}ms): {decision}")
            self._trace.log_event(
                source="pipeline",
                event_type="subgoal_decision_made",
                payload={
                    "step_num": step_num,
                    "subgoal": subgoal,
                    "retry_count": retry_count,
                    "decision": decision,
                    "decision_latency_ms": round(decide_time, 2),
                },
            )

            # Check for completion
            if decision.get("action") == "done":
                self.on_step(step_num, f"✓ {subgoal}", "done")
                self.successful_subgoals += 1
                return True, []

            # ── Capture region BEFORE action (for pixel diff) ─────────────
            action_x = decision.get("x", self._last_action_x)
            action_y = decision.get("y", self._last_action_y)

            before_region = None
            if decision.get("action") == "click":
                before_region = verifier.capture_region(action_x, action_y)

            # ── SPECULATIVE PRE-SCAN: start scanning for NEXT subgoal ─────
            # Runs in Thread A while Thread B executes action below
            # This hides OmniParser latency (~80ms) behind action time (~50ms+)
            future_next_scan: Optional[Future] = None
            if step_num < total_steps:
                future_next_scan = self._executor.submit(eyes.scan, True)

            # ── EXECUTE action ─────────────────────────────────────────────
            start_exec = time.time()
            exec_success = hands.execute(decision)
            exec_time = (time.time() - start_exec) * 1000
            logger.info(f"Execute ({exec_time:.0f}ms): success={exec_success}")

            # Update last action coords for next cycle's filter
            if decision.get("action") == "click" and exec_success:
                self._last_action_x = action_x
                self._last_action_y = action_y

            # Wait for UI to settle
            time.sleep(ACTION_SETTLE_TIME)

            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="subgoal_after_action",
                subtask=subgoal,
            )

            # ── VERIFY: pixel diff ─────────────────────────────────────────
            after_region = None
            if decision.get("action") == "click":
                after_region = verifier.capture_region(action_x, action_y)

            action_name = str(decision.get("action", "")).lower()

            # For non-click actions, avoid auto-success on "wait" unless task itself is a wait step.
            if action_name != "click":
                explicit_wait_task = "wait" in subgoal.lower()
                if action_name == "wait":
                    changed = bool(exec_success) and explicit_wait_task
                else:
                    changed = bool(exec_success)
            else:
                changed = verifier.verify_action(
                    subgoal=subgoal,
                    decision=decision,
                    elements_seen=filtered,
                    before_region=before_region,
                    after_region=after_region,
                    verify_condition=subgoal,
                    retry_count=retry_count,
                    before_screenshot=before_screenshot,
                    after_screenshot=after_screenshot,
                    active_window=active_window,
                    extra_context={"step_num": step_num},
                )

            # Get speculative scan result (likely already done)
            if future_next_scan is not None:
                try:
                    next_elements = future_next_scan.result(timeout=3.0)
                except Exception as e:
                    logger.warning(f"Speculative scan failed: {e}")
                    next_elements = []

            made_progress = bool(changed) or (bool(exec_success) and action_name not in {"wait"})

            if made_progress:
                success = True
                self.on_step(step_num, f"✓ {subgoal}", "done")
                self.successful_subgoals += 1
                self._trace.log_event(
                    source="pipeline",
                    event_type="subgoal_attempt_succeeded",
                    payload={
                        "step_num": step_num,
                        "subgoal": subgoal,
                        "retry_count": retry_count,
                        "changed": bool(changed),
                        "exec_success": bool(exec_success),
                        "after_screenshot": after_screenshot,
                    },
                )
                break
            else:
                retry_count += 1
                self.total_retries += 1

                if retry_count < MAX_RETRIES:
                    logger.warning(f"Retry {retry_count}/{MAX_RETRIES} for: {subgoal}")
                    self._trace.log_event(
                        source="pipeline",
                        event_type="subgoal_retry",
                        payload={
                            "step_num": step_num,
                            "subgoal": subgoal,
                            "retry_count": retry_count,
                            "max_retries": MAX_RETRIES,
                        },
                    )
                    self.on_step(step_num, f"{subgoal} (retry {retry_count})", "running")

                    # Re-scan fresh for retry (don't use prefetch)
                    elements = eyes.scan(force=True)
                    filtered = eyes.filter_near(
                        elements, action_x, action_y, radius=400
                    )
                    elements_str = eyes.elements_to_prompt_str(filtered)
                else:
                    # Max retries hit - log it, move on
                    logger.error(f"FAILED after {MAX_RETRIES} retries: {subgoal}")
                    self.on_step(step_num, f"✗ FAILED: {subgoal}", "failed")
                    self.failed_subgoals += 1
                    self._trace.log_event(
                        source="pipeline",
                        event_type="subgoal_failed",
                        payload={
                            "step_num": step_num,
                            "subgoal": subgoal,
                            "retry_count": retry_count,
                            "max_retries": MAX_RETRIES,
                            "after_screenshot": after_screenshot,
                        },
                    )

        return success, next_elements

    def run_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
        prefetched_elements: Optional[List[ScreenElement]] = None,
        tracker: Optional[SubgoalTracker] = None,
    ) -> Tuple[bool, List[ScreenElement]]:
        """
        Execute a stateful planner subtask with reactive anomaly handling.
        """
        if self.is_stopped():
            return False, []

        if not self._normalize_unknown_subtask(subtask):
            self.total_subgoals += 1
            self.failed_subgoals += 1
            self.on_step(step_num, f"✗ FAILED: unsafe unknown step '{subtask.description}'", "failed")
            return False, []

        action_type = (subtask.action_type or "").strip().lower()
        if action_type in {"launch_app", "ensure_app_open", "open_application", "open_app"}:
            return self._run_ensure_app_open_subtask(
                subtask=subtask,
                step_num=step_num,
                total_steps=total_steps,
            )
        if action_type in {"close_window", "close_app", "exit_app"}:
            return self._run_close_window_subtask(
                subtask=subtask,
                step_num=step_num,
                total_steps=total_steps,
            )
        if action_type in {"navigate_to", "open_url"}:
            url_target = self._extract_url_target(subtask)
            if url_target:
                return self._run_navigate_to_subtask(
                    subtask=subtask,
                    step_num=step_num,
                    total_steps=total_steps,
                )
        if action_type in {"system_control", "os_control", "device_control"}:
            return self._run_system_control_subtask(
                subtask=subtask,
                step_num=step_num,
                total_steps=total_steps,
            )

        self.total_subgoals += 1
        self.on_step(step_num, subtask.description, "running")

        eyes = self._get_eyes()
        hands = self._get_hands()
        verifier = self._get_verifier()

        expected_state = self._expected_state_text(subtask)
        next_elements: List[ScreenElement] = []
        retry_count = 0
        success = False

        while retry_count < MAX_RETRIES and not self.is_stopped():
            if retry_count == 0 and prefetched_elements:
                elements = prefetched_elements
            else:
                elements = eyes.scan(force=True)

            filtered = eyes.filter_near(
                elements,
                cx=self._last_action_x,
                cy=self._last_action_y,
                radius=400,
            )
            elements_str = eyes.elements_to_prompt_str(filtered)

            active_window = self._trace.get_active_window_title()
            before_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="before_decision",
                subtask=subtask.description,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="subtask_attempt_started",
                payload={
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "active_window": active_window,
                    "expected_state": expected_state,
                    "elements_total": len(elements),
                    "elements_filtered": len(filtered),
                    "before_screenshot": before_screenshot,
                    "filtered_elements_preview": self._element_preview(filtered),
                },
            )

            decision_source = "deterministic"
            decision = self._deterministic_decision_for_subtask(subtask)
            if decision is None:
                decision_source = "qwen"
                start_decide = time.time()
                decision = hands.decide(
                    subgoal=subtask.description,
                    elements_str=elements_str,
                    expected_state=expected_state,
                    tracker=tracker,
                )
                decide_time = (time.time() - start_decide) * 1000
            else:
                decide_time = 0.0

            logger.info(f"Stateful decision ({decide_time:.0f}ms): {decision}")
            self._trace.log_event(
                source="pipeline",
                event_type="subtask_decision_made",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "decision_source": decision_source,
                    "decision": decision,
                    "decision_latency_ms": round(decide_time, 2),
                },
            )

            if decision.get("action") == "done":
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                done_screenshot = self._capture_step_screenshot(
                    step_num=step_num,
                    retry_count=retry_count,
                    tag="decision_done",
                    subtask=subtask.description,
                )
                self._trace.log_event(
                    source="pipeline",
                    event_type="subtask_completed_done_action",
                    payload={
                        "step_num": step_num,
                        "subtask": subtask.description,
                        "retry_count": retry_count,
                        "screenshot": done_screenshot,
                    },
                )
                return True, []

            action_x = decision.get("x", self._last_action_x)
            action_y = decision.get("y", self._last_action_y)

            before_region = None
            if decision.get("action") == "click":
                before_region = verifier.capture_region(action_x, action_y)

            future_next_scan: Optional[Future] = None
            if step_num < total_steps:
                future_next_scan = self._executor.submit(eyes.scan, True)

            exec_success = hands.execute(decision)
            self._trace.log_event(
                source="pipeline",
                event_type="subtask_action_executed",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "decision": decision,
                    "exec_success": bool(exec_success),
                },
            )
            if decision.get("action") == "click" and exec_success:
                self._last_action_x = action_x
                self._last_action_y = action_y

            time.sleep(ACTION_SETTLE_TIME)

            after_screenshot = self._capture_step_screenshot(
                step_num=step_num,
                retry_count=retry_count,
                tag="after_action",
                subtask=subtask.description,
            )

            if decision.get("action") == "click":
                after_region = verifier.capture_region(action_x, action_y)
                changed = verifier.verify_action(
                    subgoal=subtask.description,
                    decision=decision,
                    elements_seen=filtered,
                    before_region=before_region,
                    after_region=after_region,
                    verify_condition=self._subtask_verify_condition(subtask),
                    retry_count=retry_count,
                    before_screenshot=before_screenshot,
                    after_screenshot=after_screenshot,
                    active_window=active_window,
                    extra_context={
                        "step_num": step_num,
                        "expected_state": expected_state,
                    },
                )
            else:
                action_name = str(decision.get("action", "")).lower()
                if action_name == "wait" and action_type != "wait":
                    changed = False
                else:
                    changed = bool(exec_success)

            if future_next_scan is not None:
                try:
                    next_elements = future_next_scan.result(timeout=3.0)
                except Exception as e:
                    logger.warning(f"Speculative scan failed: {e}")
                    next_elements = []

            current_elements = eyes.scan(force=True)
            post_ok = self._postconditions_met(current_elements, subtask.postconditions)

            self._trace.log_event(
                source="pipeline",
                event_type="subtask_postcheck",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "changed": bool(changed),
                    "exec_success": bool(exec_success),
                    "postconditions_met": bool(post_ok),
                    "postconditions": subtask.postconditions,
                    "after_screenshot": after_screenshot,
                },
            )

            action_name = str(decision.get("action", "")).lower()
            made_progress = bool(changed) or (bool(exec_success) and action_name not in {"wait"})

            if made_progress and post_ok:
                success = True
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                self._trace.log_event(
                    source="pipeline",
                    event_type="subtask_attempt_succeeded",
                    payload={
                        "step_num": step_num,
                        "subtask": subtask.description,
                        "retry_count": retry_count,
                        "changed": bool(changed),
                        "exec_success": bool(exec_success),
                        "postconditions_met": bool(post_ok),
                    },
                )
                break

            # Reactive correction for state mismatch/anomaly.
            correction = hands.diagnose_anomaly(
                expected=subtask.postconditions or [expected_state],
                actual_elements=current_elements,
                failed_action=decision,
            )
            self._trace.log_event(
                source="pipeline",
                event_type="subtask_correction_requested",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "correction": correction,
                },
            )
            correction_success = hands.execute(correction)
            self._trace.log_event(
                source="pipeline",
                event_type="subtask_correction_executed",
                payload={
                    "step_num": step_num,
                    "subtask": subtask.description,
                    "retry_count": retry_count,
                    "correction": correction,
                    "success": bool(correction_success),
                },
            )
            if correction_success:
                if correction.get("action") == "click":
                    cx = correction.get("x", self._last_action_x)
                    cy = correction.get("y", self._last_action_y)
                    if cx and cy:
                        self._last_action_x = cx
                        self._last_action_y = cy
                time.sleep(ACTION_SETTLE_TIME)

                corrected_elements = eyes.scan(force=True)
                if self._postconditions_met(corrected_elements, subtask.postconditions):
                    success = True
                    self.on_step(step_num, f"✓ {subtask.description} (corrected)", "done")
                    self.successful_subgoals += 1
                    corrected_screenshot = self._capture_step_screenshot(
                        step_num=step_num,
                        retry_count=retry_count,
                        tag="after_correction",
                        subtask=subtask.description,
                    )
                    self._trace.log_event(
                        source="pipeline",
                        event_type="subtask_corrected_success",
                        payload={
                            "step_num": step_num,
                            "subtask": subtask.description,
                            "retry_count": retry_count,
                            "screenshot": corrected_screenshot,
                        },
                    )
                    break

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self._trace.log_event(
                    source="pipeline",
                    event_type="subtask_retry",
                    payload={
                        "step_num": step_num,
                        "subtask": subtask.description,
                        "retry_count": retry_count,
                        "max_retries": MAX_RETRIES,
                    },
                )
                self.on_step(
                    step_num,
                    f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})",
                    "running",
                )
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1
                failure_screenshot = self._capture_step_screenshot(
                    step_num=step_num,
                    retry_count=retry_count,
                    tag="subtask_failed",
                    subtask=subtask.description,
                )
                self._trace.log_event(
                    source="pipeline",
                    event_type="subtask_failed",
                    payload={
                        "step_num": step_num,
                        "subtask": subtask.description,
                        "retry_count": retry_count,
                        "max_retries": MAX_RETRIES,
                        "screenshot": failure_screenshot,
                    },
                )

        return success, next_elements

    def run_task_plan(
        self,
        task_plan: TaskPlan,
        on_status: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[int, str, str], None]] = None,
    ) -> str:
        """
        Execute a hierarchical TaskPlan with state-aware reactive verification.
        """
        status_cb = on_status or self.on_status
        step_cb = on_step or self.on_step

        orig_status = self.on_status
        orig_step = self.on_step
        self.on_status = status_cb
        self.on_step = step_cb

        try:
            self.reset()
            total_steps = len(task_plan.subtasks)
            if total_steps == 0:
                return "No subtasks generated for this plan."

            start_screenshot = self._trace.capture_screenshot(
                source="pipeline",
                tag="task_plan_start",
                extra_payload={"goal": task_plan.goal},
            )
            self._trace.start_run(
                source="pipeline",
                goal=task_plan.goal,
                metadata={
                    "initial_state": task_plan.initial_state,
                    "goal_state": task_plan.goal_state,
                    "subtask_count": total_steps,
                    "start_screenshot": start_screenshot,
                },
            )
            self._trace.log_event(
                source="pipeline",
                event_type="task_plan_started",
                payload={
                    "goal": task_plan.goal,
                    "initial_state": task_plan.initial_state,
                    "goal_state": task_plan.goal_state,
                    "intermediate_states": task_plan.intermediate_states,
                    "relevant_api_ids": [api.get("id") for api in task_plan.relevant_apis],
                    "subtasks": [subtask.to_dict() for subtask in task_plan.subtasks],
                },
            )

            status_cb(f"Initial state: {task_plan.initial_state or 'Unknown'}")
            if task_plan.relevant_apis:
                api_names = ", ".join(api.get("name", api.get("id", "api")) for api in task_plan.relevant_apis)
                status_cb(f"Relevant APIs: {api_names}")

            results: List[bool] = []
            prefetched: Optional[List[ScreenElement]] = None
            start_time = time.time()
            tracker = SubgoalTracker(completed=[], failed=[], current="")

            for i, subtask in enumerate(task_plan.subtasks, start=1):
                if self.is_stopped():
                    logger.info("Pipeline stopped by user")
                    self._trace.log_event(
                        source="pipeline",
                        event_type="task_plan_stopped",
                        payload={"step_num": i, "subtask": subtask.description},
                    )
                    break

                verify_condition = self._subtask_verify_condition(subtask)
                tracker.current = subtask.description

                # State gate: the previous subgoal's verify condition must still hold.
                if i > 1:
                    previous_subtask = task_plan.subtasks[i - 2]
                    previous_condition = self._subtask_verify_condition(previous_subtask)
                    if previous_condition and not self._semantic_verify_condition(previous_condition):
                        status_cb(
                            f"Previous subgoal state missing, re-running step {i-1}: {previous_subtask.description}"
                        )
                        prev_success, _ = self.run_subtask(
                            subtask=previous_subtask,
                            step_num=i - 1,
                            total_steps=total_steps,
                            prefetched_elements=None,
                            tracker=tracker,
                        )
                        if not prev_success:
                            results.append(False)
                            tracker.failed.append(previous_subtask.description)
                            self._trace.log_event(
                                source="pipeline",
                                event_type="task_plan_aborted_after_failure",
                                payload={
                                    "step_num": i - 1,
                                    "total_steps": total_steps,
                                    "subtask": previous_subtask.description,
                                },
                            )
                            break

                # Skip if already satisfied from current UI state.
                if verify_condition and self._semantic_verify_condition(verify_condition):
                    results.append(True)
                    tracker.completed.append(verify_condition)
                    self.successful_subgoals += 1
                    self.on_step(i, f"✓ {subtask.description} (already satisfied)", "done")
                    self._trace.log_event(
                        source="pipeline",
                        event_type="task_plan_step_skipped_verified",
                        payload={
                            "step_num": i,
                            "subtask": subtask.description,
                            "verify_condition": verify_condition,
                        },
                    )
                    continue

                status_cb(f"Step {i}/{total_steps}: {subtask.description}")
                self._trace.log_event(
                    source="pipeline",
                    event_type="task_plan_step_started",
                    payload={
                        "step_num": i,
                        "total_steps": total_steps,
                        "subtask": subtask.description,
                        "input_state": subtask.input_state,
                        "output_state": subtask.output_state,
                    },
                )
                success, next_prefetched = self.run_subtask(
                    subtask=subtask,
                    step_num=i,
                    total_steps=total_steps,
                    prefetched_elements=prefetched,
                    tracker=tracker,
                )
                results.append(success)
                prefetched = next_prefetched if next_prefetched else None
                if success:
                    if verify_condition:
                        tracker.completed.append(verify_condition)
                else:
                    tracker.failed.append(subtask.description)
                self._trace.log_event(
                    source="pipeline",
                    event_type="task_plan_step_finished",
                    payload={
                        "step_num": i,
                        "subtask": subtask.description,
                        "success": bool(success),
                    },
                )

                if not success:
                    status_cb(f"Stopping execution after failed step {i}/{total_steps}: {subtask.description}")
                    self._trace.log_event(
                        source="pipeline",
                        event_type="task_plan_aborted_after_failure",
                        payload={
                            "step_num": i,
                            "total_steps": total_steps,
                            "subtask": subtask.description,
                        },
                    )
                    break

            elapsed = time.time() - start_time
            completed = sum(1 for result in results if result)
            goal_state = task_plan.goal_state or task_plan.goal

            final_screenshot = self._trace.capture_screenshot(
                source="pipeline",
                tag="task_plan_end",
                extra_payload={"goal": task_plan.goal, "completed": completed, "total": total_steps},
            )
            self._trace.log_event(
                source="pipeline",
                event_type="task_plan_completed",
                payload={
                    "goal": task_plan.goal,
                    "goal_state": goal_state,
                    "completed_subtasks": completed,
                    "total_subtasks": total_steps,
                    "elapsed_seconds": round(elapsed, 3),
                    "successful_subgoals": self.successful_subgoals,
                    "failed_subgoals": self.failed_subgoals,
                    "total_retries": self.total_retries,
                    "final_screenshot": final_screenshot,
                },
            )

            if completed == total_steps:
                return (
                    f"Successfully reached goal state '{goal_state}' "
                    f"({completed}/{total_steps} subtasks) in {elapsed:.1f}s"
                )

            return (
                f"Reached partial state progress toward '{goal_state}': "
                f"{completed}/{total_steps} subtasks in {elapsed:.1f}s"
            )

        finally:
            self.on_status = orig_status
            self.on_step = orig_step

    def run_stateful_task(
        self,
        task_plan: TaskPlan,
        on_status: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[int, str, str], None]] = None,
    ) -> str:
        """Alias for architecture naming."""
        return self.run_task_plan(task_plan=task_plan, on_status=on_status, on_step=on_step)

    def run_task(
        self,
        subgoals: List[str],
        on_status: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[int, str, str], None]] = None
    ) -> str:
        """
        Run all subgoals with full pipeline.
        Prefetches next subgoal's scan during current subgoal's action.

        Args:
            subgoals: List of subgoal strings to execute
            on_status: Override status callback
            on_step: Override step callback

        Returns:
            Result summary string
        """
        # Use provided callbacks or defaults
        status_cb = on_status or self.on_status
        step_cb = on_step or self.on_step

        # Store original callbacks and use provided ones
        orig_status = self.on_status
        orig_step = self.on_step
        self.on_status = status_cb
        self.on_step = step_cb

        try:
            self.reset()
            results: List[bool] = []
            total_steps = len(subgoals)

            start_screenshot = self._trace.capture_screenshot(
                source="pipeline",
                tag="subgoal_list_start",
                extra_payload={"subgoal_count": total_steps},
            )
            self._trace.start_run(
                source="pipeline",
                goal=" ; ".join(subgoals[:5]),
                metadata={
                    "mode": "subgoal_list",
                    "subgoal_count": total_steps,
                    "subgoals": subgoals,
                    "start_screenshot": start_screenshot,
                },
            )

            # Initial scan (no prefetch for first subgoal)
            prefetched: Optional[List[ScreenElement]] = None

            start_time = time.time()

            for i, subgoal in enumerate(subgoals, start=1):
                if self.is_stopped():
                    logger.info("Pipeline stopped by user")
                    self._trace.log_event(
                        source="pipeline",
                        event_type="subgoal_list_stopped",
                        payload={"step_num": i, "subgoal": subgoal},
                    )
                    break

                status_cb(f"Step {i}/{total_steps}: {subgoal}")

                success, next_prefetched = self.run_subgoal(
                    subgoal=subgoal,
                    step_num=i,
                    total_steps=total_steps,
                    prefetched_elements=prefetched
                )
                results.append(success)
                self._trace.log_event(
                    source="pipeline",
                    event_type="subgoal_list_step_finished",
                    payload={
                        "step_num": i,
                        "subgoal": subgoal,
                        "success": bool(success),
                    },
                )

                # Hand off speculative scan to next iteration
                prefetched = next_prefetched if next_prefetched else None

            elapsed = time.time() - start_time
            completed = sum(results)
            total = len(subgoals)

            # Build result message
            if completed == total:
                result = f"Successfully completed {completed}/{total} subgoals in {elapsed:.1f}s"
            else:
                result = f"Completed {completed}/{total} subgoals in {elapsed:.1f}s ({total-completed} failed)"

            end_screenshot = self._trace.capture_screenshot(
                source="pipeline",
                tag="subgoal_list_end",
                extra_payload={"completed": completed, "total": total},
            )
            self._trace.log_event(
                source="pipeline",
                event_type="subgoal_list_completed",
                payload={
                    "completed": completed,
                    "total": total,
                    "elapsed_seconds": round(elapsed, 3),
                    "result": result,
                    "end_screenshot": end_screenshot,
                },
            )

            logger.info(result)
            return result

        finally:
            # Restore original callbacks
            self.on_status = orig_status
            self.on_step = orig_step

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "total_subgoals": self.total_subgoals,
            "successful": self.successful_subgoals,
            "failed": self.failed_subgoals,
            "total_retries": self.total_retries,
            "success_rate": (
                self.successful_subgoals / self.total_subgoals * 100
                if self.total_subgoals > 0 else 0
            )
        }

    def shutdown(self):
        """Shutdown the pipeline and release resources."""
        self.stop()
        self._executor.shutdown(wait=False)
        logger.info("Pipeline shutdown complete")


# Singleton instance
_pipeline: Optional[Pipeline] = None


def get_pipeline(**kwargs) -> Pipeline:
    """
    Get or create the global Pipeline instance.

    Args:
        **kwargs: Arguments passed to Pipeline constructor

    Returns:
        Global Pipeline instance
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(**kwargs)
    return _pipeline


def reset_pipeline():
    """Reset the global Pipeline instance."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.shutdown()
    _pipeline = None
