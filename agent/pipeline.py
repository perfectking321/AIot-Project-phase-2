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
from typing import Optional, List, Callable, Tuple, Any, Dict

from agent.eyes import get_eyes, Eyes, ScreenElement
from agent.hands import get_hands, Hands
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

    def _elements_text(self, elements: List[ScreenElement]) -> str:
        """Build normalized text corpus from detected elements."""
        return " ".join((e.label or "").lower() for e in elements)

    def _postconditions_met(self, elements: List[ScreenElement], postconditions: List[str]) -> bool:
        """
        Heuristic postcondition check against currently visible labels.
        """
        if not postconditions:
            return True

        corpus = self._elements_text(elements)
        stopwords = {"the", "a", "an", "is", "are", "to", "of", "for", "and", "in", "on"}
        checkable = 0
        matched = 0

        for condition in postconditions:
            cond = (condition or "").strip().lower()
            if not cond:
                continue

            if cond in corpus:
                checkable += 1
                matched += 1
                continue

            keywords = [tok for tok in re.findall(r"[a-z0-9]+", cond) if tok not in stopwords and len(tok) > 2]
            if not keywords:
                continue

            checkable += 1
            if any(keyword in corpus for keyword in keywords):
                matched += 1

        if checkable == 0:
            return True

        # Be tolerant to OCR/parsing noise: require majority of checkable conditions.
        required = max(1, (checkable + 1) // 2)
        return matched >= required

    def _expected_state_text(self, subtask: Subtask) -> str:
        """Compact expected-state text for the decision model."""
        if subtask.output_state:
            return subtask.output_state
        if subtask.postconditions:
            return ", ".join(subtask.postconditions)
        return ""

    def run_subgoal(
        self,
        subgoal: str,
        step_num: int,
        total_steps: int,
        prefetched_elements: Optional[List[ScreenElement]] = None
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
            # ── HANDS: Qwen decides action ────────────────────────────────
            start_decide = time.time()
            decision = hands.decide(subgoal, elements_str, expected_state="")
            decide_time = (time.time() - start_decide) * 1000
            logger.info(f"Qwen decision ({decide_time:.0f}ms): {decision}")

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

            # ── VERIFY: pixel diff ─────────────────────────────────────────
            after_region = None
            if decision.get("action") == "click":
                after_region = verifier.capture_region(action_x, action_y)

            # For non-click actions (type, press, etc.), assume success
            if decision.get("action") != "click":
                changed = True
            else:
                changed = verifier.verify_action(
                    subgoal=subgoal,
                    decision=decision,
                    elements_seen=filtered,
                    before_region=before_region,
                    after_region=after_region,
                    retry_count=retry_count
                )

            # Get speculative scan result (likely already done)
            if future_next_scan is not None:
                try:
                    next_elements = future_next_scan.result(timeout=3.0)
                except Exception as e:
                    logger.warning(f"Speculative scan failed: {e}")
                    next_elements = []

            if changed or exec_success:
                success = True
                self.on_step(step_num, f"✓ {subgoal}", "done")
                self.successful_subgoals += 1
                break
            else:
                retry_count += 1
                self.total_retries += 1

                if retry_count < MAX_RETRIES:
                    logger.warning(f"Retry {retry_count}/{MAX_RETRIES} for: {subgoal}")
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

        return success, next_elements

    def run_subtask(
        self,
        subtask: Subtask,
        step_num: int,
        total_steps: int,
        prefetched_elements: Optional[List[ScreenElement]] = None,
    ) -> Tuple[bool, List[ScreenElement]]:
        """
        Execute a stateful planner subtask with reactive anomaly handling.
        """
        if self.is_stopped():
            return False, []

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

            start_decide = time.time()
            decision = hands.decide(
                subgoal=subtask.description,
                elements_str=elements_str,
                expected_state=expected_state,
            )
            decide_time = (time.time() - start_decide) * 1000
            logger.info(f"Stateful decision ({decide_time:.0f}ms): {decision}")

            if decision.get("action") == "done":
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
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
            if decision.get("action") == "click" and exec_success:
                self._last_action_x = action_x
                self._last_action_y = action_y

            time.sleep(ACTION_SETTLE_TIME)

            if decision.get("action") == "click":
                after_region = verifier.capture_region(action_x, action_y)
                changed = verifier.verify_action(
                    subgoal=subtask.description,
                    decision=decision,
                    elements_seen=filtered,
                    before_region=before_region,
                    after_region=after_region,
                    retry_count=retry_count,
                )
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

            if (changed or exec_success) and post_ok:
                success = True
                self.on_step(step_num, f"✓ {subtask.description}", "done")
                self.successful_subgoals += 1
                break

            # Reactive correction for state mismatch/anomaly.
            correction = hands.diagnose_anomaly(
                expected=subtask.postconditions or [expected_state],
                actual_elements=current_elements,
                failed_action=decision,
            )
            correction_success = hands.execute(correction)
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
                    break

            retry_count += 1
            self.total_retries += 1
            if retry_count < MAX_RETRIES:
                self.on_step(
                    step_num,
                    f"{subtask.description} (retry {retry_count}/{MAX_RETRIES})",
                    "running",
                )
            else:
                self.on_step(step_num, f"✗ FAILED: {subtask.description}", "failed")
                self.failed_subgoals += 1

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

            status_cb(f"Initial state: {task_plan.initial_state or 'Unknown'}")
            if task_plan.relevant_apis:
                api_names = ", ".join(api.get("name", api.get("id", "api")) for api in task_plan.relevant_apis)
                status_cb(f"Relevant APIs: {api_names}")

            results: List[bool] = []
            prefetched: Optional[List[ScreenElement]] = None
            start_time = time.time()

            for i, subtask in enumerate(task_plan.subtasks, start=1):
                if self.is_stopped():
                    logger.info("Pipeline stopped by user")
                    break

                status_cb(f"Step {i}/{total_steps}: {subtask.description}")
                success, next_prefetched = self.run_subtask(
                    subtask=subtask,
                    step_num=i,
                    total_steps=total_steps,
                    prefetched_elements=prefetched,
                )
                results.append(success)
                prefetched = next_prefetched if next_prefetched else None

            elapsed = time.time() - start_time
            completed = sum(1 for result in results if result)
            goal_state = task_plan.goal_state or task_plan.goal

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

            # Initial scan (no prefetch for first subgoal)
            prefetched: Optional[List[ScreenElement]] = None

            start_time = time.time()

            for i, subgoal in enumerate(subgoals, start=1):
                if self.is_stopped():
                    logger.info("Pipeline stopped by user")
                    break

                status_cb(f"Step {i}/{total_steps}: {subgoal}")

                success, next_prefetched = self.run_subgoal(
                    subgoal=subgoal,
                    step_num=i,
                    total_steps=total_steps,
                    prefetched_elements=prefetched
                )
                results.append(success)

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
