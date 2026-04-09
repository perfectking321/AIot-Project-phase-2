# VOXCODE Architecture Overhaul — Claude Code Prompt

You are working on VOXCODE, a voice-controlled Windows automation agent.
The codebase is in the current directory. Read all relevant files before making changes.

## GOAL
Implement a 3-model split architecture with pipeline parallelism, INT4 Qwen local
inference, speculative pre-scanning, token compression, and pixel-diff verification.
Target latency: ~2.2s end-to-end on consumer GPU.

---

## ARCHITECTURE TO IMPLEMENT
User voice → Whisper STT (preloaded)
↓
Groq 70B — ONE call only — produces subgoal list JSON
↓
┌─── for each subgoal (loop) ───────────────────────┐
│                                                    │
│  Thread A (parallel):          Thread B:           │
│  OmniParser scan screen   →   Qwen 7B INT4 local  │
│  YOLO + Florence-2 icons       gets: subgoal +     │
│  returns: [{label, center,     filtered elements   │
│  bbox, confidence}]            returns: action JSON │
│                    ↘          ↙                    │
│                  pyautogui.execute()               │
│                       ↓                            │
│              pixel_diff_check() ~5ms               │
│                       ↓                            │
│           pass → next subgoal (speculative scan    │
│                  already running in Thread A)      │
│           fail → audit_log + retry via Qwen only  │
│                  (max 3 retries, no Groq call)     │
└────────────────────────────────────────────────────┘

---

## FILE CHANGES REQUIRED

### 1. CREATE: `agent/eyes.py` (NEW FILE)

OmniParser wrapper with Florence-2 ENABLED and element filtering.

```python
"""
VOXCODE Eyes — OmniParser with Florence-2 icon captioning enabled.
Returns filtered, labeled screen elements for Qwen consumption.
"""
import time
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("voxcode.eyes")

@dataclass
class ScreenElement:
    id: int
    label: str
    center: tuple      # (x, y)
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float
    element_type: str  # 'text', 'icon', 'button'

class Eyes:
    """
    Wraps OmniParser. Florence-2 captioning ENABLED so icons get real names
    instead of 'icon'. Implements element filtering by proximity to reduce
    Qwen prompt tokens (token compression formula).
    """
    def __init__(self):
        from agent.omniparser import OmniParser
        # Enable Florence-2 caption model for icon identification
        self.parser = OmniParser(
            use_caption_model=True,  # CHANGED from False to True
            preload=True
        )
        self._last_parsed = None
        self._last_parse_time = 0
        self.cache_ttl = 0.1  # 100ms cache — avoid double-scanning same frame

    def scan(self, force=False) -> list[ScreenElement]:
        """
        Scan screen. Returns all elements as ScreenElement list.
        Cached for cache_ttl seconds to support speculative pre-scanning.
        """
        now = time.time()
        if not force and self._last_parsed and (now - self._last_parse_time) < self.cache_ttl:
            return self._last_parsed

        parsed = self.parser.parse_screen()
        elements = []
        for e in parsed.elements:
            elements.append(ScreenElement(
                id=e.id,
                label=e.label,
                center=e.center,
                bbox=e.bbox,
                confidence=e.confidence,
                element_type=e.element_type
            ))

        self._last_parsed = elements
        self._last_parse_time = now
        logger.info(f"Eyes: scanned {len(elements)} elements")
        return elements

    def filter_near(self, elements: list[ScreenElement],
                    cx: int, cy: int, radius: int = 300) -> list[ScreenElement]:
        """
        Token compression: return only elements within radius pixels of (cx, cy).
        Reduces Qwen prompt from ~800 tokens (50 elements) to ~200 tokens (10 elements).
        If no elements in radius, return closest 10 overall.
        """
        nearby = [e for e in elements
                  if abs(e.center[0] - cx) <= radius and abs(e.center[1] - cy) <= radius]
        if len(nearby) >= 3:
            return nearby[:15]
        # fallback: sort all by distance, return closest 10
        elements_sorted = sorted(
            elements,
            key=lambda e: ((e.center[0]-cx)**2 + (e.center[1]-cy)**2)**0.5
        )
        return elements_sorted[:10]

    def elements_to_prompt_str(self, elements: list[ScreenElement]) -> str:
        """Format elements for Qwen prompt. Minimal tokens."""
        lines = []
        for e in elements:
            lines.append(f"- {e.element_type} '{e.label}' at {e.center} (conf:{e.confidence:.2f})")
        return "\n".join(lines)


# Singleton
_eyes = None
def get_eyes() -> Eyes:
    global _eyes
    if _eyes is None:
        _eyes = Eyes()
    return _eyes
```

---

### 2. CREATE: `agent/hands.py` (NEW FILE)

Qwen 7B local inference for action decisions. Uses Ollama with INT4 model.

```python
"""
VOXCODE Hands — Qwen 7B INT4 local model for per-subgoal action decisions.
Receives ONE subgoal + filtered screen elements. Returns one action JSON.
Never sees the full task. Never calls Groq.
"""
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger("voxcode.hands")

# Ollama INT4 model — pull with: ollama pull qwen2.5:7b-instruct-q4_K_M
QWEN_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

SYSTEM_PROMPT = """You are a Windows UI automation agent.
You receive ONE task and a list of visible screen elements with their positions.
You must output ONE action as JSON. No explanation. JSON only.

Available actions:
{"action":"click","x":640,"y":80}
{"action":"type","text":"hello world"}
{"action":"press","key":"enter"}
{"action":"hotkey","keys":["ctrl","l"]}
{"action":"scroll","amount":-3}
{"action":"wait","seconds":1}
{"action":"done"}

Rules:
- Pick the element that best matches the task
- Use exact coordinates from the element list
- If task is already complete, return {"action":"done"}
- If no matching element found, return {"action":"wait","seconds":1}
"""

class Hands:
    """
    Qwen 7B INT4 local inference.
    Fast because: INT4 quantized, tiny output (15 tokens), filtered input (~200 tokens).
    """
    def __init__(self, model: str = QWEN_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
        self._verify_model()

    def _verify_model(self):
        """Check model available. Log warning if not."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(QWEN_MODEL.split(":")[0] in m for m in models):
                logger.warning(
                    f"Model {QWEN_MODEL} not found. "
                    f"Run: ollama pull {QWEN_MODEL}\n"
                    f"Available: {models}"
                )
        except Exception as e:
            logger.warning(f"Could not verify Ollama models: {e}")

    def decide(self, subgoal: str, elements_str: str) -> dict:
        """
        Given subgoal + screen elements string, return action dict.
        Target: ~150-200ms on consumer GPU with INT4.
        """
        prompt = f"""TASK: {subgoal}

SCREEN ELEMENTS:
{elements_str}

Output ONE action JSON:"""

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,      # deterministic
                        "num_predict": 40,        # max 40 tokens (action JSON is ~15)
                        "num_ctx": 512,           # small context = fast prefill
                    }
                },
                timeout=10
            )
            response_text = resp.json()["response"].strip()
            logger.debug(f"Qwen raw: {response_text}")

            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response_text[start:end])

        except json.JSONDecodeError as e:
            logger.warning(f"Qwen JSON parse error: {e} | raw: {response_text}")
        except Exception as e:
            logger.error(f"Qwen inference error: {e}")

        return {"action": "wait", "seconds": 1}

    def execute(self, decision: dict) -> bool:
        """Execute pyautogui action from decision dict. Returns True if action taken."""
        import pyautogui
        import time

        action = decision.get("action", "wait")

        try:
            if action == "click":
                pyautogui.click(decision["x"], decision["y"])
                logger.info(f"Click at ({decision['x']}, {decision['y']})")
                return True
            elif action == "type":
                pyautogui.write(decision["text"], interval=0.03)
                logger.info(f"Type: {decision['text']}")
                return True
            elif action == "press":
                pyautogui.press(decision["key"])
                logger.info(f"Press: {decision['key']}")
                return True
            elif action == "hotkey":
                pyautogui.hotkey(*decision["keys"])
                logger.info(f"Hotkey: {decision['keys']}")
                return True
            elif action == "scroll":
                pyautogui.scroll(decision.get("amount", -3))
                return True
            elif action == "wait":
                time.sleep(decision.get("seconds", 1))
                return True
            elif action == "done":
                return True
        except Exception as e:
            logger.error(f"Execute error: {e}")

        return False


_hands = None
def get_hands() -> Hands:
    global _hands
    if _hands is None:
        _hands = Hands()
    return _hands
```

---

### 3. CREATE: `agent/verifier.py` (NEW FILE)

Pixel diff verification + audit logging. No API calls on failure.

```python
"""
VOXCODE Verifier — Fast pixel-diff screen change detection + audit logging.
No LLM calls. No Groq on failure. Just log and retry via Qwen.
"""
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger("voxcode.verifier")

AUDIT_LOG_PATH = Path("audit_log.jsonl")

class Verifier:
    """
    Pixel diff check: compare region around action point before/after.
    If unchanged → action likely failed → log to audit file.
    Fast: ~5ms per check.
    """
    def __init__(self, change_threshold: float = 5.0, check_radius: int = 80):
        self.threshold = change_threshold
        self.radius = check_radius

    def capture_region(self, cx: int, cy: int) -> Optional[np.ndarray]:
        """Capture screen region around point."""
        try:
            import pyautogui
            from PIL import Image
            x1 = max(0, cx - self.radius)
            y1 = max(0, cy - self.radius)
            x2 = cx + self.radius
            y2 = cy + self.radius
            shot = pyautogui.screenshot(region=(x1, y1, x2-x1, y2-y1))
            return np.array(shot)
        except Exception as e:
            logger.warning(f"Region capture failed: {e}")
            return None

    def did_screen_change(self, before: np.ndarray, after: np.ndarray) -> bool:
        """
        Pixel diff formula: mean absolute difference of region.
        Returns True if screen changed (action likely succeeded).
        Threshold=5.0 means avg pixel shift of 5/255 across region.
        """
        if before is None or after is None:
            return True  # assume changed if we can't check
        if before.shape != after.shape:
            return True
        diff = np.abs(before.astype(float) - after.astype(float)).mean()
        logger.debug(f"Pixel diff: {diff:.2f} (threshold: {self.threshold})")
        return diff > self.threshold

    def audit_log(self, entry: dict):
        """
        Append audit entry to JSONL file. Never raises.
        Each line = one action attempt with full context.
        """
        try:
            entry["timestamp"] = datetime.now().isoformat()
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")

    def verify_action(
        self,
        subgoal: str,
        decision: dict,
        elements_seen: list,
        before_region: Optional[np.ndarray],
        after_region: Optional[np.ndarray],
        retry_count: int = 0
    ) -> bool:
        """
        Full verify cycle. Returns True if action succeeded.
        Logs to audit regardless of outcome.
        """
        changed = self.did_screen_change(before_region, after_region)

        self.audit_log({
            "subgoal": subgoal,
            "decision": decision,
            "elements_count": len(elements_seen),
            "pixel_changed": changed,
            "retry_count": retry_count,
            "action": decision.get("action"),
            "coords": {
                "x": decision.get("x"),
                "y": decision.get("y")
            }
        })

        if not changed:
            logger.warning(
                f"Action may have failed (no pixel change): "
                f"{decision} for subgoal '{subgoal}'"
            )

        return changed


_verifier = None
def get_verifier() -> Verifier:
    global _verifier
    if _verifier is None:
        _verifier = Verifier()
    return _verifier
```

---

### 4. CREATE: `agent/pipeline.py` (NEW FILE)

Main orchestration loop with pipeline parallelism + speculative pre-scanning.
This is the core of the new architecture.

```python
"""
VOXCODE Pipeline — 3-model orchestration with pipeline parallelism.

Flow per subgoal:
  Thread A: OmniParser scan (runs ahead speculatively)
  Thread B: Qwen decide + execute + pixel verify
  Overlap: while B executes action, A scans next subgoal's screen

Latency target: ~242ms per cycle on consumer GPU (RTX 3060)
"""
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

from agent.eyes import get_eyes, ScreenElement
from agent.hands import get_hands
from agent.verifier import get_verifier

logger = logging.getLogger("voxcode.pipeline")

MAX_RETRIES = 3
ACTION_SETTLE_TIME = 0.3  # seconds to wait for UI to update after action


class Pipeline:
    """
    Orchestrates Eyes → Hands → Verifier with pipeline parallelism.

    Key optimizations implemented:
    1. Amdahl: Qwen INT4 = 3.73x faster inference
    2. Pipeline parallelism: OmniParser scan overlaps with Qwen inference
    3. Speculative pre-scan: scan N+1 during action execution of N
    4. Token compression: filter to ~10 nearest elements (~200 tokens vs 800)
    5. Pixel diff verify: 5ms check, no API call on failure
    """

    def __init__(
        self,
        on_status=None,
        on_step=None
    ):
        self.eyes = get_eyes()
        self.hands = get_hands()
        self.verifier = get_verifier()
        self.on_status = on_status or (lambda msg: None)
        self.on_step = on_step or (lambda step, msg, status: None)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop = False
        # Track last action coords for proximity-based filtering
        self._last_action_x = 960   # default screen center
        self._last_action_y = 540

    def stop(self):
        self._stop = True

    def run_subgoal(self, subgoal: str, step_num: int,
                    prefetched_elements: Optional[list] = None) -> tuple[bool, list]:
        """
        Execute one subgoal with full pipeline optimization.
        Returns (success, elements_for_next_subgoal).

        prefetched_elements: elements from speculative pre-scan of THIS subgoal.
        Returns elements scanned speculatively for NEXT subgoal.
        """
        if self._stop:
            return False, []

        self.on_step(step_num, f"Subgoal: {subgoal}", "running")

        # ── EYES: get elements (use prefetched if available) ──────────────
        if prefetched_elements is not None:
            elements = prefetched_elements
            logger.info(f"Using prefetched elements ({len(elements)} total)")
        else:
            elements = self.eyes.scan()

        # Token compression: filter to elements near last action point
        filtered = self.eyes.filter_near(
            elements,
            cx=self._last_action_x,
            cy=self._last_action_y,
            radius=350
        )
        elements_str = self.eyes.elements_to_prompt_str(filtered)
        logger.info(f"Filtered to {len(filtered)} elements for Qwen (was {len(elements)})")

        retry_count = 0
        success = False
        next_elements = []

        while retry_count < MAX_RETRIES and not self._stop:

            # ── HANDS: Qwen decides action ────────────────────────────────
            decision = self.hands.decide(subgoal, elements_str)
            logger.info(f"Qwen decision: {decision}")

            if decision.get("action") == "done":
                self.on_step(step_num, subgoal, "done")
                return True, []

            # ── Capture region BEFORE action (for pixel diff) ─────────────
            action_x = decision.get("x", self._last_action_x)
            action_y = decision.get("y", self._last_action_y)
            before_region = self.verifier.capture_region(action_x, action_y)

            # ── SPECULATIVE PRE-SCAN: start scanning for NEXT subgoal ─────
            # Runs in Thread A while Thread B executes action below
            # This hides OmniParser latency (~80ms) behind action time (~50ms+)
            future_next_scan: Future = self._executor.submit(self.eyes.scan)

            # ── EXECUTE action ─────────────────────────────────────────────
            self.hands.execute(decision)

            # Update last action coords for next cycle's filter
            if decision.get("action") == "click":
                self._last_action_x = action_x
                self._last_action_y = action_y

            # Wait for UI to settle
            time.sleep(ACTION_SETTLE_TIME)

            # ── VERIFY: pixel diff ─────────────────────────────────────────
            after_region = self.verifier.capture_region(action_x, action_y)

            changed = self.verifier.verify_action(
                subgoal=subgoal,
                decision=decision,
                elements_seen=filtered,
                before_region=before_region,
                after_region=after_region,
                retry_count=retry_count
            )

            # Get speculative scan result (likely already done)
            try:
                next_elements = future_next_scan.result(timeout=2.0)
            except Exception:
                next_elements = []

            if changed:
                success = True
                self.on_step(step_num, subgoal, "done")
                break
            else:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logger.warning(f"Retry {retry_count}/{MAX_RETRIES} for: {subgoal}")
                    self.on_step(step_num, f"{subgoal} (retry {retry_count})", "running")
                    # Re-scan fresh for retry (don't use prefetch)
                    elements = self.eyes.scan(force=True)
                    filtered = self.eyes.filter_near(
                        elements, action_x, action_y, radius=350
                    )
                    elements_str = self.eyes.elements_to_prompt_str(filtered)
                else:
                    # Max retries hit — log it, move on
                    logger.error(f"FAILED after {MAX_RETRIES} retries: {subgoal}")
                    self.on_step(step_num, f"FAILED: {subgoal}", "failed")
                    # next_elements already captured above

        return success, next_elements

    def run_task(self, subgoals: list[str],
                 on_status=None, on_step=None) -> str:
        """
        Run all subgoals with full pipeline.
        Prefetches next subgoal's scan during current subgoal's action.
        """
        on_status = on_status or self.on_status
        on_step = on_step or self.on_step

        self._stop = False
        results = []

        # Initial scan (no prefetch for first subgoal)
        prefetched = None

        for i, subgoal in enumerate(subgoals):
            if self._stop:
                break

            on_status(f"Step {i+1}/{len(subgoals)}: {subgoal}")

            success, next_prefetched = self.run_subgoal(
                subgoal=subgoal,
                step_num=i+1,
                prefetched_elements=prefetched
            )
            results.append(success)

            # Hand off speculative scan to next iteration
            prefetched = next_prefetched if next_prefetched else None

        completed = sum(results)
        total = len(subgoals)
        return f"Completed {completed}/{total} subgoals"


# Singleton
_pipeline = None
def get_pipeline(**kwargs) -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(**kwargs)
    return _pipeline
```

---

### 5. MODIFY: `brain/llm.py`

Add `plan_subgoals()` method to GroqClient for structured subgoal output.
Add this method inside the `GroqClient` class:

```python
def plan_subgoals(self, command: str, screen_context: str = "") -> list[str]:
    """
    One Groq call → returns list of subgoal strings.
    Groq's ONLY job in the new architecture. Called once per command.
    """
    prompt = f"""You are a Windows automation planner.
Convert the user command into 3-7 ordered subgoals.
Each subgoal must be ONE atomic UI action (click, type, navigate, etc).
Respond ONLY with a JSON array of strings. No explanation.

COMMAND: {command}
SCREEN CONTEXT: {screen_context if screen_context else "Unknown"}

Example output:
["open Chrome browser", "click address bar", "type youtube.com", "press enter", "click search bar", "type cats", "press enter"]

Subgoals:"""

    try:
        response = self.generate(prompt)
        content = response.content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            import json
            subgoals = json.loads(content[start:end])
            if isinstance(subgoals, list) and all(isinstance(s, str) for s in subgoals):
                return subgoals
    except Exception as e:
        import logging
        logging.getLogger("voxcode.llm").error(f"plan_subgoals failed: {e}")

    # Fallback: treat whole command as one subgoal
    return [command]
```

---

### 6. MODIFY: `tui/app.py`

Replace `_execute_desktop_command` method body with the new pipeline.
Find the method and replace its entire body:

```python
async def _execute_desktop_command(self, command: str) -> None:
    """Execute using new 3-model pipeline: Groq plan → Qwen execute → pixel verify"""
    import asyncio
    from agent.pipeline import get_pipeline

    self.set_status("Planning subgoals...")
    self.log_status("Groq 70B → planning subgoals (one API call)")

    # ── GROQ: plan subgoals (ONE call) ────────────────────────────────────
    try:
        from brain.llm import get_llm_client
        llm = get_llm_client()

        # Get screen context for better planning
        active_win = ""
        try:
            import pygetwindow as gw
            w = gw.getActiveWindow()
            if w:
                active_win = w.title
        except Exception:
            pass

        subgoals = await asyncio.to_thread(
            llm.plan_subgoals, command, active_win
        )

        self.log_info(f"[dim]Subgoals ({len(subgoals)}):[/]")
        for i, sg in enumerate(subgoals, 1):
            self.log_info(f"  [dim]{i}.[/] {sg}")
        self.log_info("")

    except Exception as e:
        self.log_error(f"Planning failed: {e}")
        self.set_status("Ready")
        return

    # ── PIPELINE: Qwen + OmniParser per subgoal ───────────────────────────
    self.set_status("Executing...")
    self.log_status("Pipeline: OmniParser eyes → Qwen 7B hands → pixel verify")

    def on_step(step_num: int, msg: str, status: str):
        self.log_agent_step(step_num, msg, status)

    def on_status_cb(msg: str):
        self.set_status(msg)

    try:
        pipeline = get_pipeline(on_status=on_status_cb, on_step=on_step)
        result = await asyncio.to_thread(
            pipeline.run_task,
            subgoals,
            on_status_cb,
            on_step
        )

        self.log_info("")
        if "Failed" not in result:
            self.log_success(f"✓ {result}")
        else:
            self.log_warning(f"⚠ {result}")

    except Exception as e:
        self.log_error(f"Pipeline error: {e}")
        import logging
        logging.getLogger("voxcode.tui").error(f"Pipeline error", exc_info=True)

    self.set_status("Ready")
```

---

### 7. MODIFY: `agent/omniparser.py`

Find `use_caption_model: bool = False` in the `__init__` signature and change default to True:

```python
# BEFORE:
def __init__(self, ..., use_caption_model: bool = False):

# AFTER:
def __init__(self, ..., use_caption_model: bool = True):
```

Also find this line in `get_omniparser()`:
```python
# BEFORE:
_omniparser_instance = OmniParser(preload=False)

# AFTER:
_omniparser_instance = OmniParser(preload=True, use_caption_model=True)
```

---

### 8. CREATE: `setup_qwen_int4.sh` (helper script)

```bash
#!/bin/bash
# Pull Qwen 7B INT4 model for local inference
# INT4 = 3.73x faster than FP16, fits in 4-5GB VRAM

echo "Pulling Qwen2.5 7B INT4 quantized model..."
ollama pull qwen2.5:7b-instruct-q4_K_M

echo ""
echo "Done. Verify with: ollama run qwen2.5:7b-instruct-q4_K_M 'say hi'"
echo "Expected speed: 80-194 tokens/s depending on GPU"
echo "  RTX 4090: ~194 tok/s → ~77ms per action"
echo "  RTX 3060: ~80 tok/s  → ~187ms per action"
echo "  CPU only: ~12 tok/s  → ~1250ms per action (too slow)"
```

---

### 9. CREATE: `test_pipeline.py` (validation test)

```python
"""
Test the new pipeline architecture end-to-end.
Run: python test_pipeline.py
"""
import time
import sys

def test():
    print("=" * 60)
    print("VOXCODE Pipeline Test")
    print("=" * 60)

    # Test 1: Eyes
    print("\n[1] Testing Eyes (OmniParser + Florence-2)...")
    t0 = time.time()
    from agent.eyes import get_eyes
    eyes = get_eyes()
    elements = eyes.scan()
    t1 = time.time()
    print(f"    Scan time: {(t1-t0)*1000:.0f}ms")
    print(f"    Elements: {len(elements)}")
    named = [e for e in elements if e.label != "icon"]
    print(f"    Named (not 'icon'): {len(named)}")
    for e in named[:5]:
        print(f"      - '{e.label}' at {e.center}")

    # Test 2: Token compression
    print("\n[2] Testing token compression...")
    filtered = eyes.filter_near(elements, cx=960, cy=540, radius=350)
    full_str = eyes.elements_to_prompt_str(elements)
    filtered_str = eyes.elements_to_prompt_str(filtered)
    print(f"    Full elements: {len(elements)} → ~{len(full_str.split())*1.3:.0f} tokens")
    print(f"    Filtered:      {len(filtered)} → ~{len(filtered_str.split())*1.3:.0f} tokens")

    # Test 3: Hands (Qwen)
    print("\n[3] Testing Hands (Qwen 7B INT4)...")
    from agent.hands import get_hands
    hands = get_hands()
    t0 = time.time()
    decision = hands.decide("click the search bar", filtered_str)
    t1 = time.time()
    print(f"    Qwen inference: {(t1-t0)*1000:.0f}ms")
    print(f"    Decision: {decision}")

    # Test 4: Verifier
    print("\n[4] Testing Verifier (pixel diff)...")
    from agent.verifier import get_verifier
    verifier = get_verifier()
    t0 = time.time()
    region = verifier.capture_region(960, 540)
    t1 = time.time()
    print(f"    Region capture: {(t1-t0)*1000:.0f}ms")
    import time as t
    t.sleep(0.1)
    region2 = verifier.capture_region(960, 540)
    changed = verifier.did_screen_change(region, region2)
    print(f"    Pixel diff (idle): changed={changed} (expect False)")

    # Test 5: Full pipeline timing
    print("\n[5] Full pipeline timing (dry run)...")
    from agent.pipeline import get_pipeline
    pipeline = get_pipeline()
    t0 = time.time()
    result = pipeline.run_task(["move mouse slightly"])
    t1 = time.time()
    print(f"    One subgoal wall time: {(t1-t0)*1000:.0f}ms")
    print(f"    Result: {result}")

    print("\n" + "=" * 60)
    print("Test complete. Check audit_log.jsonl for action records.")

if __name__ == "__main__":
    test()
```

---

## REQUIRED DEPENDENCIES

Add to `requirements.txt` if not present:
Already in requirements.txt but verify versions:
ultralytics>=8.0.0      # YOLO
easyocr>=1.7.0          # OCR
transformers>=4.36.0    # Florence-2 (uncomment in requirements.txt)
accelerate>=0.25.0      # Florence-2 acceleration (uncomment)
torch>=2.0.0

Uncomment these lines in `requirements.txt`:
Optional: Icon captioning (Florence-2)
transformers>=4.36.0
accelerate>=0.25.0
→ Remove the `#` from both lines.

---

## SETUP STEPS AFTER CODE CHANGES

Tell Claude Code to also add these instructions to README.md under a new section "## New Architecture Setup":

```markdown
## New Architecture Setup

1. Pull Qwen INT4 model:
   bash setup_qwen_int4.sh

2. Install Florence-2 deps:
   pip install transformers accelerate

3. Download OmniParser weights (Florence-2 captioner):
   python -c "from agent.omniparser import get_omniparser; get_omniparser(preload=True)"

4. Test pipeline:
   python test_pipeline.py

5. Run VOXCODE:
   python main.py

Expected latency (RTX 3060 / similar):
  - STT: ~400ms (one time)
  - Groq plan: ~500ms (one time per command)
  - Per subgoal: ~242ms (pipelined)
  - 5-subgoal task total: ~2.2 seconds
```

---

## WHAT NOT TO CHANGE

- `voice/stt.py` — leave as is
- `voice/hotkey.py` — leave as is
- `voice/tts.py` — leave as is
- `agent/tools.py` — leave as is (still used by browser agent)
- `agent/skills/` — leave as is
- `tui/app.py` `_execute_browser_command` — leave as is
- `config.py` — leave as is
- All `test_*.py` files except adding `test_pipeline.py`

---

## VALIDATION AFTER CHANGES

Run in order:
```bash
python test_pipeline.py          # verify all 3 models work
python main.py                   # run full app
# say: "open notepad"           # simple test
# say: "search cats on YouTube" # full pipeline test
# check: audit_log.jsonl        # verify actions are logged
```

Expected audit_log.jsonl entry per action:
```json
{"subgoal":"click search bar","decision":{"action":"click","x":640,"y":80},"elements_count":9,"pixel_changed":true,"retry_count":0,"action":"click","coords":{"x":640,"y":80},"timestamp":"2026-04-08T..."}