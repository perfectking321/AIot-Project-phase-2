"""
VOXCODE Hands - Qwen 7B INT4 local model for per-subgoal action decisions.
Receives ONE subgoal + filtered screen elements. Returns one action JSON.
Never sees the full task.

Uses Ollama for local inference with INT4 quantized model for speed.
Falls back to Groq if Ollama is not available.
"""
import json
import logging
import time
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger("voxcode.hands")

# Ollama model - use what's available locally
# Check with: ollama list
# Common options:
#   - qwen2.5:7b (recommended - balanced)
#   - qwen2.5:3b (faster, less accurate)
#   - qwen2.5:14b (slower, more accurate)
QWEN_MODEL = "qwen2.5:7b"  # Changed to match installed model
OLLAMA_HOST = "http://localhost:11434"

# System prompt for action decisions - REACTIVE and CONTEXT-AWARE
SYSTEM_PROMPT = """You are a Windows UI automation agent. You MUST look at the SCREEN ELEMENTS to understand what is currently visible, then decide the best action to accomplish the TASK.

CRITICAL: Always analyze the screen elements first! The screen may show unexpected dialogs, popups, or states.

Available actions:
{"action":"click","x":640,"y":80}
{"action":"type","text":"hello world"}
{"action":"press","key":"enter"}
{"action":"hotkey","keys":["ctrl","l"]}
{"action":"hotkey","keys":["win"]}
{"action":"scroll","amount":-3}
{"action":"wait","seconds":1}
{"action":"done"}

DECISION RULES:

1. FIRST, analyze what's on screen from the elements list:
   - Is there a profile selection dialog? Click on a profile name.
   - Is there a popup/dialog blocking? Handle it first.
   - Is the expected UI visible? Proceed with the task.

2. For "click" actions:
   - Find the element that matches what you need to click
   - Use the EXACT coordinates from the element (shown as "at (x, y)")
   - Example: element "[5] link 'Aryan' at (344, 379)" → {"action":"click","x":344,"y":379}

3. Common screen states to handle:
   - "Who's using Chrome?" / profile selector → Click on a profile name (e.g., "Aryan", "Guest mode")
   - "Sign in" dialogs → Look for "Skip" or "Not now" or just close it
   - Cookie consent → Click "Accept" or "Reject all"
   - YouTube search → Click the search box (usually has "Search" label)

4. If task says "open Chrome" but you see Chrome profile selector:
   - Click on ANY profile to continue (first one is fine)

5. If task says "click search box" and you see YouTube:
   - Look for elements containing "Search" and click it

6. If you cannot find the right element, return {"action":"wait","seconds":2}

Output ONLY the JSON action, nothing else."""


class Hands:
    """
    Qwen 7B INT4 local inference for action decisions.

    Fast because:
    - INT4 quantized model (3.73x faster than FP16)
    - Tiny output (~15 tokens for action JSON)
    - Filtered input (~200 tokens vs 800)
    - Small context window (512 tokens)

    Target latency: ~150-200ms on consumer GPU (RTX 3060)

    Falls back to Groq if Ollama is unavailable.
    """

    def __init__(
        self,
        model: str = QWEN_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = 30,
        use_groq_fallback: bool = False  # Disabled by default - use local Qwen only
    ):
        """
        Initialize Hands with Ollama backend.

        Args:
            model: Ollama model name (default: qwen2.5:7b-instruct-q4_K_M)
            host: Ollama API host
            timeout: Request timeout in seconds
            use_groq_fallback: Fall back to Groq if Ollama unavailable
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.use_groq_fallback = use_groq_fallback
        self._available_models: Optional[list] = None
        self._use_groq: bool = False
        self._groq_client = None

        # Verify model is available
        if not self._verify_model():
            if use_groq_fallback:
                logger.info("Falling back to Groq for action decisions")
                self._use_groq = True
                self._init_groq()
            else:
                logger.error("Ollama not available and Groq fallback disabled")

    def _verify_model(self) -> bool:
        """Check if model is available. Log warning if not."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            self._available_models = models

            # Check if our model (or a variant) is available
            model_base = self.model.split(":")[0]
            found = any(model_base in m for m in models)

            if not found:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Run: ollama pull {self.model}\n"
                    f"Available models: {models}"
                )
                # Try to find any qwen model as fallback
                qwen_models = [m for m in models if "qwen" in m.lower()]
                if qwen_models:
                    self.model = qwen_models[0]
                    logger.info(f"Using fallback model: {self.model}")
                    return True
                return False
            return True

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.host}.\n"
                "Make sure Ollama is running: ollama serve"
            )
            return False
        except Exception as e:
            logger.warning(f"Could not verify Ollama models: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        if self._use_groq:
            return self._groq_client is not None
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except:
            return False

    def _init_groq(self):
        """Initialize Groq client for fallback."""
        try:
            from brain.llm import GroqClient
            self._groq_client = GroqClient()
            logger.info("Groq client initialized for Hands fallback")
        except Exception as e:
            logger.error(f"Failed to initialize Groq fallback: {e}")
            self._groq_client = None

    def _decide_via_groq(self, subgoal: str, elements_str: str, expected_state: str = "") -> Dict[str, Any]:
        """Use Groq API for action decision (fallback when Ollama unavailable)."""
        if self._groq_client is None:
            return {"action": "wait", "seconds": 1}

        prompt = f"""TASK: {subgoal}
EXPECTED STATE: {expected_state or "Not specified"}

SCREEN ELEMENTS:
{elements_str}

Output ONE action JSON. ONLY JSON, no explanation:"""

        start_time = time.time()
        try:
            response = self._groq_client.generate(prompt, system=SYSTEM_PROMPT)
            response_text = response.content.strip()

            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Groq inference: {elapsed:.0f}ms | raw: {response_text[:100]}")

            action = self._parse_action_json(response_text)
            if action:
                logger.info(f"Groq decision ({elapsed:.0f}ms): {action}")
                return action

        except Exception as e:
            logger.error(f"Groq inference error: {e}")

        return {"action": "wait", "seconds": 1}

    def _preprocess_subgoal(self, subgoal: str) -> Optional[Dict[str, Any]]:
        """
        Handle common subgoal patterns directly without LLM.
        This speeds up obvious actions and ensures reliability.

        Returns action dict if handled, None if LLM needed.
        """
        subgoal_lower = subgoal.lower().strip()

        # Handle "press Windows key" / "open Start menu"
        if "windows key" in subgoal_lower or "start menu" in subgoal_lower:
            return {"action": "hotkey", "keys": ["win"]}

        # Handle "press enter"
        if subgoal_lower in ["press enter", "press enter to launch", "press enter key"]:
            return {"action": "press", "key": "enter"}

        # Handle "press tab"
        if "press tab" in subgoal_lower:
            return {"action": "press", "key": "tab"}

        # Handle "press escape"
        if "press escape" in subgoal_lower or "press esc" in subgoal_lower:
            return {"action": "press", "key": "escape"}

        # Handle "use Ctrl+L" (focus address bar)
        if "ctrl+l" in subgoal_lower or "ctrl l" in subgoal_lower:
            return {"action": "hotkey", "keys": ["ctrl", "l"]}

        # Handle "use Ctrl+T" (new tab)
        if "ctrl+t" in subgoal_lower or "ctrl t" in subgoal_lower:
            return {"action": "hotkey", "keys": ["ctrl", "t"]}

        # Handle "type [text]" - extract the text
        if subgoal_lower.startswith("type "):
            text = subgoal[5:].strip()
            # Remove quotes if present
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            if text:
                return {"action": "type", "text": text}

        # Handle "wait" subgoals
        if "wait" in subgoal_lower:
            # Extract seconds if specified
            import re
            match = re.search(r'(\d+)\s*second', subgoal_lower)
            seconds = int(match.group(1)) if match else 2
            return {"action": "wait", "seconds": seconds}

        # Not a simple pattern - needs LLM
        return None

    def decide(self, subgoal: str, elements_str: str, expected_state: str = "") -> Dict[str, Any]:
        """
        Given subgoal + screen elements string, return action dict.
        First tries pattern matching, then uses Ollama/Groq for complex decisions.

        Args:
            subgoal: The specific task to accomplish (e.g., "click search bar")
            elements_str: Formatted string of visible screen elements
            expected_state: State expected after this action/subtask

        Returns:
            Action dictionary, e.g., {"action": "click", "x": 640, "y": 80}
        """
        # First, try to handle common patterns directly (fast path)
        direct_action = self._preprocess_subgoal(subgoal)
        if direct_action is not None:
            logger.info(f"Direct action (pattern match): {direct_action}")
            return direct_action

        # Use Groq fallback if Ollama unavailable
        if self._use_groq:
            return self._decide_via_groq(subgoal, elements_str, expected_state=expected_state)

        prompt = f"""TASK: {subgoal}
EXPECTED STATE: {expected_state or "Not specified"}

SCREEN ELEMENTS:
{elements_str}

Output ONE action JSON:"""

        start_time = time.time()

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,       # Low for deterministic output
                        "num_predict": 50,        # Max 50 tokens (action JSON is ~15-20)
                        "num_ctx": 1024,          # Small context = fast prefill
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "").strip()

            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Qwen inference: {elapsed:.0f}ms | raw: {response_text[:100]}")

            # Extract JSON from response
            action = self._parse_action_json(response_text)

            if action:
                logger.info(f"Qwen decision ({elapsed:.0f}ms): {action}")
                return action

        except requests.exceptions.Timeout:
            logger.warning(f"Qwen timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            logger.error("Qwen: Ollama connection failed")
        except json.JSONDecodeError as e:
            logger.warning(f"Qwen JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Qwen inference error: {e}")

        # Fallback: wait action
        return {"action": "wait", "seconds": 1}

    def _elements_to_text(self, actual_elements: Any) -> str:
        """Normalize element structures into compact prompt text."""
        if isinstance(actual_elements, str):
            return actual_elements

        lines: List[str] = []

        if isinstance(actual_elements, list):
            for idx, elem in enumerate(actual_elements):
                if isinstance(elem, dict):
                    label = elem.get("label", "unknown")
                    elem_type = elem.get("type") or elem.get("element_type", "element")
                    center = elem.get("center", ("?", "?"))
                    lines.append(f"[{idx}] {elem_type} '{label}' at {center}")
                else:
                    label = getattr(elem, "label", "unknown")
                    elem_type = getattr(elem, "element_type", "element")
                    center = getattr(elem, "center", ("?", "?"))
                    lines.append(f"[{idx}] {elem_type} '{label}' at {center}")

        return "\n".join(lines) if lines else "(no elements)"

    def diagnose_anomaly(
        self,
        expected: Any,
        actual_elements: Any,
        failed_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ask the action model for a corrective step when state mismatch occurs.
        """
        expected_text = expected if isinstance(expected, str) else ", ".join([str(e) for e in expected or []])
        elements_text = self._elements_to_text(actual_elements)
        failed_text = json.dumps(failed_action) if failed_action else "none"

        prompt = f"""ANOMALY DETECTED.
EXPECTED STATE: {expected_text or "unknown"}
FAILED ACTION: {failed_text}

CURRENT SCREEN ELEMENTS:
{elements_text}

Return ONE corrective action JSON only."""

        try:
            if self._use_groq:
                response = self._groq_client.generate(prompt, system=SYSTEM_PROMPT)
                action = self._parse_action_json(response.content.strip())
                if action:
                    return action
            else:
                resp = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "system": SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 60,
                            "num_ctx": 1024,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                        },
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                action = self._parse_action_json(resp.json().get("response", "").strip())
                if action:
                    return action
        except Exception as e:
            logger.warning(f"Anomaly diagnosis failed: {e}")

        return {"action": "wait", "seconds": 1}

    def _parse_action_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract action JSON from model response.
        Handles various response formats (raw JSON, markdown blocks, etc.)

        Args:
            response_text: Raw model response

        Returns:
            Parsed action dict or None
        """
        text = response_text.strip()

        # Try direct JSON parse first
        try:
            action = json.loads(text)
            if isinstance(action, dict) and "action" in action:
                return action
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                json_str = text[start:end]
                action = json.loads(json_str)
                if isinstance(action, dict) and "action" in action:
                    return action
            except json.JSONDecodeError:
                pass

        # Try to extract from code block
        if "```" in text:
            import re
            code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if code_match:
                try:
                    action = json.loads(code_match.group(1))
                    if isinstance(action, dict) and "action" in action:
                        return action
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Could not parse action from: {text[:100]}")
        return None

    def execute(self, decision: Dict[str, Any]) -> bool:
        """
        Execute pyautogui action from decision dict.

        Args:
            decision: Action dictionary from decide()

        Returns:
            True if action was executed successfully
        """
        import pyautogui
        import time as time_module

        action = decision.get("action", "wait")

        try:
            if action == "click":
                x, y = decision.get("x", 0), decision.get("y", 0)
                if x > 0 and y > 0:
                    pyautogui.click(x, y)
                    logger.info(f"Executed: click at ({x}, {y})")
                    return True
                else:
                    logger.warning(f"Invalid click coordinates: ({x}, {y})")
                    return False

            elif action == "type":
                text = decision.get("text", "")
                if text:
                    # Use typewrite for ASCII, write for unicode
                    pyautogui.write(text, interval=0.02)
                    logger.info(f"Executed: type '{text[:30]}...'")
                    return True
                return False

            elif action == "press":
                key = decision.get("key", "")
                if key:
                    pyautogui.press(key)
                    logger.info(f"Executed: press '{key}'")
                    return True
                return False

            elif action == "hotkey":
                keys = decision.get("keys", [])
                if keys:
                    pyautogui.hotkey(*keys)
                    logger.info(f"Executed: hotkey {keys}")
                    return True
                return False

            elif action == "scroll":
                amount = decision.get("amount", -3)
                pyautogui.scroll(amount)
                logger.info(f"Executed: scroll {amount}")
                return True

            elif action == "wait":
                seconds = decision.get("seconds", 1)
                time_module.sleep(seconds)
                logger.info(f"Executed: wait {seconds}s")
                return True

            elif action == "done":
                logger.info("Executed: done (task complete)")
                return True

            else:
                logger.warning(f"Unknown action: {action}")
                return False

        except Exception as e:
            logger.error(f"Execute error: {e}")
            return False


# Singleton instance
_hands: Optional[Hands] = None


def get_hands(model: str = QWEN_MODEL, host: str = OLLAMA_HOST) -> Hands:
    """
    Get or create the global Hands instance.

    Args:
        model: Ollama model name
        host: Ollama API host

    Returns:
        Global Hands instance
    """
    global _hands
    if _hands is None:
        _hands = Hands(model=model, host=host)
    return _hands


def reset_hands():
    """Reset the global Hands instance (useful for testing)."""
    global _hands
    _hands = None
