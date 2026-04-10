#!/usr/bin/env python3
"""
Debug helper for VOXCODE STT -> Planner (Qwen) flow.

This script lets you inspect:
1) STT output text (from --audio or --text)
2) Exact planner prompt sent to Qwen
3) Raw Qwen response
4) Parsed states + subtasks

Examples:
  python debug_qwen_stt_goals.py --text "open google and search cats"
  python debug_qwen_stt_goals.py --audio sample.wav
  python debug_qwen_stt_goals.py --text "open youtube" --json-out qwen_debug.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def detect_active_window() -> str:
    """Best-effort active window title detection."""
    try:
        import pygetwindow as gw  # type: ignore

        window = gw.getActiveWindow()
        if window and getattr(window, "title", ""):
            return window.title
    except Exception:
        pass
    return "No active window detected"


def transcribe_audio_file(audio_path: Path) -> Dict[str, Any]:
    """Transcribe WAV audio file via project SpeechToText."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    from voice.stt import SpeechToText

    audio_data = audio_path.read_bytes()
    stt = SpeechToText(preload=True)
    result = stt.transcribe(audio_data)
    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration": result.duration,
    }


def task_plan_to_dict(plan: Any) -> Dict[str, Any]:
    """Serialize TaskPlan without assuming internals beyond public fields."""
    subtasks: List[Dict[str, Any]] = []
    for subtask in plan.subtasks:
        if hasattr(subtask, "to_dict"):
            subtasks.append(subtask.to_dict())
        else:
            subtasks.append(
                {
                    "id": getattr(subtask, "id", None),
                    "description": getattr(subtask, "description", ""),
                    "action_type": getattr(subtask, "action_type", ""),
                    "preconditions": getattr(subtask, "preconditions", []),
                    "postconditions": getattr(subtask, "postconditions", []),
                    "params": getattr(subtask, "params", {}),
                    "input_state": getattr(subtask, "input_state", ""),
                    "output_state": getattr(subtask, "output_state", ""),
                }
            )

    return {
        "goal": getattr(plan, "goal", ""),
        "initial_state": getattr(plan, "initial_state", ""),
        "intermediate_states": getattr(plan, "intermediate_states", []),
        "goal_state": getattr(plan, "goal_state", ""),
        "relevant_apis": getattr(plan, "relevant_apis", []),
        "subtasks": subtasks,
    }


def print_header(title: str) -> None:
    line = "=" * 88
    print("\n" + line)
    print(title)
    print(line)


def _split_model_name(name: str) -> tuple[str, str]:
    """Split Ollama model name into (base, tag)."""
    cleaned = (name or "").strip().lower()
    parts = cleaned.split(":", 1)
    base = parts[0] if parts else ""
    tag = parts[1] if len(parts) > 1 else ""
    return base, tag


def _is_model_available(requested: str, available: List[str]) -> bool:
    """Check model availability with tag-aware matching."""
    requested = (requested or "").strip().lower()
    if not requested:
        return False

    if requested in [m.lower() for m in available]:
        return True

    req_base, req_tag = _split_model_name(requested)
    for candidate in available:
        cand_base, cand_tag = _split_model_name(candidate)
        if cand_base != req_base:
            continue
        if not req_tag:
            return True
        if req_tag == cand_tag:
            return True
    return False


def ensure_planner_model_available(planner: Any) -> Dict[str, Any]:
    """Switch planner.llm.model to an installed model if configured one is missing."""
    info: Dict[str, Any] = {
        "requested_model": getattr(planner.llm, "model", ""),
        "selected_model": getattr(planner.llm, "model", ""),
        "switched": False,
        "available_models": [],
    }

    if not hasattr(planner.llm, "list_models"):
        return info

    try:
        models = planner.llm.list_models()
        available = [m.get("name", "").strip() for m in models if isinstance(m, dict) and m.get("name")]
        info["available_models"] = available

        requested = info["requested_model"]
        if _is_model_available(str(requested), available):
            return info

        req_base, _ = _split_model_name(str(requested))
        same_family = [m for m in available if _split_model_name(m)[0] == req_base]
        qwen_models = [m for m in available if "qwen" in m.lower()]

        fallback = ""
        if same_family:
            fallback = same_family[0]
        elif qwen_models:
            fallback = qwen_models[0]
        elif available:
            fallback = available[0]

        if fallback and fallback != requested and hasattr(planner.llm, "model"):
            planner.llm.model = fallback
            info["selected_model"] = fallback
            info["switched"] = True

    except Exception:
        return info

    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect STT -> Qwen planner prompt/response/states for a task"
    )
    parser.add_argument("--text", type=str, default="", help="Input text (as if from STT)")
    parser.add_argument("--audio", type=str, default="", help="Path to WAV audio for STT")
    parser.add_argument(
        "--screen-context",
        type=str,
        default="",
        help="Override current screen context (window title)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Hide full planner prompt in terminal output",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to save full debug output as JSON",
    )
    args = parser.parse_args()

    try:
        # 1) Resolve text input (direct text or STT from audio)
        stt_meta: Dict[str, Any] = {}
        if args.audio:
            stt_meta = transcribe_audio_file(Path(args.audio))
            user_text = (stt_meta.get("text") or "").strip()
        else:
            user_text = (args.text or "").strip()

        if not user_text:
            user_text = input("Enter task text (STT output): ").strip()

        if not user_text:
            print("No text provided. Exiting.")
            return 1

        # 2) Use same planner stack as app
        from brain.planner import get_planner

        planner = get_planner()
        model_resolution = ensure_planner_model_available(planner)
        screen_context = (args.screen_context or "").strip() or detect_active_window()

        relevant_apis = planner.api_registry.find_relevant(user_text)
        prompt = planner._render_prompt(
            planner.DECOMPOSE_PROMPT,
            {
                "goal": user_text,
                "screen_context": screen_context,
                "relevant_apis": json.dumps(relevant_apis, indent=2) if relevant_apis else "[]",
            },
        )

        # 3) Raw model call (same path as planner, but exposed here)
        try:
            llm_response = planner.llm.generate(prompt)
            raw_output = llm_response.content
        except Exception as exc:
            # Surface model/HTTP details clearly for debugging model availability.
            details = str(exc)
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                body_text = getattr(response_obj, "text", "")
                if body_text:
                    details = f"{details} | body: {body_text}"
            raise RuntimeError(details) from exc

        # 4) Parse using planner parser to inspect states/subgoals
        parsed_plan = planner._parse_task_plan(
            response=raw_output,
            goal=user_text,
            screen_context=screen_context,
            relevant_apis=relevant_apis,
        )

        plan_dict = task_plan_to_dict(parsed_plan)

        # 5) Print readable debug report
        print_header("VOXCODE QWEN PLANNER DEBUG")
        print(f"Input text (from STT): {user_text}")
        print(f"Screen context: {screen_context}")

        model_name = getattr(planner.llm, "model", "unknown")
        provider_name = planner.llm.__class__.__name__
        print(f"Planner provider: {provider_name}")
        print(f"Planner model: {model_name}")
        if model_resolution.get("switched"):
            print(
                "Planner model switched: "
                f"{model_resolution.get('requested_model')} -> {model_resolution.get('selected_model')}"
            )

        if stt_meta:
            print("STT metadata:")
            print(json.dumps(stt_meta, indent=2, ensure_ascii=False))

        if not args.no_prompt:
            print_header("PROMPT SENT TO QWEN")
            print(prompt)

        print_header("RAW QWEN RESPONSE")
        print(raw_output)

        print_header("PARSED PLAN (STATES + SUBGOALS)")
        print(json.dumps(plan_dict, indent=2, ensure_ascii=False))

        if args.json_out:
            output_path = Path(args.json_out)
            output = {
                "input": {
                    "text": user_text,
                    "screen_context": screen_context,
                    "stt": stt_meta,
                },
                "planner": {
                    "provider": provider_name,
                    "model": model_name,
                    "relevant_apis": relevant_apis,
                    "prompt": prompt,
                    "raw_qwen_response": raw_output,
                    "parsed_plan": plan_dict,
                },
            }
            output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nSaved debug JSON to: {output_path}")

        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
