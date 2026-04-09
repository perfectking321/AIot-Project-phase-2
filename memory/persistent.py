"""
Persistent memory vault utilities.

Backs `memory/vault/` files used by the hybrid architecture.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("voxcode.memory.persistent")

from config import config


class PersistentMemoryVault:
    """Read/write helpers for persistent vault files."""

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = Path(vault_dir or config.memory.vault_dir)
        self.user_file = self.vault_dir / "USER.yaml"
        self.history_file = self.vault_dir / "HISTORY.jsonl"
        self.apps_file = self.vault_dir / "APPS.md"
        self.failures_file = self.vault_dir / "failures.jsonl"
        self.ensure_initialized()

    def ensure_initialized(self) -> None:
        """Ensure vault directory and base files exist."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_file(
            self.user_file,
            "name: default\npreferences:\n  language: en\n  voice_feedback: false\n",
        )
        self._ensure_file(self.history_file, "")
        self._ensure_file(self.failures_file, "")
        self._ensure_file(
            self.apps_file,
            "# Known Applications\n\nThis file is updated as apps are observed/opened.\n",
        )

    def append_history(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Append one event to HISTORY.jsonl."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }
        self._append_jsonl(self.history_file, payload)

    def append_failure(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Append one failure event to failures.jsonl."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": message,
            "data": data or {},
        }
        self._append_jsonl(self.failures_file, payload)

    def record_app(self, app_name: str) -> None:
        """Ensure app name exists in APPS.md list."""
        app_name = (app_name or "").strip()
        if not app_name:
            return

        try:
            content = self.apps_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed reading APPS.md: %s", exc)
            return

        marker = f"- {app_name}"
        if marker in content:
            return

        try:
            with self.apps_file.open("a", encoding="utf-8") as handle:
                if not content.endswith("\n"):
                    handle.write("\n")
                handle.write(f"{marker}\n")
        except Exception as exc:
            logger.warning("Failed writing APPS.md: %s", exc)

    def _ensure_file(self, path: Path, default_content: str) -> None:
        if path.exists():
            return
        try:
            path.write_text(default_content, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed creating %s: %s", path, exc)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed appending JSONL to %s: %s", path, exc)
