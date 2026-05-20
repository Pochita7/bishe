"""Structured execution tracing for MASTeam runs."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str, fallback: str = "task") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._")
    return (text or fallback)[:120]


def _preview(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if response is None:
        return calls
    for msg in getattr(response, "messages", []) or []:
        for content in getattr(msg, "contents", []) or []:
            if getattr(content, "type", None) != "function_call":
                continue
            item: Dict[str, Any] = {
                "name": str(getattr(content, "name", None) or "unknown"),
            }
            for attr in ("arguments", "args", "input"):
                val = getattr(content, attr, None)
                if val:
                    item[attr] = _preview(val, 800)
                    break
            calls.append(item)
    return calls


def response_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_details", None) or {}
    return {
        "input_tokens": int(usage.get("input_token_count", 0) or 0),
        "output_tokens": int(usage.get("output_token_count", 0) or 0),
        "total_tokens": int(usage.get("total_token_count", 0) or 0),
    }


class TraceLogger:
    """Append-only JSONL trace logger, disabled unless explicitly enabled."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        trace_dir: str = "mas_traces",
        run_id: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.trace_dir = trace_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = ""
        self.task_id = ""
        self._seq = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        *,
        enabled: Optional[bool] = None,
        trace_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "TraceLogger":
        resolved_enabled = _truthy(os.environ.get("MAS_TRACE_ENABLED"))
        if enabled is not None:
            resolved_enabled = bool(enabled)
        return cls(
            enabled=resolved_enabled,
            trace_dir=trace_dir or os.environ.get("MAS_TRACE_DIR") or "mas_traces",
            run_id=run_id or os.environ.get("MAS_TRACE_RUN_ID"),
        )

    def start_task(self, task_id: str, task: str) -> None:
        if not self.enabled:
            return
        self.task_id = task_id or "task"
        os.makedirs(self.trace_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{stamp}_{_safe_name(self.task_id)}.jsonl"
        self.file_path = os.path.abspath(os.path.join(self.trace_dir, filename))
        self.event(
            "task_start",
            task_id=self.task_id,
            task_preview=_preview(task, 2000),
            run_id=self.run_id,
        )

    def event(self, event_type: str, **fields: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._seq += 1
            record: Dict[str, Any] = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "seq": self._seq,
                "event": event_type,
                "run_id": self.run_id,
                "task_id": fields.pop("task_id", None) or self.task_id,
            }
            record.update(fields)
            target = self.file_path
            if not target:
                os.makedirs(self.trace_dir, exist_ok=True)
                target = os.path.abspath(
                    os.path.join(self.trace_dir, f"{self.run_id}_session.jsonl")
                )
                self.file_path = target
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


__all__ = [
    "TraceLogger",
    "extract_tool_calls",
    "response_usage",
]
