"""Small local response cache for repeated MAS agent calls.

The cache is intentionally transparent to the MAS orchestration layer: it only
memoizes exact agent prompt responses and is enabled explicitly through
environment variables or benchmark flags.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional


TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def stable_digest(value: Any, digest_size: int = 16) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        raw = repr(value)
    return blake2b(raw.encode("utf-8", errors="ignore"), digest_size=digest_size).hexdigest()


class ResponseCache:
    """Append-only JSONL cache keyed by agent name, prompt, and config fingerprint."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        path: str = ".mas_response_cache.jsonl",
        namespace: str = "default",
        refresh: bool = False,
        max_entries: int = 50000,
    ):
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.namespace = namespace or "default"
        self.refresh = bool(refresh)
        self.max_entries = max(100, int(max_entries or 50000))
        self._loaded = False
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.session_hits = 0
        self.session_misses = 0
        self.session_writes = 0

    @classmethod
    def from_env(cls) -> "ResponseCache":
        enabled = str(os.getenv("MAS_RESPONSE_CACHE", "")).strip().lower() in TRUE_VALUES
        refresh = str(os.getenv("MAS_RESPONSE_CACHE_REFRESH", "")).strip().lower() in TRUE_VALUES
        path = os.getenv("MAS_RESPONSE_CACHE_PATH", ".mas_response_cache.jsonl")
        namespace = os.getenv("MAS_RESPONSE_CACHE_NAMESPACE", "default")
        try:
            max_entries = int(os.getenv("MAS_RESPONSE_CACHE_MAX_ENTRIES", "50000") or 50000)
        except ValueError:
            max_entries = 50000
        return cls(
            enabled=enabled,
            path=path,
            namespace=namespace,
            refresh=refresh,
            max_entries=max_entries,
        )

    def make_key(self, *, agent_name: str, prompt: str, fingerprint: Dict[str, Any]) -> str:
        return stable_digest(
            {
                "version": 1,
                "namespace": self.namespace,
                "agent_name": agent_name,
                "prompt": prompt,
                "fingerprint": fingerprint,
            }
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        self._ensure_loaded()
        if self.refresh:
            self.session_misses += 1
            return None
        entry = self._index.get(key)
        if entry and isinstance(entry.get("text"), str):
            self.session_hits += 1
            return dict(entry)
        self.session_misses += 1
        return None

    def put(
        self,
        key: str,
        *,
        agent_name: str,
        prompt: str,
        fingerprint: Dict[str, Any],
        text: str,
        tool_names: Optional[List[str]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        text = str(text or "")
        if not text or text.startswith("[Timeout]") or text.startswith("[Error:"):
            return False
        self._ensure_loaded()
        entry = {
            "key": key,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "namespace": self.namespace,
            "agent_name": agent_name,
            "prompt_hash": stable_digest(prompt, digest_size=12),
            "fingerprint_hash": stable_digest(fingerprint, digest_size=12),
            "text": text,
            "tool_names": list(tool_names or []),
        }
        with self._lock:
            self._index[key] = entry
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.session_writes += 1
                return True
            except OSError:
                return False

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "namespace": self.namespace,
            "refresh": self.refresh,
            "hits": self.session_hits,
            "misses": self.session_misses,
            "writes": self.session_writes,
            "entries_loaded": len(self._index) if self._loaded else 0,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self.path.exists():
                return
            try:
                lines = deque(maxlen=self.max_entries)
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        lines.append(line)
            except OSError:
                return
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(entry.get("key") or "")
                if key and entry.get("namespace") == self.namespace:
                    self._index[key] = entry
