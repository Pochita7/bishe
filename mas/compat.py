"""Small compatibility helpers shared by command-line entry points."""

from __future__ import annotations

import io
import sys


def ensure_utf8_stdio() -> None:
    """Use UTF-8 console streams on Windows terminals that default to GBK."""
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    if encoding in {"utf-8", "utf8"}:
        return
    if not hasattr(sys.stdout, "buffer") or not hasattr(sys.stderr, "buffer"):
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
