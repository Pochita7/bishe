from __future__ import annotations

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional


_token_usage_tracker: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "project_token_usage_tracker",
    default=None,
)
_token_usage_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "project_token_usage_source",
    default="external",
)


def new_tracker() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "calls_without_usage": 0,
        "sources": {},
    }


def push_tracker(existing: Optional[Dict[str, Any]] = None):
    tracker = existing or new_tracker()
    token = _token_usage_tracker.set(tracker)
    return token, tracker


def reset_tracker(token) -> None:
    _token_usage_tracker.reset(token)


def get_tracker() -> Optional[Dict[str, Any]]:
    return _token_usage_tracker.get(None)


@contextmanager
def token_usage_scope(source: str) -> Iterator[None]:
    token = _token_usage_source.set(source or "external")
    try:
        yield
    finally:
        _token_usage_source.reset(token)


def snapshot_tracker(tracker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = tracker if tracker is not None else get_tracker()
    if not base:
        return new_tracker()
    return deepcopy(base)


def record_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    source: Optional[str] = None,
    model: str = "",
    operation: str = "",
) -> bool:
    tracker = get_tracker()
    if tracker is None:
        return False

    src = source or _token_usage_source.get() or "external"
    entry = tracker["sources"].setdefault(
        src,
        {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "models": {},
            "operations": {},
        },
    )

    tracker["input_tokens"] += int(input_tokens or 0)
    tracker["output_tokens"] += int(output_tokens or 0)
    tracker["total_tokens"] += int(total_tokens or 0)
    tracker["calls"] += 1

    entry["calls"] += 1
    entry["input_tokens"] += int(input_tokens or 0)
    entry["output_tokens"] += int(output_tokens or 0)
    entry["total_tokens"] += int(total_tokens or 0)

    if model:
        entry["models"][model] = entry["models"].get(model, 0) + 1
    if operation:
        entry["operations"][operation] = entry["operations"].get(operation, 0) + 1

    return True


def record_missing_usage(
    *,
    source: Optional[str] = None,
    model: str = "",
    operation: str = "",
) -> bool:
    tracker = get_tracker()
    if tracker is None:
        return False

    src = source or _token_usage_source.get() or "external"
    entry = tracker["sources"].setdefault(
        src,
        {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "models": {},
            "operations": {},
        },
    )
    tracker["calls_without_usage"] += 1
    if model:
        entry["models"][model] = entry["models"].get(model, 0) + 1
    if operation:
        entry["operations"][operation] = entry["operations"].get(operation, 0) + 1
    return True


def _usage_get(usage: Any, key: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def extract_usage(response: Any) -> Optional[Dict[str, int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None

    inp = (
        _usage_get(usage, "prompt_tokens")
        or _usage_get(usage, "input_tokens")
        or _usage_get(usage, "input_token_count")
        or 0
    )
    out = (
        _usage_get(usage, "completion_tokens")
        or _usage_get(usage, "output_tokens")
        or _usage_get(usage, "output_token_count")
        or 0
    )
    total = (
        _usage_get(usage, "total_tokens")
        or _usage_get(usage, "total_token_count")
        or 0
    )

    inp = int(inp or 0)
    out = int(out or 0)
    total = int(total or 0)
    if total == 0 and (inp or out):
        total = inp + out
    if total == 0 and inp == 0 and out == 0:
        return None

    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
    }


def record_openai_response_usage(
    response: Any,
    *,
    source: Optional[str] = None,
    model: str = "",
    operation: str = "",
) -> bool:
    usage = extract_usage(response)
    if usage is None:
        return record_missing_usage(source=source, model=model, operation=operation)
    return record_usage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        source=source,
        model=model,
        operation=operation,
    )
