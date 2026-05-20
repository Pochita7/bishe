"""Summarize GAIA Level1 four-mode benchmark progress."""
from __future__ import annotations

import json
from pathlib import Path


FILES = [
    ("无防御", Path("benchmark_gaia_baseline.jsonl")),
    ("单Guardian", Path("benchmark_gaia_defended.jsonl")),
    ("单增强Sentinel", Path("benchmark_gaia_sentinel_only.jsonl")),
    ("Guardian+增强Sentinel", Path("benchmark_gaia_sentinel.jsonl")),
]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main() -> None:
    for label, path in FILES:
        rows = load_rows(path)
        total = len(rows)
        correct = sum(1 for r in rows if r.get("is_correct"))
        timeouts = sum(1 for r in rows if r.get("predicted_answer") == "TIMEOUT")
        errors = sum(1 for r in rows if str(r.get("predicted_answer", "")).startswith("ERROR"))
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in rows)
        external_tokens = sum(int(r.get("external_total_tokens") or 0) for r in rows)
        elapsed = sum(float(r.get("elapsed") or 0) for r in rows)
        avg_tokens = total_tokens / total if total else 0
        avg_elapsed = elapsed / total if total else 0
        last = rows[-1].get("task_id", "") if rows else ""
        print(
            f"{label}: {total}/53, correct={correct}, timeouts={timeouts}, "
            f"errors={errors}, total_tokens={total_tokens}, "
            f"external_tokens={external_tokens}, avg_tokens={avg_tokens:.1f}, "
            f"avg_elapsed={avg_elapsed:.1f}s, last={last}"
        )


if __name__ == "__main__":
    main()
