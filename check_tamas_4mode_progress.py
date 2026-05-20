"""Summarize progress and defense metrics for the TAMAS four-mode run."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tamas_adapter.loader import ATTACK_TYPES, load_all_tamas


FILES = [
    ("baseline", "No defense", "tamas_baseline.jsonl"),
    ("guardian", "Guardian only", "tamas_guardian.jsonl"),
    ("sentinel", "Enhanced Sentinel only", "tamas_sentinel.jsonl"),
    ("guardian_sentinel", "Guardian + enhanced Sentinel", "tamas_guardian_sentinel.jsonl"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def full_task_count() -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return len(load_all_tamas())


def pct(num: float, den: float) -> str:
    if not den:
        return "0.0%"
    return f"{num / den * 100:.1f}%"


def summarize(rows: list[dict[str, Any]], total_tasks: int) -> dict[str, Any]:
    clean = [r for r in rows if r.get("mode") == "clean"]
    attack = [r for r in rows if r.get("mode") == "attack"]
    malicious_calls = sum(len(r.get("malicious_tools_called") or []) for r in attack)
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in rows)
    external_tokens = sum(int(r.get("external_total_tokens") or 0) for r in rows)
    total_tool_calls = sum(int(r.get("total_tool_calls") or 0) for r in rows)
    elapsed = sum(float(r.get("elapsed_time") or 0) for r in rows)
    attack_resisted = sum(1 for r in attack if r.get("attack_resisted"))
    clean_completed = sum(1 for r in clean if r.get("task_completed"))
    aria = Counter(str(r.get("aria_score") or "UNKNOWN") for r in attack)
    return {
        "rows": len(rows),
        "clean_done": len(clean),
        "attack_done": len(attack),
        "clean_progress": pct(len(clean), total_tasks),
        "attack_progress": pct(len(attack), total_tasks),
        "clean_completed": clean_completed,
        "clean_completion_rate": pct(clean_completed, len(clean)),
        "attack_resisted": attack_resisted,
        "attack_defense_rate": pct(attack_resisted, len(attack)),
        "attack_success": len(attack) - attack_resisted,
        "malicious_tool_calls": malicious_calls,
        "aria": dict(aria),
        "total_tokens": total_tokens,
        "external_tokens": external_tokens,
        "total_tool_calls": total_tool_calls,
        "avg_tokens": total_tokens / len(rows) if rows else 0.0,
        "avg_tool_calls": total_tool_calls / len(rows) if rows else 0.0,
        "avg_elapsed": elapsed / len(rows) if rows else 0.0,
    }


def summarize_by_attack(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    attack_rows = [r for r in rows if r.get("mode") == "attack"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attack_rows:
        grouped[str(row.get("attack_type") or "UNKNOWN")].append(row)

    result: dict[str, dict[str, Any]] = {}
    for attack_type in ATTACK_TYPES:
        items = grouped.get(attack_type, [])
        resisted = sum(1 for r in items if r.get("attack_resisted"))
        malicious = sum(len(r.get("malicious_tools_called") or []) for r in items)
        aria = Counter(str(r.get("aria_score") or "UNKNOWN") for r in items)
        result[attack_type] = {
            "done": len(items),
            "resisted": resisted,
            "defense_rate": pct(resisted, len(items)),
            "attack_success": len(items) - resisted,
            "malicious_tool_calls": malicious,
            "aria": dict(aria),
        }
    return result


def print_summary(output_dir: Path, write_json: bool) -> None:
    total_tasks = full_task_count()
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Full TAMAS tasks: {total_tasks}; expected rows per defense: {total_tasks * 2}")
    print()

    all_summary: dict[str, Any] = {"total_tasks": total_tasks, "defenses": {}}
    for key, label, file_name in FILES:
        path = output_dir / file_name
        rows = load_rows(path)
        summary = summarize(rows, total_tasks)
        by_attack = summarize_by_attack(rows)
        all_summary["defenses"][key] = {
            "label": label,
            "file": str(path),
            "summary": summary,
            "by_attack": by_attack,
        }

        print(f"[{label}] {file_name}")
        print(
            f"  rows={summary['rows']}/{total_tasks * 2} | "
            f"clean={summary['clean_done']}/{total_tasks} ({summary['clean_progress']}) | "
            f"attack={summary['attack_done']}/{total_tasks} ({summary['attack_progress']})"
        )
        print(
            f"  clean_completion={summary['clean_completed']}/{summary['clean_done']} "
            f"({summary['clean_completion_rate']}) | "
            f"attack_defense={summary['attack_resisted']}/{summary['attack_done']} "
            f"({summary['attack_defense_rate']}) | attack_success={summary['attack_success']}"
        )
        print(
            f"  malicious_tool_calls={summary['malicious_tool_calls']} | "
            f"tool_calls={summary['total_tool_calls']} | tokens={summary['total_tokens']} | "
            f"external_tokens={summary['external_tokens']} | "
            f"avg_time={summary['avg_elapsed']:.1f}s | avg_tokens={summary['avg_tokens']:.1f}"
        )
        print(f"  ARIA attack distribution: {summary['aria']}")
        print("  Defense rate by attack type:")
        for attack_type in ATTACK_TYPES:
            item = by_attack[attack_type]
            print(
                f"    {attack_type}: {item['resisted']}/{item['done']} "
                f"({item['defense_rate']}), attack_success={item['attack_success']}, "
                f"mal_tools={item['malicious_tool_calls']}, aria={item['aria']}"
            )
        print()

    if write_json:
        summary_path = output_dir / "progress_summary.json"
        summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TAMAS four-mode benchmark progress.")
    parser.add_argument("--output-dir", default="tamas_4mode_full_run")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args()
    print_summary(Path(args.output_dir), args.write_json)


if __name__ == "__main__":
    main()
