"""Rerun selected TAMAS samples from old Guardian+Sentinel results.

The script is intentionally small and resumable. It reads an existing TAMAS
result JSONL, selects rows by a simple failure predicate, then reruns those
task ids with the current MAS implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from run_full_benchmark import make_sentinel_kwargs, run_tamas_single
from tamas_adapter.loader import extract_task_info, load_all_tamas


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_rows(rows: Iterable[Dict[str, Any]], mode: str, selector: str) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("mode") != mode:
            continue
        predicted = str(row.get("predicted_answer") or "").lower()
        is_suspended = "execution suspended by sentinel" in predicted
        task_failed = not bool(row.get("task_completed"))
        attack_failed = not bool(row.get("attack_resisted"))
        tool_count = int(row.get("total_tool_calls") or 0)
        if not tool_count and isinstance(row.get("tools_called"), list):
            tool_count = len(row.get("tools_called") or [])
        if selector == "suspended" and is_suspended:
            selected.append(row)
        elif selector == "actual-incomplete" and (task_failed or is_suspended or tool_count == 0):
            selected.append(row)
        elif selector == "failed-completion" and task_failed:
            selected.append(row)
        elif selector == "failed-attack" and attack_failed:
            selected.append(row)
        elif selector == "failed-any" and (task_failed or attack_failed or is_suspended):
            selected.append(row)
        elif selector == "all":
            selected.append(row)
    return selected


def _write_summary(path: Path, rows: List[Dict[str, Any]], total: int) -> None:
    mode_counts = Counter(row.get("mode", "") for row in rows)
    suspended = sum(
        1
        for row in rows
        if "execution suspended by sentinel" in str(row.get("predicted_answer") or "").lower()
    )
    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_selected": total,
        "completed_reruns": len(rows),
        "remaining": max(0, total - len(rows)),
        "by_mode": dict(mode_counts),
        "task_completed": sum(1 for row in rows if row.get("task_completed")),
        "attack_resisted": sum(1 for row in rows if row.get("attack_resisted")),
        "still_suspended": suspended,
        "malicious_tool_calls": sum(len(row.get("malicious_tools_called") or []) for row in rows),
        "aria": dict(Counter(row.get("aria_score", "") for row in rows)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun selected old TAMAS samples.")
    parser.add_argument(
        "--source",
        default="tamas_4mode_full_run/tamas_guardian_sentinel.jsonl",
        help="Old TAMAS result JSONL to mine for suspended samples.",
    )
    parser.add_argument("--mode", choices=["clean", "attack"], default="clean")
    parser.add_argument(
        "--select",
        choices=["suspended", "actual-incomplete", "failed-completion", "failed-attack", "failed-any", "all"],
        default="suspended",
        help="Which rows to rerun from the source JSONL.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--bootstrap-events", type=int, default=0)
    parser.add_argument("--sentinel-llm-model", default=os.environ.get("SENTINEL_LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--sentinel-llm-min-score", type=float, default=35.0)
    parser.add_argument("--sentinel-llm-max-calls-per-scope", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"diagnostic_runs/sentinel_false_positive_rerun_{args.mode}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    selection_path = out_dir / "selection.json"

    old_rows = _load_jsonl(Path(args.source))
    selected = _select_rows(old_rows, args.mode, args.select)
    if args.limit > 0:
        selected = selected[: args.limit]

    task_by_id = {extract_task_info(item)["id"]: extract_task_info(item) for item in load_all_tamas()}
    completed = {f"{row.get('id')}:{row.get('mode')}" for row in _load_jsonl(result_path)}
    selection_path.write_text(
        json.dumps(
            {
                "source": args.source,
                "mode": args.mode,
                "selector": args.select,
                "total_selected": len(selected),
                "ids": [row.get("id") for row in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary(summary_path, _load_jsonl(result_path), len(selected))

    sentinel_kwargs = make_sentinel_kwargs(
        str(out_dir / "guardian_sentinel"),
        bootstrap_events=args.bootstrap_events,
        enable_llm_review=True,
        llm_review_model=args.sentinel_llm_model,
        llm_review_min_score=args.sentinel_llm_min_score,
        llm_review_max_calls_per_scope=args.sentinel_llm_max_calls_per_scope,
    )

    for index, old_row in enumerate(selected, start=1):
        task_id = str(old_row.get("id") or "")
        combo = f"{task_id}:{args.mode}"
        if combo in completed:
            continue
        task_info = task_by_id.get(task_id)
        if task_info is None:
            continue
        if not args.quiet:
            print(f"[{index}/{len(selected)}] rerun {combo}", flush=True)
        row = await run_tamas_single(
            task_info,
            mode=args.mode,
            use_guardian=True,
            use_sentinel=True,
            sentinel_kwargs=sentinel_kwargs,
            verbose=not args.quiet,
        )
        row["old_predicted_answer"] = str(old_row.get("predicted_answer") or "")[:500]
        row["old_task_completed"] = bool(old_row.get("task_completed"))
        row["old_attack_resisted"] = bool(old_row.get("attack_resisted"))
        row["rerun_source"] = args.source
        _append_jsonl(result_path, row)
        completed.add(combo)
        _write_summary(summary_path, _load_jsonl(result_path), len(selected))

    rows = _load_jsonl(result_path)
    _write_summary(summary_path, rows, len(selected))
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
