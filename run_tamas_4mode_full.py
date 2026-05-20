"""Run the full TAMAS benchmark in four defense modes with resume support.

The runner writes one JSONL file per defense mode. Each line is flushed after a
single inference, so the process can be stopped and resumed later by reusing the
same output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_benchmark import make_sentinel_kwargs, run_tamas_single
from tamas_adapter.loader import load_all_tamas, ATTACK_TYPES, SCENARIOS, extract_task_info


DEFENSES: list[dict[str, Any]] = [
    {
        "key": "baseline",
        "label": "No defense",
        "use_guardian": False,
        "use_sentinel": False,
        "output": "tamas_baseline.jsonl",
    },
    {
        "key": "guardian",
        "label": "Guardian only",
        "use_guardian": True,
        "use_sentinel": False,
        "output": "tamas_guardian.jsonl",
    },
    {
        "key": "sentinel",
        "label": "Enhanced Sentinel only",
        "use_guardian": False,
        "use_sentinel": True,
        "output": "tamas_sentinel.jsonl",
    },
    {
        "key": "guardian_sentinel",
        "label": "Guardian + enhanced Sentinel",
        "use_guardian": True,
        "use_sentinel": True,
        "output": "tamas_guardian_sentinel.jsonl",
    },
]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(row.get("id") or "")
            mode = str(row.get("mode") or "")
            if task_id and mode:
                completed.add(f"{task_id}:{mode}")
    return completed


def write_manifest(output_dir: Path, total_tasks: int, args: argparse.Namespace) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total_tasks,
        "total_inferences": total_tasks * 2 * len(DEFENSES),
        "phase_order": ["clean", "attack"],
        "defenses": DEFENSES,
        "attack_types": ATTACK_TYPES,
        "scenarios": SCENARIOS,
        "args": vars(args),
    }
    path = output_dir / "manifest.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def sentinel_kwargs_for(output_dir: Path, defense_key: str, args: argparse.Namespace) -> dict[str, Any]:
    prefix = str(output_dir / f"{defense_key}_sentinel")
    return make_sentinel_kwargs(
        prefix,
        args.sentinel_bootstrap_events,
        args.sentinel_window_seconds,
        not args.sentinel_global_realtime,
        args.sentinel_long_event_window,
        args.sentinel_long_min_count,
        args.sentinel_llm_review,
        args.sentinel_llm_model,
        args.sentinel_llm_min_score,
        args.sentinel_llm_max_calls_per_scope,
    )


async def run_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_tamas()
    write_manifest(output_dir, len(data), args)

    print("=" * 78, flush=True)
    print("TAMAS full four-mode benchmark", flush=True)
    print(f"Output dir: {output_dir.resolve()}", flush=True)
    print(f"Tasks: {len(data)} | inferences: {len(data) * 2 * len(DEFENSES)}", flush=True)
    print("Order: all clean first, then all attack", flush=True)
    print("=" * 78, flush=True)

    for mode in ["clean", "attack"]:
        print(f"\n## Phase: {mode}", flush=True)
        for defense in DEFENSES:
            out_path = output_dir / defense["output"]
            completed = load_completed(out_path)
            defense_completed = sum(1 for item in data if f"{extract_task_info(item)['id']}:{mode}" in completed)
            remaining = len(data) - defense_completed
            print(
                f"\n[{defense['label']}] {mode}: "
                f"{defense_completed}/{len(data)} done, {remaining} remaining -> {out_path}",
                flush=True,
            )
            if remaining <= 0:
                continue

            kwargs = None
            if defense["use_sentinel"]:
                kwargs = sentinel_kwargs_for(output_dir, defense["key"], args)

            for index, item in enumerate(data, start=1):
                task_info = extract_task_info(item)
                combo = f"{task_info['id']}:{mode}"
                if combo in completed:
                    continue

                print(
                    f"[{mode}][{defense['key']}][{index}/{len(data)}] "
                    f"{task_info['attack_type']}/{task_info['scenario']}/{task_info['index']}",
                    flush=True,
                )
                row = await run_tamas_single(
                    task_info,
                    mode=mode,
                    use_guardian=bool(defense["use_guardian"]),
                    use_sentinel=bool(defense["use_sentinel"]),
                    sentinel_kwargs=kwargs,
                    verbose=not args.quiet,
                )
                row["run_output_dir"] = str(output_dir)
                row["defense_key"] = defense["key"]
                append_jsonl(out_path, row)
                completed.add(combo)

    print("\nTAMAS four-mode benchmark finished.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full TAMAS benchmark in four defense modes.")
    parser.add_argument("--output-dir", default="tamas_4mode_full_run")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sentinel-bootstrap-events", type=int, default=24)
    parser.add_argument("--sentinel-window-seconds", type=int, default=120)
    parser.add_argument("--sentinel-long-event-window", type=int, default=200)
    parser.add_argument("--sentinel-long-min-count", type=int, default=6)
    parser.add_argument("--sentinel-global-realtime", action="store_true")
    parser.add_argument("--sentinel-llm-review", action="store_true")
    parser.add_argument("--sentinel-llm-model", default=os.environ.get("SENTINEL_LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--sentinel-llm-min-score", type=float, default=35.0)
    parser.add_argument("--sentinel-llm-max-calls-per-scope", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
