"""Run TAMAS external comparison baselines with resume support.

This runner is intentionally separate from the main four-mode TAMAS run. It
adds two literature-inspired baselines:

- prompt_aug: Prompt Augmentation / Spotlighting style instruction hierarchy.
- melon_lite: Masked re-execution plus tool-call comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mas.compat import ensure_utf8_stdio

ensure_utf8_stdio()

from run_full_benchmark import (
    run_tamas_melon_lite_single,
    run_tamas_prompt_aug_single,
)
from tamas_adapter.loader import ATTACK_TYPES, SCENARIOS, extract_task_info, load_all_tamas


COMPARISON_DEFENSES: list[dict[str, Any]] = [
    {
        "key": "prompt_aug",
        "label": "Prompt Augmentation / Spotlighting",
        "output": "tamas_prompt_aug.jsonl",
    },
    {
        "key": "melon_lite",
        "label": "MELON-lite masked re-execution",
        "output": "tamas_melon_lite.jsonl",
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


def select_data(args: argparse.Namespace) -> list[Any]:
    data = load_all_tamas()
    if not args.max_tasks_per_group:
        return data

    selected = []
    counts: Counter[tuple[str, str]] = Counter()
    for item in data:
        info = extract_task_info(item)
        key = (info["attack_type"], info["scenario"])
        if counts[key] >= args.max_tasks_per_group:
            continue
        selected.append(item)
        counts[key] += 1
    return selected


def write_manifest(output_dir: Path, total_tasks: int, args: argparse.Namespace) -> None:
    phase_count = 2 if args.phase == "all" else 1
    model_runs_per_task = sum(2 if key == "melon_lite" else 1 for key in args.defenses)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": total_tasks,
        "total_result_rows": total_tasks * phase_count * len(args.defenses),
        "total_model_runs": total_tasks * phase_count * model_runs_per_task,
        "total_inferences": total_tasks * phase_count * model_runs_per_task,
        "phase_order": args.phase if args.phase != "all" else ["clean", "attack"],
        "defenses": [
            item for item in COMPARISON_DEFENSES if item["key"] in set(args.defenses)
        ],
        "attack_types": ATTACK_TYPES,
        "scenarios": SCENARIOS,
        "args": vars(args),
    }
    path = output_dir / "manifest.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = select_data(args)
    selected_defenses = [d for d in COMPARISON_DEFENSES if d["key"] in set(args.defenses)]
    phases = ["clean", "attack"] if args.phase == "all" else [args.phase]
    write_manifest(output_dir, len(data), args)

    print("=" * 78, flush=True)
    print("TAMAS external comparison baselines", flush=True)
    print(f"Output dir: {output_dir.resolve()}", flush=True)
    print(f"Tasks: {len(data)} | phases: {phases} | defenses: {[d['key'] for d in selected_defenses]}", flush=True)
    print("=" * 78, flush=True)

    for mode in phases:
        print(f"\n## Phase: {mode}", flush=True)
        for defense in selected_defenses:
            out_path = output_dir / defense["output"]
            completed = load_completed(out_path)
            defense_completed = sum(
                1 for item in data if f"{extract_task_info(item)['id']}:{mode}" in completed
            )
            remaining = len(data) - defense_completed
            print(
                f"\n[{defense['label']}] {mode}: "
                f"{defense_completed}/{len(data)} done, {remaining} remaining -> {out_path}",
                flush=True,
            )
            if remaining <= 0:
                continue

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
                if defense["key"] == "prompt_aug":
                    row = await run_tamas_prompt_aug_single(
                        task_info,
                        mode=mode,
                        verbose=not args.quiet,
                    )
                elif defense["key"] == "melon_lite":
                    row = await run_tamas_melon_lite_single(
                        task_info,
                        mode=mode,
                        verbose=not args.quiet,
                        jaccard_threshold=args.melon_jaccard_threshold,
                        min_shadow_tools=args.melon_min_shadow_tools,
                    )
                else:
                    raise ValueError(f"Unknown defense: {defense['key']}")

                row["run_output_dir"] = str(output_dir)
                row["defense_key"] = defense["key"]
                append_jsonl(out_path, row)
                completed.add(combo)

    print("\nTAMAS comparison baselines finished.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAMAS external comparison baselines.")
    parser.add_argument("--output-dir", default="tamas_comparison_run")
    parser.add_argument("--phase", choices=["clean", "attack", "all"], default="all")
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=[item["key"] for item in COMPARISON_DEFENSES],
        default=[item["key"] for item in COMPARISON_DEFENSES],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--max-tasks-per-group",
        type=int,
        default=0,
        help="Optional pilot limit per attack_type/scenario group. Full run uses 0.",
    )
    parser.add_argument("--melon-jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--melon-min-shadow-tools", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
