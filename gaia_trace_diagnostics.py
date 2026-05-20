"""Run representative GAIA Level 1 cases with detailed MASTeam traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from gaia_solver.solver import build_task_prompt, compare_answers, load_gaia_tasks
from mas.factory import create_gaia_team


DEFAULT_CASES: Dict[int, str] = {
    1: "WebSearcher + CodeExecutor: web facts, numeric computation, scaled-unit final answer",
    2: "WebSearcher: Wikipedia/factual retrieval and counting",
    8: "FileReader: DOCX reading plus logical reasoning",
    10: "FileReader: XLSX color/map extraction",
    17: "MediaAnalyst: image/chess visual reasoning",
    31: "MediaAnalyst: audio transcription plus exact ingredient list",
    35: "FileReader + CodeExecutor: Python attachment execution",
}


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_preview(text: Any, limit: int = 300) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[:limit] + "...(truncated)"


def _parse_cases(value: str) -> List[int]:
    result: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    return result


def analyze_trace(trace_file: str) -> Dict[str, Any]:
    path = Path(trace_file)
    if not trace_file or not path.exists():
        return {"trace_file": trace_file, "events": 0, "issues": ["trace file missing"]}

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    event_counts = Counter(e.get("event") for e in events)
    agent_elapsed: Dict[str, float] = defaultdict(float)
    agent_calls: Dict[str, int] = defaultdict(int)
    tool_calls: Counter[str] = Counter()
    issues: List[str] = []
    longest_agents: List[Dict[str, Any]] = []

    for event in events:
        etype = event.get("event")
        if etype == "agent_end":
            agent = str(event.get("agent") or "unknown")
            elapsed = float(event.get("elapsed") or 0)
            agent_elapsed[agent] += elapsed
            agent_calls[agent] += 1
            longest_agents.append({"agent": agent, "elapsed": elapsed})
            for call in event.get("tool_calls") or []:
                tool_calls[str(call.get("name") or "unknown")] += 1
        elif etype in {"agent_timeout", "agent_cancelled", "agent_error"}:
            agent = str(event.get("agent") or "unknown")
            issues.append(f"{etype}: {agent} ({event.get('error') or event.get('timeout') or ''})")
        elif etype == "agent_tool_limit_warning":
            issues.append(
                f"tool limit warning: {event.get('agent')} used "
                f"{event.get('tool_count')} tools"
            )
        elif etype == "runtime_monitor" and event.get("decision") == "block":
            issues.append(
                f"runtime blocked: {event.get('source_agent')} -> "
                f"{event.get('target')} ({event.get('reason')})"
            )

    longest_agents.sort(key=lambda item: item["elapsed"], reverse=True)
    return {
        "trace_file": str(path),
        "events": len(events),
        "event_counts": dict(event_counts),
        "agent_calls": dict(agent_calls),
        "agent_elapsed": {k: round(v, 3) for k, v in agent_elapsed.items()},
        "tool_calls": dict(tool_calls),
        "longest_agents": longest_agents[:5],
        "issues": issues,
    }


def write_markdown_report(path: Path, results: List[Dict[str, Any]]) -> None:
    lines = [
        "# GAIA MAS Trace Diagnostics",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Case | Capability | Pred | Expected | Correct | Time | Tokens | Tools | Trace |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        metrics = r.get("metrics") or {}
        trace_file = r.get("trace_file") or ""
        lines.append(
            "| {case} | {capability} | {pred} | {expected} | {correct} | "
            "{elapsed} | {tokens} | {tools} | {trace} |".format(
                case=r.get("case_index"),
                capability=str(r.get("capability", "")).replace("|", "/"),
                pred=_safe_preview(r.get("predicted_answer"), 80).replace("|", "/"),
                expected=_safe_preview(r.get("ground_truth"), 80).replace("|", "/"),
                correct=r.get("is_correct"),
                elapsed=round(float(r.get("elapsed") or 0), 1),
                tokens=metrics.get("total_tokens", 0),
                tools=", ".join(f"{k}:{v}" for k, v in (metrics.get("tool_calls") or {}).items()),
                trace=trace_file,
            )
        )

    lines.extend(["", "## Findings", ""])
    for r in results:
        trace_summary = r.get("trace_summary") or {}
        issues = trace_summary.get("issues") or []
        longest = trace_summary.get("longest_agents") or []
        lines.append(f"### Case {r.get('case_index')} - {r.get('capability')}")
        lines.append(f"- Correct: {r.get('is_correct')}; pred={r.get('predicted_answer')!r}; expected={r.get('ground_truth')!r}")
        lines.append(f"- Agents: {trace_summary.get('agent_calls', {})}")
        lines.append(f"- Tools: {trace_summary.get('tool_calls', {})}")
        lines.append(f"- Longest agent calls: {longest}")
        lines.append(f"- Issues: {issues if issues else 'none recorded'}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def run_case(
    task: Dict[str, Any],
    *,
    case_index: int,
    capability: str,
    trace_dir: str,
    run_id: str,
    task_timeout: int,
    worker_timeout: int,
) -> Dict[str, Any]:
    prompt = build_task_prompt(task)
    task_id = str(task.get("task_id") or f"case-{case_index}")
    expected = str(task.get("Final answer") or "")
    start = time.time()
    team = create_gaia_team(
        enable_sentinel=False,
        max_rounds=3,
        verbose=False,
        worker_timeout=worker_timeout,
        max_tool_calls_per_worker=10,
        trace_enabled=True,
        trace_dir=trace_dir,
        trace_run_id=run_id,
    )

    predicted = "NO_ANSWER"
    metrics_dict: Dict[str, Any] = {}
    error = ""

    try:
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=task_id, expected_answer=expected),
            timeout=task_timeout,
        )
        predicted = str(result.get("answer") or "NO_ANSWER")
        metrics = result.get("metrics")
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else {}
    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        error = f"task timeout after {task_timeout}s"
        metrics = team.get_partial_metrics(answer=predicted, expected_answer=expected)
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else {}
    except Exception as exc:
        predicted = f"ERROR: {str(exc)[:200]}"
        error = str(exc)
        metrics = team.get_partial_metrics(answer=predicted, expected_answer=expected)
        metrics_dict = metrics.to_dict() if hasattr(metrics, "to_dict") else {}

    elapsed = time.time() - start
    trace_file = getattr(team.trace, "file_path", "")
    is_correct = compare_answers(predicted, expected)
    trace_summary = analyze_trace(trace_file)
    if error:
        trace_summary.setdefault("issues", []).append(error)

    return {
        "case_index": case_index,
        "task_id": task_id,
        "capability": capability,
        "question": task.get("Question", ""),
        "file_name": task.get("file_name", ""),
        "predicted_answer": predicted,
        "ground_truth": expected,
        "is_correct": is_correct,
        "elapsed": round(elapsed, 3),
        "metrics": metrics_dict,
        "trace_file": trace_file,
        "trace_summary": trace_summary,
    }


async def main_async(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"gaia_diag_{_now_id()}"
    trace_dir = args.trace_dir or os.path.join("mas_traces", run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MAS_TRACE_ENABLED"] = "1"
    os.environ["MAS_TRACE_DIR"] = trace_dir
    os.environ["MAS_TRACE_RUN_ID"] = run_id

    tasks = load_gaia_tasks()
    case_indices = _parse_cases(args.cases) if args.cases else list(DEFAULT_CASES)
    results: List[Dict[str, Any]] = []

    print(f"[TraceDiag] run_id={run_id}")
    print(f"[TraceDiag] trace_dir={trace_dir}")
    print(f"[TraceDiag] cases={case_indices}")

    for case_index in case_indices:
        task = tasks[case_index - 1]
        capability = DEFAULT_CASES.get(case_index, "custom GAIA case")
        print(f"\n[TraceDiag] Case {case_index}: {capability}")
        print(f"  Q: {_safe_preview(task.get('Question'), 140)}")
        result = await run_case(
            task,
            case_index=case_index,
            capability=capability,
            trace_dir=trace_dir,
            run_id=run_id,
            task_timeout=args.task_timeout,
            worker_timeout=args.worker_timeout,
        )
        results.append(result)
        print(
            "  -> correct={correct} pred={pred!r} expected={expected!r} "
            "elapsed={elapsed:.1f}s trace={trace}".format(
                correct=result["is_correct"],
                pred=result["predicted_answer"],
                expected=result["ground_truth"],
                elapsed=result["elapsed"],
                trace=result["trace_file"],
            )
        )
        issues = (result.get("trace_summary") or {}).get("issues") or []
        if issues:
            print(f"  issues: {issues[:3]}")

    json_path = output_dir / f"{run_id}_results.json"
    md_path = output_dir / f"{run_id}_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(md_path, results)
    print(f"\n[TraceDiag] wrote {json_path}")
    print(f"[TraceDiag] wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="1-based GAIA case indices, comma-separated")
    parser.add_argument("--task-timeout", type=int, default=240)
    parser.add_argument("--worker-timeout", type=int, default=90)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--output-dir", default="diagnostic_runs")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
