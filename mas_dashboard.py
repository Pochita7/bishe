"""Web dashboard for running MAS defense experiments.

Run:
    python mas_dashboard.py

Then open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from gaia_solver.config import BASE_URL, MODEL_NAME, get_text_client
from gaia_solver.solver import build_task_prompt, compare_answers, load_gaia_tasks
from gaia_solver import tools as gaia_tools
from mas.factory import create_gaia_team, create_tamas_team
from mas.metrics import TaskMetrics
from mas.sentinel import SecurityEventBus
from tamas_adapter.evaluator import evaluate_output
from tamas_adapter.guardian import Guardian
from tamas_adapter.loader import ATTACK_TYPES, SCENARIOS, extract_task_info, load_all_tamas
from tamas_adapter.prompt_builder import build_attack_prompt, build_clean_prompt
from tamas_adapter.tools import (
    clear_tool_call_log,
    get_tamas_function_tools,
    get_tool_call_log,
)


APP_TITLE = "MAS Defense Console"
RUN_DIR = Path("dashboard_runs")
RUN_DIR.mkdir(exist_ok=True)

app = FastAPI(title=APP_TITLE)


class RunRequest(BaseModel):
    dataset: Literal["gaia", "tamas"] = "tamas"
    guardian: bool = False
    sentinel: bool = True
    limit: int = Field(default=3, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    tamas_mode: Literal["clean", "attack", "both"] = "attack"
    attack_type: str = "all"
    scenario: str = "all"
    sentinel_bootstrap_events: int = Field(default=0, ge=0, le=500)
    sentinel_window_seconds: int = Field(default=120, ge=10, le=3600)
    sentinel_long_event_window: int = Field(default=200, ge=0, le=5000)
    sentinel_long_min_count: int = Field(default=6, ge=1, le=100)
    max_rounds: int = Field(default=2, ge=1, le=5)
    timeout_seconds: int = Field(default=300, ge=30, le=1800)


@dataclass
class DashboardJob:
    id: str
    request: RunRequest
    loop: asyncio.AbstractEventLoop
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    task: Optional[asyncio.Task] = None
    cancel_requested: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        item = {
            "seq": len(self.events) + 1,
            "type": event_type,
            "time": time.strftime("%H:%M:%S"),
            "job_id": self.id,
            "payload": payload or {},
        }
        self.events.append(item)
        if len(self.events) > 1200:
            self.events = self.events[-1000:]

        def push() -> None:
            for queue in list(self.subscribers):
                queue.put_nowait(item)

        self.loop.call_soon_threadsafe(push)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "request": self.request.model_dump(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "event_count": len(self.events),
            "summary": self.summary,
        }


JOBS: Dict[str, DashboardJob] = {}


def defense_mode(req: RunRequest) -> str:
    if req.guardian and req.sentinel:
        return "Guardian+Sentinel"
    if req.guardian:
        return "Guardian-only"
    if req.sentinel:
        return "Sentinel-only"
    return "Baseline"


def metric_fields(metrics: Optional[TaskMetrics]) -> Dict[str, Any]:
    if metrics is None:
        return {}
    return {
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "external_total_tokens": metrics.external_total_tokens,
        "total_external_llm_calls": metrics.total_external_llm_calls,
        "total_tool_calls": metrics.total_tool_calls,
        "tool_calls": dict(metrics.tool_calls),
        "agent_calls": dict(metrics.agent_calls),
        "total_agent_calls": metrics.total_agent_calls,
        "rounds": metrics.rounds,
        "turns": metrics.turns,
        "planner_time": round(metrics.planner_time, 1),
        "worker_time": round(metrics.worker_time, 1),
    }


def compact(value: Any, limit: int = 900) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...(truncated)"
    if isinstance(value, dict):
        return {str(k): compact(v, limit=360) for k, v in list(value.items())[:18]}
    if isinstance(value, (list, tuple, set)):
        return [compact(v, limit=360) for v in list(value)[:18]]
    return repr(value)[:limit]


def emit_security_event(job: DashboardJob, event: Dict[str, Any]) -> None:
    params = event.get("params") or {}
    preview = (
        params.get("task_preview")
        or params.get("prompt_preview")
        or params.get("content_preview")
        or params.get("result_preview")
        or params.get("step_preview")
        or params.get("assignment_preview")
        or params.get("kwargs")
        or ""
    )
    job.emit(
        "security_event",
        {
            "timestamp": event.get("timestamp", ""),
            "source": event.get("source_agent", "unknown"),
            "target": event.get("target", "unknown"),
            "behavior": event.get("behavior_type", "unknown"),
            "gate": event.get("gate", ""),
            "decision": event.get("decision", "allow"),
            "reason": event.get("reason", ""),
            "matched_rules": event.get("matched_rules", []),
            "task_id": event.get("task_id", params.get("task_id", "")),
            "preview": compact(preview, 1000),
            "params": compact(params, 1000),
        },
    )


def build_sentinel_kwargs(job_id: str, req: RunRequest, prefix: str) -> Dict[str, Any]:
    base = RUN_DIR / f"{job_id}_{prefix}"
    return {
        "behavior_db_path": str(base.with_name(base.name + "_behavior_db.jsonl")),
        "assessment_log_path": str(base.with_name(base.name + "_assessments.jsonl")),
        "bootstrap_events": req.sentinel_bootstrap_events,
        "realtime_window_seconds": req.sentinel_window_seconds,
        "long_horizon_event_limit": req.sentinel_long_event_window,
        "long_horizon_min_count": req.sentinel_long_min_count,
        "scope_realtime_by_task": True,
    }


def attach_observer(
    job: DashboardJob,
    team: Any,
    *,
    guardian: Optional[Guardian],
    event_bus: Optional[SecurityEventBus],
) -> SecurityEventBus:
    bus = getattr(team, "security_event_bus", None) or event_bus or SecurityEventBus()
    team.security_event_bus = bus
    if guardian is not None:
        guardian.set_event_sink(bus.publish)
        setattr(team, "guardian", guardian)
    bus.subscribe(lambda event: emit_security_event(job, event))
    return bus


async def run_gaia_case(
    job: DashboardJob,
    task: Dict[str, Any],
    case_index: int,
    total_cases: int,
) -> Dict[str, Any]:
    req = job.request
    task_id = task.get("task_id", f"gaia-{case_index}")
    ground_truth = task.get("Final answer", "")
    prompt = build_task_prompt(task)
    job.emit(
        "case_started",
        {
            "case_index": case_index,
            "total_cases": total_cases,
            "dataset": "gaia",
            "task_id": task_id,
            "question": compact(task.get("Question", ""), 1000),
            "file_name": task.get("file_name", ""),
        },
    )

    client = get_text_client()
    guardian = Guardian(scenario="gaia", strict_mode=False) if req.guardian else None
    external_bus = SecurityEventBus() if not req.sentinel else None
    if req.guardian:
        team = create_tamas_team(
            scenario="gaia",
            domain_tools=None,
            guardian=guardian,
            enable_sentinel=req.sentinel,
            sentinel_kwargs=build_sentinel_kwargs(job.id, req, "gaia"),
            client=client,
            max_rounds=max(2, req.max_rounds),
            verbose=False,
            worker_timeout=req.timeout_seconds,
            max_tool_calls_per_worker=10,
        )
    else:
        team = create_gaia_team(
            client=client,
            enable_sentinel=req.sentinel,
            sentinel_kwargs=build_sentinel_kwargs(job.id, req, "gaia"),
            max_rounds=max(2, req.max_rounds),
            verbose=False,
            worker_timeout=req.timeout_seconds,
            max_tool_calls_per_worker=10,
        )
    attach_observer(job, team, guardian=guardian, event_bus=external_bus)

    started = time.time()
    result: Dict[str, Any]
    try:
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=task_id, expected_answer=ground_truth),
            timeout=req.timeout_seconds,
        )
        predicted = result.get("answer", "") or "NO_ANSWER"
        metrics = metric_fields(result.get("metrics"))
    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        metrics = metric_fields(team.get_partial_metrics(answer=predicted, expected_answer=ground_truth))
        result = {"messages": [], "sentinel_status": getattr(getattr(team, "sentinel", None), "status", None)}
    finally:
        try:
            gaia_tools._cleanup_browser()
        except Exception:
            pass

    elapsed = round(time.time() - started, 1)
    is_correct = compare_answers(predicted, ground_truth)
    case_result = {
        "task_id": task_id,
        "dataset": "gaia",
        "predicted_answer": compact(predicted, 1200),
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "elapsed": elapsed,
        "defense_mode": defense_mode(req),
        "sentinel_status": compact(result.get("sentinel_status"), 1200),
        **metrics,
    }
    job.emit("case_finished", case_result)
    return case_result


async def run_tamas_case(
    job: DashboardJob,
    task_info: Dict[str, Any],
    mode: Literal["clean", "attack"],
    case_index: int,
    total_cases: int,
) -> Dict[str, Any]:
    req = job.request
    scenario = task_info["scenario"]
    attack_type = task_info["attack_type"]
    task_id = f"{task_info['id']}:{mode}"
    prompt = build_clean_prompt(task_info) if mode == "clean" else build_attack_prompt(task_info)
    job.emit(
        "case_started",
        {
            "case_index": case_index,
            "total_cases": total_cases,
            "dataset": "tamas",
            "mode": mode,
            "task_id": task_id,
            "attack_type": attack_type,
            "scenario": scenario,
            "query": compact(task_info["clean_query"] if mode == "clean" else task_info["attack_query"], 1000),
        },
    )

    clear_tool_call_log()
    domain_tools = get_tamas_function_tools(
        scenario=scenario,
        include_malicious=True,
        include_benign=True,
    )
    guardian = None
    external_bus = SecurityEventBus() if not req.sentinel else None
    if req.guardian:
        guardian = Guardian(scenario=scenario, strict_mode=False)
        guardian.scan_tool_descriptions(
            [{"name": tool.name, "description": tool.description} for tool in domain_tools],
            verbose=False,
        )

    team = create_tamas_team(
        scenario=scenario,
        domain_tools=domain_tools,
        guardian=guardian,
        enable_sentinel=req.sentinel,
        sentinel_kwargs=build_sentinel_kwargs(job.id, req, "tamas"),
        client=get_text_client(),
        max_rounds=req.max_rounds,
        verbose=False,
        worker_timeout=req.timeout_seconds,
        max_tool_calls_per_worker=8,
    )
    attach_observer(job, team, guardian=guardian, event_bus=external_bus)

    started = time.time()
    try:
        result = await asyncio.wait_for(team.run(task=prompt, task_id=task_id), timeout=req.timeout_seconds)
        predicted = result.get("answer", "") or "NO_ANSWER"
        metrics = metric_fields(result.get("metrics"))
    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        metrics = metric_fields(team.get_partial_metrics(answer=predicted))
        result = {"messages": [], "sentinel_status": getattr(getattr(team, "sentinel", None), "status", None)}
    finally:
        try:
            gaia_tools._cleanup_browser()
        except Exception:
            pass

    messages = result.get("messages", [])
    full_output = "\n".join(
        f"[{message.get('source', '')}]: {message.get('content', '')}"
        for message in messages
        if message.get("content")
    )
    tool_log = get_tool_call_log()
    if tool_log:
        full_output += f"\n[ToolLog] TOOL_CALLS: {'; '.join(tool_log)}"
    if guardian is not None:
        full_output += f"\n[Guardian Audit Log] {guardian.get_log().summary()}"

    eval_result = evaluate_output(
        user_query=task_info["clean_query"],
        actual_output=full_output,
        attack_type=attack_type,
        mode=mode,
        injected_query=task_info["attack_query"] if mode == "attack" else "",
    )
    case_result = {
        "task_id": task_id,
        "dataset": "tamas",
        "mode": mode,
        "attack_type": attack_type,
        "scenario": scenario,
        "predicted_answer": compact(predicted, 1200),
        "task_completed": eval_result["task_completed"],
        "attack_resisted": eval_result["attack_resisted"],
        "aria_score": eval_result["aria_score"],
        "score": eval_result["score"],
        "reasoning": compact(eval_result["reasoning"], 1000),
        "malicious_tools_called": eval_result.get("malicious_tools_called", []),
        "elapsed": round(time.time() - started, 1),
        "defense_mode": defense_mode(req),
        "sentinel_status": compact(result.get("sentinel_status"), 1200),
        **metrics,
    }
    job.emit("case_finished", case_result)
    return case_result


async def run_dashboard_job(job: DashboardJob) -> None:
    req = job.request
    job.status = "running"
    job.emit(
        "job_started",
        {
            "defense_mode": defense_mode(req),
            "dataset": req.dataset,
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "request": req.model_dump(),
        },
    )

    results: List[Dict[str, Any]] = []
    try:
        if req.dataset == "gaia":
            tasks = load_gaia_tasks()
            selected = tasks[req.offset:req.offset + req.limit]
            for index, task in enumerate(selected, start=1):
                if job.cancel_requested:
                    break
                results.append(await run_gaia_case(job, task, index, len(selected)))
        else:
            attack_types = None if req.attack_type == "all" else [req.attack_type]
            scenarios = None if req.scenario == "all" else [req.scenario]
            raw_items = load_all_tamas(attack_types=attack_types, scenarios=scenarios)
            selected_items = raw_items[req.offset:req.offset + req.limit]
            modes: List[Literal["clean", "attack"]]
            modes = ["clean", "attack"] if req.tamas_mode == "both" else [req.tamas_mode]
            total_cases = len(selected_items) * len(modes)
            case_index = 0
            for item in selected_items:
                task_info = extract_task_info(item)
                for mode in modes:
                    if job.cancel_requested:
                        break
                    case_index += 1
                    results.append(await run_tamas_case(job, task_info, mode, case_index, total_cases))

        job.status = "cancelled" if job.cancel_requested else "completed"
        job.summary = summarize_results(results, req)
        job.finished_at = time.time()
        job.emit("job_finished", {"status": job.status, "summary": job.summary})
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.finished_at = time.time()
        job.emit("job_finished", {"status": "cancelled", "summary": job.summary})
    except Exception as exc:
        job.status = "failed"
        job.finished_at = time.time()
        job.emit("job_failed", {"error": repr(exc)})


def summarize_results(results: List[Dict[str, Any]], req: RunRequest) -> Dict[str, Any]:
    total = len(results)
    summary: Dict[str, Any] = {
        "total_cases": total,
        "defense_mode": defense_mode(req),
        "dataset": req.dataset,
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in results),
        "total_tool_calls": sum(int(r.get("total_tool_calls", 0) or 0) for r in results),
        "avg_elapsed": round(sum(float(r.get("elapsed", 0) or 0) for r in results) / total, 1) if total else 0.0,
    }
    if req.dataset == "gaia":
        correct = sum(1 for r in results if r.get("is_correct"))
        summary["correct"] = correct
        summary["accuracy"] = round(correct / total * 100, 1) if total else 0.0
    else:
        completed = sum(1 for r in results if r.get("task_completed"))
        resisted = sum(1 for r in results if r.get("attack_resisted"))
        summary["task_completed"] = completed
        summary["attack_resisted"] = resisted
        summary["completion_rate"] = round(completed / total * 100, 1) if total else 0.0
        summary["resistance_rate"] = round(resisted / total * 100, 1) if total else 0.0
    return summary


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return DASHBOARD_HTML


@app.get("/api/options")
async def options() -> Dict[str, Any]:
    gaia_count = len(load_gaia_tasks())
    return {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "gaia_count": gaia_count,
        "attack_types": ATTACK_TYPES,
        "scenarios": SCENARIOS,
        "defaults": RunRequest().model_dump(),
    }


@app.post("/api/runs")
async def start_run(req: RunRequest) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    loop = asyncio.get_running_loop()
    job = DashboardJob(id=job_id, request=req, loop=loop)
    JOBS[job_id] = job
    job.task = loop.create_task(run_dashboard_job(job))
    return job.snapshot()


@app.get("/api/runs")
async def list_runs() -> Dict[str, Any]:
    return {"runs": [job.snapshot() for job in sorted(JOBS.values(), key=lambda item: item.started_at, reverse=True)[:20]]}


@app.get("/api/runs/{job_id}")
async def get_run(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.snapshot()


@app.post("/api/runs/{job_id}/cancel")
async def cancel_run(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.cancel_requested = True
    if job.task is not None and not job.task.done():
        job.task.cancel()
    return job.snapshot()


@app.get("/api/runs/{job_id}/events")
async def stream_events(job_id: str, request: Request) -> EventSourceResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def generator():
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.add(queue)
        try:
            for item in job.events:
                yield {"data": json.dumps(item, ensure_ascii=False)}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"data": json.dumps(item, ensure_ascii=False)}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "ping", "time": time.strftime("%H:%M:%S")})}
        finally:
            job.subscribers.discard(queue)

    return EventSourceResponse(generator())


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MAS Defense Console</title>
  <style>
    :root {
      --paper: #f4eddf;
      --panel: #fffaf0;
      --ink: #1f2a2b;
      --muted: #6a7778;
      --line: #d8ccb8;
      --accent: #0e7c86;
      --accent-2: #d95d39;
      --safe: #188050;
      --warn: #b46b00;
      --danger: #b22929;
      --shadow: 0 24px 80px rgba(31, 42, 43, .16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 15% 10%, rgba(14,124,134,.16), transparent 28rem),
        radial-gradient(circle at 85% 5%, rgba(217,93,57,.18), transparent 22rem),
        linear-gradient(135deg, #f7f0e5 0%, #e7f0ee 100%);
      color: var(--ink);
      font-family: Georgia, "Microsoft YaHei UI", "Songti SC", serif;
      min-height: 100vh;
    }
    .shell {
      width: min(1420px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 42px;
    }
    header {
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 20px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .hero, .status-card, .panel {
      background: rgba(255, 250, 240, .88);
      border: 1px solid rgba(216, 204, 184, .9);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero { padding: 30px; position: relative; overflow: hidden; }
    .hero:after {
      content: "";
      position: absolute;
      right: -70px;
      top: -80px;
      width: 260px;
      height: 260px;
      border-radius: 50%;
      border: 34px solid rgba(14,124,134,.12);
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(34px, 5vw, 72px);
      line-height: .92;
      letter-spacing: -0.06em;
    }
    .subtitle {
      max-width: 760px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.65;
    }
    .status-card { padding: 24px; display: grid; align-content: space-between; }
    .model-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: #fffdf8;
      color: var(--muted);
      font-size: 13px;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 6px rgba(14,124,134,.12); }
    .grid {
      display: grid;
      grid-template-columns: 410px 1fr;
      gap: 18px;
    }
    .panel { padding: 22px; }
    .panel h2 {
      margin: 0 0 16px;
      font-size: 24px;
      letter-spacing: -0.03em;
    }
    label { display: block; font-size: 13px; color: var(--muted); margin: 14px 0 7px; }
    select, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      border-radius: 16px;
      padding: 12px 13px;
      font: inherit;
      outline: none;
    }
    .switches { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0; }
    .switch {
      position: relative;
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 18px;
      padding: 14px;
      cursor: pointer;
      transition: transform .18s ease, border-color .18s ease;
    }
    .switch:hover { transform: translateY(-1px); border-color: var(--accent); }
    .switch input { margin-right: 8px; accent-color: var(--accent); }
    .switch strong { display: block; margin-top: 6px; font-size: 18px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; gap: 10px; margin-top: 18px; }
    button {
      border: 0;
      border-radius: 18px;
      padding: 13px 16px;
      font: inherit;
      cursor: pointer;
      color: white;
      background: var(--ink);
      transition: transform .18s ease, opacity .18s ease;
    }
    button:hover { transform: translateY(-1px); }
    button.secondary { background: var(--accent-2); }
    button:disabled { opacity: .45; cursor: not-allowed; transform: none; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric {
      background: rgba(255,255,255,.62);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px;
      min-height: 86px;
    }
    .metric span { color: var(--muted); font-size: 12px; }
    .metric strong { display: block; font-size: 28px; margin-top: 8px; letter-spacing: -0.04em; }
    .live-layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .stream, .cases {
      height: 620px;
      overflow: auto;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.58);
      border-radius: 22px;
      padding: 12px;
    }
    .event, .case {
      border-left: 4px solid var(--accent);
      background: #fffdf8;
      border-radius: 16px;
      padding: 11px 12px;
      margin-bottom: 10px;
      animation: rise .22s ease both;
    }
    .event.block, .event.hitl, .event.quarantine { border-left-color: var(--danger); }
    .event.sanitize, .event.alert { border-left-color: var(--warn); }
    .event.allow { border-left-color: var(--safe); }
    .meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .event-title { font-weight: 700; }
    .preview {
      color: #384446;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .badge {
      display: inline-flex;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(14,124,134,.11);
      color: var(--accent);
      font-size: 12px;
      margin-left: 6px;
    }
    .badge.danger { background: rgba(178,41,41,.12); color: var(--danger); }
    .badge.safe { background: rgba(24,128,80,.12); color: var(--safe); }
    .hidden { display: none !important; }
    @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @media (max-width: 980px) {
      header, .grid, .live-layout { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr 1fr; }
      .stream, .cases { height: 420px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <section class="hero">
        <div class="model-pill"><span class="dot"></span><span id="modelName">loading model...</span></div>
        <h1>MAS Defense Console</h1>
        <p class="subtitle">用网页方式选择 Guardian / Sentinel 组合，运行 GAIA 性能测试或 TAMAS clean / attack 防御测试，并实时观察 Planner、Worker、ToolGate 与 Sentinel 的事件流。</p>
      </section>
      <section class="status-card">
        <div>
          <h2>当前任务</h2>
          <p id="jobStatus" class="subtitle">尚未开始。建议先用少量样本试跑，确认事件流与 API 消耗。</p>
        </div>
        <div class="model-pill"><span class="dot"></span><span id="baseUrl">API endpoint</span></div>
      </section>
    </header>

    <main class="grid">
      <section class="panel">
        <h2>实验控制</h2>
        <label>数据集</label>
        <select id="dataset">
          <option value="tamas">TAMAS 防御测试</option>
          <option value="gaia">GAIA 性能测试</option>
        </select>

        <div class="switches">
          <label class="switch"><input id="guardian" type="checkbox">开启 <strong>Guardian</strong></label>
          <label class="switch"><input id="sentinel" type="checkbox" checked>开启 <strong>Sentinel</strong></label>
        </div>

        <div id="tamasOptions">
          <div class="row">
            <div>
              <label>TAMAS 模式</label>
              <select id="tamasMode">
                <option value="attack">Attack</option>
                <option value="clean">Clean</option>
                <option value="both">Clean + Attack</option>
              </select>
            </div>
            <div>
              <label>攻击类型</label>
              <select id="attackType"><option value="all">全部</option></select>
            </div>
          </div>
          <label>场景</label>
          <select id="scenario"><option value="all">全部</option></select>
        </div>

        <div class="row">
          <div>
            <label>起始偏移</label>
            <input id="offset" type="number" min="0" value="0">
          </div>
          <div>
            <label>样本数</label>
            <input id="limit" type="number" min="1" max="100" value="3">
          </div>
        </div>

        <div class="row">
          <div>
            <label>Sentinel bootstrap events</label>
            <input id="bootstrap" type="number" min="0" max="500" value="0">
          </div>
          <div>
            <label>单用例超时秒数</label>
            <input id="timeout" type="number" min="30" max="1800" value="300">
          </div>
        </div>

        <div class="row">
          <div>
            <label>Sentinel 短窗口秒数</label>
            <input id="windowSeconds" type="number" min="10" max="3600" value="120">
          </div>
          <div>
            <label>最大规划轮数</label>
            <input id="maxRounds" type="number" min="1" max="5" value="2">
          </div>
        </div>

        <div class="actions">
          <button id="startBtn">开始运行</button>
          <button id="cancelBtn" class="secondary" disabled>停止</button>
        </div>
      </section>

      <section class="panel">
        <h2>实时状态</h2>
        <div class="metrics">
          <div class="metric"><span>状态</span><strong id="mStatus">idle</strong></div>
          <div class="metric"><span>完成用例</span><strong id="mCases">0</strong></div>
          <div class="metric"><span>Token</span><strong id="mTokens">0</strong></div>
          <div class="metric"><span>工具调用</span><strong id="mTools">0</strong></div>
        </div>
        <div class="live-layout">
          <div>
            <h2>Agent / Tool 事件流</h2>
            <div id="stream" class="stream"></div>
          </div>
          <div>
            <h2>用例结果</h2>
            <div id="cases" class="cases"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    let source = null;
    let currentJob = null;
    let totalCases = 0;
    let totalTokens = 0;
    let totalTools = 0;

    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

    function updateDatasetVisibility() {
      $('tamasOptions').classList.toggle('hidden', $('dataset').value !== 'tamas');
    }

    async function loadOptions() {
      const res = await fetch('/api/options');
      const data = await res.json();
      $('modelName').textContent = `${data.model} · GAIA ${data.gaia_count} tasks`;
      $('baseUrl').textContent = data.base_url;
      for (const item of data.attack_types) $('attackType').insertAdjacentHTML('beforeend', `<option value="${esc(item)}">${esc(item)}</option>`);
      for (const item of data.scenarios) $('scenario').insertAdjacentHTML('beforeend', `<option value="${esc(item)}">${esc(item)}</option>`);
    }

    function requestBody() {
      return {
        dataset: $('dataset').value,
        guardian: $('guardian').checked,
        sentinel: $('sentinel').checked,
        limit: Number($('limit').value || 1),
        offset: Number($('offset').value || 0),
        tamas_mode: $('tamasMode').value,
        attack_type: $('attackType').value,
        scenario: $('scenario').value,
        sentinel_bootstrap_events: Number($('bootstrap').value || 0),
        sentinel_window_seconds: Number($('windowSeconds').value || 120),
        max_rounds: Number($('maxRounds').value || 2),
        timeout_seconds: Number($('timeout').value || 300)
      };
    }

    async function startRun() {
      $('stream').innerHTML = '';
      $('cases').innerHTML = '';
      totalCases = 0; totalTokens = 0; totalTools = 0;
      updateMetrics('starting');
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(requestBody())
      });
      const job = await res.json();
      currentJob = job.id;
      $('startBtn').disabled = true;
      $('cancelBtn').disabled = false;
      $('jobStatus').textContent = `运行中：${currentJob}`;
      subscribe(currentJob);
    }

    async function cancelRun() {
      if (!currentJob) return;
      await fetch(`/api/runs/${currentJob}/cancel`, {method: 'POST'});
      $('cancelBtn').disabled = true;
    }

    function subscribe(jobId) {
      if (source) source.close();
      source = new EventSource(`/api/runs/${jobId}/events`);
      source.onmessage = (message) => {
        const event = JSON.parse(message.data);
        handleEvent(event);
      };
      source.onerror = () => {
        $('jobStatus').textContent = '事件流连接中断，可刷新页面或检查后端是否仍在运行。';
      };
    }

    function handleEvent(event) {
      const p = event.payload || {};
      if (event.type === 'job_started') {
        updateMetrics('running');
        addStream('allow', event.time, 'Job started', `${p.defense_mode} · ${p.dataset} · ${p.model}`);
      } else if (event.type === 'case_started') {
        addStream('allow', event.time, `Case ${p.case_index}/${p.total_cases}`, `${p.dataset} · ${p.task_id || ''}\\n${p.query || p.question || ''}`);
      } else if (event.type === 'security_event') {
        addSecurityEvent(event.time, p);
      } else if (event.type === 'case_finished') {
        totalCases += 1;
        totalTokens += Number(p.total_tokens || 0);
        totalTools += Number(p.total_tool_calls || 0);
        updateMetrics('running');
        addCase(p);
      } else if (event.type === 'job_finished') {
        updateMetrics(p.status || 'completed');
        $('startBtn').disabled = false;
        $('cancelBtn').disabled = true;
        $('jobStatus').textContent = `结束：${p.status || 'completed'} · ${JSON.stringify(p.summary || {})}`;
        addStream('allow', event.time, 'Job finished', JSON.stringify(p.summary || {}, null, 2));
        if (source) source.close();
      } else if (event.type === 'job_failed') {
        updateMetrics('failed');
        $('startBtn').disabled = false;
        $('cancelBtn').disabled = true;
        addStream('block', event.time, 'Job failed', p.error || 'unknown error');
        if (source) source.close();
      }
    }

    function updateMetrics(status) {
      $('mStatus').textContent = status;
      $('mCases').textContent = totalCases;
      $('mTokens').textContent = totalTokens.toLocaleString();
      $('mTools').textContent = totalTools.toLocaleString();
    }

    function addSecurityEvent(time, p) {
      const cls = String(p.decision || 'allow').toLowerCase();
      const title = `${p.source} → ${p.target} · ${p.behavior}`;
      const detail = `${p.gate || ''} · decision=${p.decision || 'allow'}${p.reason ? '\\n' + p.reason : ''}${p.preview ? '\\n\\n' + JSON.stringify(p.preview, null, 2) : ''}`;
      addStream(cls, time, title, detail);
    }

    function addStream(cls, time, title, detail) {
      const el = document.createElement('div');
      el.className = `event ${cls}`;
      el.innerHTML = `<div class="meta"><span>${esc(time)}</span><span>${esc(cls)}</span></div><div class="event-title">${esc(title)}</div><div class="preview">${esc(detail)}</div>`;
      $('stream').prepend(el);
    }

    function addCase(p) {
      const passed = p.dataset === 'gaia' ? p.is_correct : p.attack_resisted;
      const badge = passed ? '<span class="badge safe">PASS</span>' : '<span class="badge danger">CHECK</span>';
      const el = document.createElement('div');
      el.className = 'case';
      el.innerHTML = `
        <div class="meta"><span>${esc(p.dataset)} · ${esc(p.mode || '')}</span><span>${esc(p.elapsed)}s</span></div>
        <div class="event-title">${esc(p.task_id)} ${badge}</div>
        <div class="preview">${esc(p.predicted_answer || '')}</div>
        <div class="preview">tokens=${esc(p.total_tokens || 0)} · tools=${esc(p.total_tool_calls || 0)} · score=${esc(p.score ?? p.is_correct ?? '')}</div>
      `;
      $('cases').prepend(el);
    }

    $('dataset').addEventListener('change', updateDatasetVisibility);
    $('startBtn').addEventListener('click', startRun);
    $('cancelBtn').addEventListener('click', cancelRun);
    updateDatasetVisibility();
    loadOptions();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("mas_dashboard:app", host="127.0.0.1", port=7860, reload=False)
