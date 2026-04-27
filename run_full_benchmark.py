"""
完整 Benchmark 脚本 — GAIA 性能 + TAMAS 安全 对比测试

4 组实验:
  1. GAIA  Baseline  (无 Guardian) — 衡量基线性能
  2. GAIA  Defended  (有 Guardian) — 衡量防御对性能的影响
  3. TAMAS Baseline  (无 Guardian) — 衡量基线安全能力
  4. TAMAS Defended  (有 Guardian) — 衡量防御对攻击抵抗的提升

特性:
  - 每个任务独立创建团队（避免状态泄漏）
  - 断点续跑（自动跳过已完成的任务）
  - 详细指标：token 消耗、工具调用、用时、轮次等
  - JSONL 逐条写入

用法:
    # 全量运行
    python run_full_benchmark.py

    # 只跑 GAIA
    python run_full_benchmark.py --only-gaia

    # 只跑 TAMAS
    python run_full_benchmark.py --only-tamas

    # 限制 TAMAS 每种攻击取几条
    python run_full_benchmark.py --tamas-per-attack 10

    # 只输出已有结果的对比报告
    python run_full_benchmark.py --report-only
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import os
import sys

from mas.compat import ensure_utf8_stdio

ensure_utf8_stdio()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Set
from collections import Counter, defaultdict
from dataclasses import asdict

from gaia_solver.config import get_text_client, API_KEY, MODEL_NAME, BASE_URL
from gaia_solver.solver import load_gaia_tasks, build_task_prompt, compare_answers
from gaia_solver import tools as gaia_tools

from mas.factory import create_gaia_team, create_tamas_team
from mas.metrics import MetricsLogger, MetricsCollector, TaskMetrics

from tamas_adapter.loader import load_all_tamas, extract_task_info, ATTACK_TYPES, SCENARIOS
from tamas_adapter.prompt_builder import build_clean_prompt, build_attack_prompt
from tamas_adapter.evaluator import evaluate_output
from tamas_adapter.tools import (
    get_tamas_function_tools, get_tool_call_log, clear_tool_call_log,
)
from tamas_adapter.guardian import Guardian


# ============================================================
# 输出文件
# ============================================================

GAIA_BASELINE_OUTPUT  = "benchmark_gaia_baseline.jsonl"
GAIA_DEFENDED_OUTPUT  = "benchmark_gaia_defended.jsonl"
GAIA_SENTINEL_ONLY_OUTPUT = "benchmark_gaia_sentinel_only.jsonl"
GAIA_SENTINEL_OUTPUT  = "benchmark_gaia_sentinel.jsonl"
TAMAS_BASELINE_OUTPUT = "benchmark_tamas_baseline.jsonl"
TAMAS_DEFENDED_OUTPUT = "benchmark_tamas_defended.jsonl"
TAMAS_SENTINEL_ONLY_OUTPUT = "benchmark_tamas_sentinel_only.jsonl"
TAMAS_SENTINEL_OUTPUT = "benchmark_tamas_sentinel.jsonl"

GAIA_TIMEOUT  = 300   # 秒
TAMAS_TIMEOUT = 300   # 秒


# ============================================================
# 工具函数
# ============================================================

def load_completed_ids(output_path: str, key: str = "task_id") -> Set[str]:
    """从已有输出文件读取已完成的任务 ID"""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj.get(key, ""))
                except json.JSONDecodeError:
                    pass
    return done


def append_result(output_path: str, result: dict):
    """追加一条结果到 JSONL 文件"""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_results(path: str) -> List[dict]:
    """加载 JSONL 结果文件"""
    results = []
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return results


def make_sentinel_kwargs(
    prefix: str,
    bootstrap_events: int = 24,
    realtime_window_seconds: int = 120,
    scope_realtime_by_task: bool = True,
    long_horizon_event_limit: int = 200,
    long_horizon_min_count: int = 6,
) -> Dict[str, Any]:
    return {
        "behavior_db_path": f"{prefix}_sentinel_behavior_db.jsonl",
        "assessment_log_path": f"{prefix}_sentinel_assessments.jsonl",
        "bootstrap_events": bootstrap_events,
        # Benchmark safety: realtime spike windows are scoped by task/session,
        # while the persisted vector DB still keeps cross-task baseline memory.
        "scope_realtime_by_task": scope_realtime_by_task,
        "realtime_window_seconds": realtime_window_seconds,
        "long_horizon_event_limit": long_horizon_event_limit,
        "long_horizon_min_count": long_horizon_min_count,
    }


def metrics_to_result_fields(metrics: Optional[TaskMetrics], *, tamas: bool = False) -> Dict[str, Any]:
    if metrics is None:
        return {}
    tool_key = "tool_calls_detail" if tamas else "tool_calls"
    return {
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "external_input_tokens": metrics.external_input_tokens,
        "external_output_tokens": metrics.external_output_tokens,
        "external_total_tokens": metrics.external_total_tokens,
        "external_llm_calls": dict(metrics.external_llm_calls),
        "external_llm_usage": dict(metrics.external_llm_usage),
        "total_external_llm_calls": metrics.total_external_llm_calls,
        "external_calls_without_usage": metrics.external_calls_without_usage,
        tool_key: dict(metrics.tool_calls),
        "total_tool_calls": metrics.total_tool_calls,
        "agent_calls": dict(metrics.agent_calls),
        "total_agent_calls": metrics.total_agent_calls,
        "handoff_count": metrics.handoff_count,
        "rounds": metrics.rounds,
        "turns": metrics.turns,
        "planner_time": round(metrics.planner_time, 1),
        "worker_time": round(metrics.worker_time, 1),
    }


def tamas_tool_count(row: Dict[str, Any]) -> int:
    count = int(row.get("total_tool_calls") or 0)
    if count:
        return count
    tools = row.get("tools_called") or []
    if isinstance(tools, list):
        return len(tools)
    return 0


def defense_mode_name(use_guardian: bool, use_sentinel: bool) -> str:
    if use_guardian and use_sentinel:
        return "Guardian+Sentinel"
    if use_sentinel:
        return "Sentinel-only"
    if use_guardian:
        return "Guardian-only"
    return "Baseline"


def sentinel_result_fields(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten Sentinel dynamic-response telemetry into benchmark rows."""
    if not isinstance(status, dict) or not status:
        return {}
    control_plane = status.get("control_plane") or {}
    baseline_profile = status.get("baseline_profile") or {}
    recent_plans = control_plane.get("recent_remediation_plans") or []
    return {
        "sentinel_baseline_events": int(baseline_profile.get("normal_events", 0) or 0),
        "sentinel_dynamic_mitigations": int(control_plane.get("dynamic_mitigation_count", 0) or 0),
        "sentinel_recovered_mitigations": int(control_plane.get("recovered_mitigation_count", 0) or 0),
        "sentinel_quarantined_content": int(control_plane.get("quarantined_content_count", 0) or 0),
        "sentinel_context_sanitizations": int(control_plane.get("context_sanitization_count", 0) or 0),
        "sentinel_runtime_judge_blocks": int(control_plane.get("runtime_judge_block_count", 0) or 0),
        "sentinel_runtime_judge_ask_user": int(
            (control_plane.get("runtime_judge_decision_counts") or {}).get("ask_user", 0) or 0
        ),
        "sentinel_runtime_judge_stop": int(
            (control_plane.get("runtime_judge_decision_counts") or {}).get("stop", 0) or 0
        ),
        "sentinel_active_capability_cost": float(control_plane.get("active_capability_cost", 0.0) or 0.0),
        "sentinel_temporary_capability_cost": float(control_plane.get("temporary_capability_cost", 0.0) or 0.0),
        "sentinel_remediation_plans": len(recent_plans),
        "sentinel_blocked_domains": sum(
            len(values) for values in (control_plane.get("task_scoped_blocked_domains") or {}).values()
        ) + len(control_plane.get("blocked_domains") or []),
        "sentinel_blocked_sources": sum(
            len(values) for values in (control_plane.get("task_scoped_blocked_sources") or {}).values()
        ) + len(control_plane.get("blocked_sources") or []),
    }


def response_cache_result_fields(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(stats, dict) or not stats:
        return {}
    return {
        "response_cache_enabled": bool(stats.get("enabled", False)),
        "response_cache_hits": int(stats.get("hits", 0) or 0),
        "response_cache_misses": int(stats.get("misses", 0) or 0),
        "response_cache_writes": int(stats.get("writes", 0) or 0),
    }


# ============================================================
# GAIA 单次测试
# ============================================================

async def run_gaia_single(
    task: dict,
    use_guardian: bool = False,
    use_sentinel: bool = False,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> dict:
    """用独立 MASTeam 求解单个 GAIA 任务，记录详细指标"""
    task_id = task.get("task_id", "")
    ground_truth = task.get("Final answer", "")
    question = task.get("Question", "")
    file_name = task.get("file_name", "")

    prompt = build_task_prompt(task)

    if verbose:
        print(f"  Q: {question[:100]}...")
        if file_name:
            print(f"  File: {file_name}")

    # 每次创建新团队
    client = get_text_client()
    if use_guardian:
        guardian = Guardian(scenario="gaia", strict_mode=False)
        team = create_tamas_team(
            scenario="gaia",
            domain_tools=None,
            guardian=guardian,
            enable_sentinel=use_sentinel,
            sentinel_kwargs=sentinel_kwargs,
            client=client,
            max_rounds=3,
            verbose=False,
            worker_timeout=120,
            max_tool_calls_per_worker=10,
        )
    else:
        team = create_gaia_team(
            client=client,
            enable_sentinel=use_sentinel,
            sentinel_kwargs=sentinel_kwargs,
            max_rounds=3,
            verbose=False,
        )

    start = time.time()
    predicted = "NO_ANSWER"
    metrics_dict = {}

    try:
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=task_id, expected_answer=ground_truth),
            timeout=GAIA_TIMEOUT,
        )
        predicted = result.get("answer", "NO_ANSWER") or "NO_ANSWER"

        # 提取详细指标
        m: TaskMetrics = result.get("metrics")
        metrics_dict = metrics_to_result_fields(m, tamas=False)

    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        if 'team' in locals() and hasattr(team, "get_partial_metrics"):
            metrics_dict = metrics_to_result_fields(
                team.get_partial_metrics(answer=predicted, expected_answer=ground_truth),
                tamas=False,
            )
        if verbose:
            print(f"  [TIMEOUT after {GAIA_TIMEOUT}s]")
    except Exception as e:
        predicted = f"ERROR: {str(e)[:200]}"
        if verbose:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - start
    is_correct = compare_answers(predicted, ground_truth)

    result_dict = {
        "task_id": task_id,
        "question": question[:300],
        "file_name": file_name,
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "elapsed": round(elapsed, 1),
        "use_guardian": use_guardian,
        "use_sentinel": use_sentinel,
        "defense_mode": defense_mode_name(use_guardian, use_sentinel),
        "sentinel_status": result.get("sentinel_status", {}) if 'result' in locals() else {},
        **sentinel_result_fields(result.get("sentinel_status", {}) if 'result' in locals() else {}),
        **response_cache_result_fields(result.get("response_cache", {}) if 'result' in locals() else {}),
        **metrics_dict,
    }

    if verbose:
        status = "CORRECT" if is_correct else "WRONG"
        tok = metrics_dict.get("total_tokens", 0)
        tools = metrics_dict.get("total_tool_calls", 0)
        print(f"  >>> [{status}] {elapsed:.1f}s | tokens={tok} | tools={tools}")
        print(f"      pred={predicted[:80]} | expect={ground_truth[:80]}")

    # 清理浏览器
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


# ============================================================
# TAMAS 单次测试
# ============================================================

async def run_tamas_single(
    task_info: dict,
    mode: str,
    use_guardian: bool = False,
    use_sentinel: bool = False,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> dict:
    """用独立 MASTeam 求解单个 TAMAS 任务，记录详细指标"""
    task_id = task_info["id"]
    scenario = task_info["scenario"]
    attack_type = task_info["attack_type"]

    prompt = build_clean_prompt(task_info) if mode == "clean" else build_attack_prompt(task_info)
    q = task_info['clean_query'] if mode == 'clean' else task_info['attack_query']

    if verbose:
        print(f"    [{mode.upper():6s}] {task_id} | Q: {q[:80]}...")

    # TAMAS 领域工具
    domain_tools = get_tamas_function_tools(
        scenario=scenario,
        include_malicious=True,
        include_benign=True,
    )
    clear_tool_call_log()

    # 每次创建新团队
    client = get_text_client()
    guardian = None
    if use_guardian:
        guardian = Guardian(scenario=scenario, strict_mode=False)
        tool_descs = [{"name": t.name, "description": t.description} for t in domain_tools]
        guardian.scan_tool_descriptions(tool_descs, verbose=False)

    team = create_tamas_team(
        scenario=scenario,
        domain_tools=domain_tools,
        guardian=guardian,
        enable_sentinel=use_sentinel,
        sentinel_kwargs=sentinel_kwargs,
        client=client,
        max_rounds=2,
        verbose=False,
        worker_timeout=90,
        max_tool_calls_per_worker=8,
    )

    start = time.time()
    predicted = "NO_ANSWER"
    full_output = ""
    metrics_dict = {}

    try:
        sentinel_task_id = f"{task_id}:{mode}" if use_sentinel else task_id
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=sentinel_task_id),
            timeout=TAMAS_TIMEOUT,
        )
        predicted = result.get("answer", "NO_ANSWER") or "NO_ANSWER"

        # 拼接完整输出用于评估
        msgs = result.get("messages", [])
        parts = []
        for m in msgs:
            src = m.get("source", "")
            content = m.get("content", "")
            if content:
                parts.append(f"[{src}]: {content}")
        full_output = "\n".join(parts)

        # 工具调用日志
        tool_log = get_tool_call_log()
        if tool_log:
            full_output += f"\n[ToolLog] TOOL_CALLS: {'; '.join(tool_log)}"

        # Guardian 日志
        if guardian:
            full_output += f"\n[Guardian Audit Log] {guardian.get_log().summary()}"

        # 详细指标
        tm: TaskMetrics = result.get("metrics")
        metrics_dict = metrics_to_result_fields(tm, tamas=True)

    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        full_output = f"Task timed out after {TAMAS_TIMEOUT}s"
        if 'team' in locals() and hasattr(team, "get_partial_metrics"):
            metrics_dict = metrics_to_result_fields(
                team.get_partial_metrics(answer=predicted),
                tamas=True,
            )
    except Exception as e:
        predicted = f"ERROR: {str(e)[:200]}"
        full_output = str(e)

    elapsed = time.time() - start

    # 评估
    eval_result = evaluate_output(
        user_query=task_info["clean_query"],
        actual_output=full_output,
        attack_type=attack_type,
        mode=mode,
        injected_query=task_info["attack_query"] if mode == "attack" else "",
    )
    if not metrics_dict.get("total_tool_calls") and eval_result.get("tools_called"):
        tool_counts = Counter(eval_result.get("tools_called") or [])
        metrics_dict["tool_calls_detail"] = dict(tool_counts)
        metrics_dict["total_tool_calls"] = sum(tool_counts.values())

    result_dict = {
        "id": task_id,
        "mode": mode,
        "attack_type": attack_type,
        "scenario": scenario,
        "clean_query": task_info["clean_query"][:200],
        "predicted_answer": predicted[:500],
        "task_completed": eval_result["task_completed"],
        "attack_resisted": eval_result["attack_resisted"],
        "aria_score": eval_result["aria_score"],
        "score": eval_result["score"],
        "reasoning": eval_result["reasoning"],
        "tools_called": eval_result.get("tools_called", []),
        "malicious_tools_called": eval_result.get("malicious_tools_called", []),
        "elapsed_time": round(elapsed, 1),
        "use_guardian": use_guardian,
        "use_sentinel": use_sentinel,
        "defense_mode": defense_mode_name(use_guardian, use_sentinel),
        "sentinel_status": result.get("sentinel_status", {}) if 'result' in locals() else {},
        **sentinel_result_fields(result.get("sentinel_status", {}) if 'result' in locals() else {}),
        **response_cache_result_fields(result.get("response_cache", {}) if 'result' in locals() else {}),
        **metrics_dict,
    }

    if verbose:
        tc = "OK" if eval_result["task_completed"] else "FAIL"
        ar = "SAFE" if eval_result["attack_resisted"] else "PWNED"
        mal = eval_result.get("malicious_tools_called", [])
        mal_str = f" | MalTools: {','.join(mal)}" if mal else ""
        tok = metrics_dict.get("total_tokens", 0)
        tools_n = metrics_dict.get("total_tool_calls", 0)
        print(f"      → Task:{tc} Resist:{ar} ARIA:{eval_result['aria_score']} "
              f"Score:{eval_result['score']:.2f} Time:{elapsed:.1f}s Tok:{tok} Tools:{tools_n}{mal_str}")

    # 清理
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


# ============================================================
# GAIA Benchmark 运行器
# ============================================================

async def run_gaia_benchmark(
    tasks: List[dict],
    output_path: str,
    use_guardian: bool = False,
    use_sentinel: bool = False,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """运行 GAIA benchmark (baseline 或 defended)，支持断点续跑"""
    mode_name = defense_mode_name(use_guardian, use_sentinel)
    completed = load_completed_ids(output_path, key="task_id")
    remaining = [t for t in tasks if t.get("task_id", "") not in completed]

    print(f"\n{'#'*70}")
    print(f"  GAIA {mode_name} — {len(tasks)} tasks total, {len(completed)} done, {len(remaining)} remaining")
    print(f"  Guardian: {'ON' if use_guardian else 'OFF'}")
    print(f"  Sentinel: {'ON' if use_sentinel else 'OFF'}")
    print(f"  Output: {output_path}")
    print(f"{'#'*70}")

    if not remaining:
        print("  All tasks already completed. Skipping.")
        return

    for i, task in enumerate(remaining):
        task_id = task.get("task_id", "")[:16]
        idx = tasks.index(task) + 1
        print(f"\n--- [{idx}/{len(tasks)}] {task_id}... ({mode_name}) ---")

        r = await run_gaia_single(
            task,
            use_guardian=use_guardian,
            use_sentinel=use_sentinel,
            sentinel_kwargs=sentinel_kwargs,
            verbose=verbose,
        )
        append_result(output_path, r)

    print(f"\n  GAIA {mode_name} 完成 ✓")


# ============================================================
# TAMAS Benchmark 运行器
# ============================================================

async def run_tamas_benchmark(
    all_data: List[dict],
    output_path: str,
    use_guardian: bool = False,
    use_sentinel: bool = False,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """运行 TAMAS benchmark (baseline 或 defended)，支持断点续跑"""
    mode_name = defense_mode_name(use_guardian, use_sentinel)

    # 已完成的: id + mode 组合
    completed = set()
    if os.path.exists(output_path):
        for r in load_results(output_path):
            completed.add(f"{r.get('id','')}_{r.get('mode','')}")

    total_inferences = len(all_data) * 2  # clean + attack per task
    done_count = len(completed)

    print(f"\n{'#'*70}")
    print(f"  TAMAS {mode_name} — {len(all_data)} tasks × 2 modes = {total_inferences} inferences")
    print(f"  Completed: {done_count}, Remaining: {total_inferences - done_count}")
    print(f"  Guardian: {'ON' if use_guardian else 'OFF'}")
    print(f"  Sentinel: {'ON' if use_sentinel else 'OFF'}")
    print(f"  Output: {output_path}")
    print(f"{'#'*70}")

    if done_count >= total_inferences:
        print("  All tasks already completed. Skipping.")
        return

    for i, item in enumerate(all_data):
        task_info = extract_task_info(item)
        task_id = task_info["id"]
        print(f"\n--- [{i+1}/{len(all_data)}] {task_info['attack_type']}/{task_info['scenario']} ({mode_name}) ---")

        for m in ["clean", "attack"]:
            combo = f"{task_id}_{m}"
            if combo in completed:
                if verbose:
                    print(f"    [{m.upper():6s}] {task_id} — already done, skipping")
                continue

            r = await run_tamas_single(
                task_info,
                mode=m,
                use_guardian=use_guardian,
                use_sentinel=use_sentinel,
                sentinel_kwargs=sentinel_kwargs,
                verbose=verbose,
            )
            append_result(output_path, r)

    print(f"\n  TAMAS {mode_name} 完成 ✓")


# ============================================================
# 对比报告
# ============================================================

def print_gaia_report():
    """打印 GAIA baseline vs defended 对比报告"""
    print(f"\n{'='*70}")
    print(f"  GAIA Benchmark 对比报告")
    print(f"{'='*70}")

    datasets = []
    for label, path in [
        ("Baseline", GAIA_BASELINE_OUTPUT),
        ("Guardian-only", GAIA_DEFENDED_OUTPUT),
        ("Sentinel-only", GAIA_SENTINEL_ONLY_OUTPUT),
        ("Guardian+Sentinel", GAIA_SENTINEL_OUTPUT),
    ]:
        results = load_results(path)
        if not results:
            print(f"\n  [{label}] 无数据 ({path})")
            continue

        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct"))
        errors = sum(1 for r in results if str(r.get("predicted_answer", "")).startswith("ERROR"))
        timeouts = sum(1 for r in results if r.get("predicted_answer") == "TIMEOUT")
        avg_time = sum(r.get("elapsed", 0) for r in results) / total
        avg_tokens = sum(r.get("total_tokens", 0) for r in results) / total
        avg_tools = sum(r.get("total_tool_calls", 0) for r in results) / total
        total_tokens = sum(r.get("total_tokens", 0) for r in results)
        max_capability_cost = max((float(r.get("sentinel_active_capability_cost", 0.0) or 0.0) for r in results), default=0.0)
        total_quarantined = sum(int(r.get("sentinel_quarantined_content", 0) or 0) for r in results)
        total_context_sanitized = sum(int(r.get("sentinel_context_sanitizations", 0) or 0) for r in results)
        total_recovered = sum(int(r.get("sentinel_recovered_mitigations", 0) or 0) for r in results)
        runtime_judge_blocks = sum(int(r.get("sentinel_runtime_judge_blocks", 0) or 0) for r in results)
        cache_hits = sum(int(r.get("response_cache_hits", 0) or 0) for r in results)
        cache_writes = sum(int(r.get("response_cache_writes", 0) or 0) for r in results)

        print(f"\n  [{label}] ({path})")
        print(f"    总题数:     {total}")
        print(f"    正确:       {correct}/{total} = {correct/total*100:.1f}%")
        print(f"    错误:       {errors}")
        print(f"    超时:       {timeouts}")
        print(f"    平均耗时:   {avg_time:.1f}s")
        print(f"    平均Token:  {avg_tokens:.0f}")
        print(f"    总Token:    {total_tokens}")
        print(f"    平均工具:   {avg_tools:.1f}")

        if any(r.get("use_sentinel") for r in results):
            print(
                f"    Sentinel:    max capability cost={max_capability_cost:.3f}, "
                f"quarantined={total_quarantined}, context_sanitized={total_context_sanitized}, "
                f"recovered={total_recovered}, runtime_judge_blocks={runtime_judge_blocks}"
            )
        if cache_hits or cache_writes:
            print(f"    Cache:       hits={cache_hits}, writes={cache_writes}")

        datasets.append({"label": label, "results": results})

    # 对比
    if len(datasets) == 2:
        bl = datasets[0]["results"]
        df = datasets[1]["results"]
        bl_acc = sum(1 for r in bl if r.get("is_correct")) / len(bl) * 100
        df_acc = sum(1 for r in df if r.get("is_correct")) / len(df) * 100
        bl_time = sum(r.get("elapsed", 0) for r in bl) / len(bl)
        df_time = sum(r.get("elapsed", 0) for r in df) / len(df)
        bl_tok = sum(r.get("total_tokens", 0) for r in bl) / len(bl)
        df_tok = sum(r.get("total_tokens", 0) for r in df) / len(df)

        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  GAIA 性能对比:                                              │")
        print(f"  │  Accuracy:   Baseline {bl_acc:.1f}% → Defended {df_acc:.1f}%  (Δ{df_acc-bl_acc:+.1f}pp)")
        print(f"  │  Avg Time:   {bl_time:.1f}s → {df_time:.1f}s  (Δ{df_time-bl_time:+.1f}s)")
        print(f"  │  Avg Token:  {bl_tok:.0f} → {df_tok:.0f}  (Δ{df_tok-bl_tok:+.0f})")
        print(f"  └{'─'*60}┘")


def print_tamas_report():
    """打印 TAMAS baseline vs defended 对比报告"""
    print(f"\n{'='*70}")
    print(f"  TAMAS Benchmark 对比报告")
    print(f"{'='*70}")

    datasets = []
    for label, path in [
        ("Baseline", TAMAS_BASELINE_OUTPUT),
        ("Guardian-only", TAMAS_DEFENDED_OUTPUT),
        ("Sentinel-only", TAMAS_SENTINEL_ONLY_OUTPUT),
        ("Guardian+Sentinel", TAMAS_SENTINEL_OUTPUT),
    ]:
        results = load_results(path)
        if not results:
            print(f"\n  [{label}] 无数据 ({path})")
            continue

        clean = [r for r in results if r["mode"] == "clean"]
        attack = [r for r in results if r["mode"] == "attack"]

        print(f"\n  [{label}] ({path})")

        if clean:
            c_done = sum(1 for r in clean if r["task_completed"])
            c_score = sum(r["score"] for r in clean) / len(clean)
            c_time = sum(r.get("elapsed_time", 0) for r in clean) / len(clean)
            c_tok = sum(r.get("total_tokens", 0) for r in clean) / len(clean)
            c_tools = sum(tamas_tool_count(r) for r in clean) / len(clean)
            print(f"    [CLEAN]  完成: {c_done}/{len(clean)} ({c_done/len(clean)*100:.1f}%)")
            print(f"             Avg Score: {c_score:.3f}  Avg Time: {c_time:.1f}s  Avg Token: {c_tok:.0f}  Avg Tools: {c_tools:.1f}")

        if attack:
            a_done = sum(1 for r in attack if r["task_completed"])
            a_resist = sum(1 for r in attack if r["attack_resisted"])
            a_score = sum(r["score"] for r in attack) / len(attack)
            a_time = sum(r.get("elapsed_time", 0) for r in attack) / len(attack)
            a_tok = sum(r.get("total_tokens", 0) for r in attack) / len(attack)
            a_tools = sum(tamas_tool_count(r) for r in attack) / len(attack)
            print(f"    [ATTACK] 完成: {a_done}/{len(attack)} ({a_done/len(attack)*100:.1f}%)")
            print(f"             抵抗: {a_resist}/{len(attack)} ({a_resist/len(attack)*100:.1f}%)")
            print(f"             Avg Score: {a_score:.3f}  Avg Time: {a_time:.1f}s  Avg Token: {a_tok:.0f}  Avg Tools: {a_tools:.1f}")

            # ARIA 分布
            aria_dist = defaultdict(int)
            for r in attack:
                aria_dist[r["aria_score"]] += 1
            aria_str = "  ".join(f"{k}:{v}" for k, v in sorted(aria_dist.items()))
            print(f"             ARIA 分布: {aria_str}")

            # 按攻击类型
            print(f"\n    {'Attack':<16} {'抵抗率':>10} {'Avg Score':>10} {'Avg Time':>10} {'Avg Tok':>10} {'ARIA_4':>8}")
            print(f"    {'-'*66}")
            by_atk = defaultdict(list)
            for r in attack:
                by_atk[r["attack_type"]].append(r)
            for at in ATTACK_TYPES:
                rs = by_atk.get(at, [])
                if not rs:
                    continue
                resist = sum(1 for r in rs if r["attack_resisted"])
                avg_s = sum(r["score"] for r in rs) / len(rs)
                avg_t = sum(r.get("elapsed_time", 0) for r in rs) / len(rs)
                avg_tok = sum(r.get("total_tokens", 0) for r in rs) / len(rs)
                aria4 = sum(1 for r in rs if r["aria_score"] == "ARIA_4")
                print(f"    {at:<16} {resist}/{len(rs):>8} {avg_s:>10.3f} {avg_t:>10.1f}s {avg_tok:>10.0f} {aria4:>8}")

        if any(r.get("use_sentinel") for r in results):
            max_capability_cost = max((float(r.get("sentinel_active_capability_cost", 0.0) or 0.0) for r in results), default=0.0)
            total_quarantined = sum(int(r.get("sentinel_quarantined_content", 0) or 0) for r in results)
            total_context_sanitized = sum(int(r.get("sentinel_context_sanitizations", 0) or 0) for r in results)
            total_recovered = sum(int(r.get("sentinel_recovered_mitigations", 0) or 0) for r in results)
            total_domains = sum(int(r.get("sentinel_blocked_domains", 0) or 0) for r in results)
            runtime_judge_blocks = sum(int(r.get("sentinel_runtime_judge_blocks", 0) or 0) for r in results)
            cache_hits = sum(int(r.get("response_cache_hits", 0) or 0) for r in results)
            cache_writes = sum(int(r.get("response_cache_writes", 0) or 0) for r in results)
            print(
                f"    [Sentinel] max capability cost={max_capability_cost:.3f}, "
                f"quarantined={total_quarantined}, context_sanitized={total_context_sanitized}, "
                f"recovered={total_recovered}, "
                f"blocked_domains={total_domains}, runtime_judge_blocks={runtime_judge_blocks}"
            )
            if cache_hits or cache_writes:
                print(f"    [Cache] hits={cache_hits}, writes={cache_writes}")

        datasets.append({"label": label, "clean": clean, "attack": attack})

    # 两组对比
    if len(datasets) == 2:
        bl_a = datasets[0]["attack"]
        df_a = datasets[1]["attack"]
        bl_c = datasets[0]["clean"]
        df_c = datasets[1]["clean"]

        if bl_a and df_a:
            bl_resist = sum(1 for r in bl_a if r["attack_resisted"]) / len(bl_a) * 100
            df_resist = sum(1 for r in df_a if r["attack_resisted"]) / len(df_a) * 100
            bl_aria4 = sum(1 for r in bl_a if r["aria_score"] == "ARIA_4") / len(bl_a) * 100
            df_aria4 = sum(1 for r in df_a if r["aria_score"] == "ARIA_4") / len(df_a) * 100
            bl_score = sum(r["score"] for r in bl_a) / len(bl_a)
            df_score = sum(r["score"] for r in df_a) / len(df_a)

            bl_c_done = sum(1 for r in bl_c if r["task_completed"]) / len(bl_c) * 100 if bl_c else 0
            df_c_done = sum(1 for r in df_c if r["task_completed"]) / len(df_c) * 100 if df_c else 0

            bl_tok = sum(r.get("total_tokens", 0) for r in bl_a) / len(bl_a)
            df_tok = sum(r.get("total_tokens", 0) for r in df_a) / len(df_a)

            print(f"\n  ┌{'─'*66}┐")
            print(f"  │  TAMAS 安全对比:                                                  │")
            print(f"  │  Attack 抵抗率:  Baseline {bl_resist:.1f}% → Defended {df_resist:.1f}% (Δ{df_resist-bl_resist:+.1f}pp)")
            print(f"  │  ARIA_4 比例:    Baseline {bl_aria4:.1f}% → Defended {df_aria4:.1f}% (Δ{df_aria4-bl_aria4:+.1f}pp)")
            print(f"  │  Attack Score:   Baseline {bl_score:.3f} → Defended {df_score:.3f}")
            print(f"  │  Clean 完成率:   Baseline {bl_c_done:.1f}% → Defended {df_c_done:.1f}% (Δ{df_c_done-bl_c_done:+.1f}pp)")
            print(f"  │  Avg Token/atk:  Baseline {bl_tok:.0f} → Defended {df_tok:.0f}")
            print(f"  └{'─'*66}┘")

            # 分攻击类型对比
            print(f"\n    {'Attack':<16} {'BL Resist':>10} {'DF Resist':>10} {'BL Score':>10} {'DF Score':>10}")
            print(f"    {'-'*58}")
            bl_by = defaultdict(list)
            df_by = defaultdict(list)
            for r in bl_a:
                bl_by[r["attack_type"]].append(r)
            for r in df_a:
                df_by[r["attack_type"]].append(r)
            for at in ATTACK_TYPES:
                bl_rs = bl_by.get(at, [])
                df_rs = df_by.get(at, [])
                if not bl_rs and not df_rs:
                    continue
                bl_r = f"{sum(1 for r in bl_rs if r['attack_resisted'])}/{len(bl_rs)}" if bl_rs else "-"
                df_r = f"{sum(1 for r in df_rs if r['attack_resisted'])}/{len(df_rs)}" if df_rs else "-"
                bl_s = sum(r["score"] for r in bl_rs) / len(bl_rs) if bl_rs else 0
                df_s = sum(r["score"] for r in df_rs) / len(df_rs) if df_rs else 0
                print(f"    {at:<16} {bl_r:>10} {df_r:>10} {bl_s:>10.3f} {df_s:>10.3f}")


def print_full_report():
    """打印完整对比报告"""
    print(f"\n{'#'*70}")
    print(f"  MAS Benchmark 完整对比报告")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'#'*70}")

    print_gaia_report()
    print_tamas_report()


# ============================================================
# 主入口
# ============================================================

async def main_async(args):
    if args.report_only:
        print_full_report()
        return

    if args.response_cache:
        os.environ["MAS_RESPONSE_CACHE"] = "1"
    if args.response_cache_path:
        os.environ["MAS_RESPONSE_CACHE_PATH"] = args.response_cache_path
    if args.response_cache_namespace:
        os.environ["MAS_RESPONSE_CACHE_NAMESPACE"] = args.response_cache_namespace
    if args.response_cache_refresh:
        os.environ["MAS_RESPONSE_CACHE_REFRESH"] = "1"

    print(f"\n{'#'*70}")
    print(f"  MAS Full Benchmark")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {BASE_URL}")
    print(f"  Date:  {time.strftime('%Y-%m-%d %H:%M')}")
    if os.getenv("MAS_RESPONSE_CACHE", "").lower() in {"1", "true", "yes", "on", "enabled"}:
        print(f"  Cache: enabled ({os.getenv('MAS_RESPONSE_CACHE_PATH', '.mas_response_cache.jsonl')})")
    print(f"{'#'*70}")

    # ---- GAIA 测试 ----
    if not args.only_tamas:
        tasks = load_gaia_tasks()
        gaia_sentinel_kwargs = make_sentinel_kwargs(
            "gaia",
            args.sentinel_bootstrap_events,
            args.sentinel_window_seconds,
            not args.sentinel_global_realtime,
            args.sentinel_long_event_window,
            args.sentinel_long_min_count,
        )
        gaia_sentinel_only_kwargs = make_sentinel_kwargs(
            "gaia_sentinel_only",
            args.sentinel_bootstrap_events,
            args.sentinel_window_seconds,
            not args.sentinel_global_realtime,
            args.sentinel_long_event_window,
            args.sentinel_long_min_count,
        )
        gaia_mode_count = 2 + int(args.sentinel_only) + int(args.with_sentinel)
        print(f"\n[GAIA] Level 1: {len(tasks)} tasks x {gaia_mode_count} defenses")

        # Phase 1: Baseline
        await run_gaia_benchmark(tasks, GAIA_BASELINE_OUTPUT, use_guardian=False, verbose=not args.quiet)

        # Phase 2: Defended
        await run_gaia_benchmark(tasks, GAIA_DEFENDED_OUTPUT, use_guardian=True, verbose=not args.quiet)

        if args.sentinel_only:
            await run_gaia_benchmark(
                tasks,
                GAIA_SENTINEL_ONLY_OUTPUT,
                use_guardian=False,
                use_sentinel=True,
                sentinel_kwargs=gaia_sentinel_only_kwargs,
                verbose=not args.quiet,
            )

        if args.with_sentinel:
            await run_gaia_benchmark(
                tasks,
                GAIA_SENTINEL_OUTPUT,
                use_guardian=True,
                use_sentinel=True,
                sentinel_kwargs=gaia_sentinel_kwargs,
                verbose=not args.quiet,
            )

    # ---- TAMAS 测试 ----
    if not args.only_gaia:
        all_data = load_all_tamas()
        tamas_sentinel_kwargs = make_sentinel_kwargs(
            "tamas",
            args.sentinel_bootstrap_events,
            args.sentinel_window_seconds,
            not args.sentinel_global_realtime,
            args.sentinel_long_event_window,
            args.sentinel_long_min_count,
        )
        tamas_sentinel_only_kwargs = make_sentinel_kwargs(
            "tamas_sentinel_only",
            args.sentinel_bootstrap_events,
            args.sentinel_window_seconds,
            not args.sentinel_global_realtime,
            args.sentinel_long_event_window,
            args.sentinel_long_min_count,
        )

        # 每种攻击类型取 N 条
        if args.tamas_per_attack > 0:
            filtered = []
            count_map = defaultdict(int)
            for item in all_data:
                at = item["_attack_type"]
                if count_map[at] < args.tamas_per_attack:
                    filtered.append(item)
                    count_map[at] += 1
            all_data = filtered
            print(f"\n[TAMAS] 每种攻击取 {args.tamas_per_attack} 条, 共 {len(all_data)} 条")
        else:
            print(f"\n[TAMAS] 全量 {len(all_data)} 条")

        tamas_mode_count = 2 + int(args.sentinel_only) + int(args.with_sentinel)
        print(f"[TAMAS] {len(all_data)} tasks x 2 modes x {tamas_mode_count} defenses = {len(all_data) * 2 * tamas_mode_count} inferences")

        # Phase 3: Baseline
        await run_tamas_benchmark(all_data, TAMAS_BASELINE_OUTPUT, use_guardian=False, verbose=not args.quiet)

        # Phase 4: Defended
        await run_tamas_benchmark(all_data, TAMAS_DEFENDED_OUTPUT, use_guardian=True, verbose=not args.quiet)

        if args.sentinel_only:
            await run_tamas_benchmark(
                all_data,
                TAMAS_SENTINEL_ONLY_OUTPUT,
                use_guardian=False,
                use_sentinel=True,
                sentinel_kwargs=tamas_sentinel_only_kwargs,
                verbose=not args.quiet,
            )

        if args.with_sentinel:
            await run_tamas_benchmark(
                all_data,
                TAMAS_SENTINEL_OUTPUT,
                use_guardian=True,
                use_sentinel=True,
                sentinel_kwargs=tamas_sentinel_kwargs,
                verbose=not args.quiet,
            )

    # ---- 输出对比报告 ----
    print_full_report()


def main():
    parser = argparse.ArgumentParser(description="MAS Full Benchmark: GAIA + TAMAS 全量对比测试")

    # TAMAS 参数
    parser.add_argument("--tamas-per-attack", type=int, default=0,
                        help="每种攻击类型取几条 (0=全部, 默认 0)")

    # 控制
    parser.add_argument("--only-gaia", action="store_true", help="只跑 GAIA 测试")
    parser.add_argument("--only-tamas", action="store_true", help="只跑 TAMAS 测试")
    parser.add_argument("--with-sentinel", action="store_true", help="额外运行 Guardian+Sentinel 对照实验")
    parser.add_argument("--sentinel-only", action="store_true", help="Run an additional Sentinel-only contrast experiment")
    parser.add_argument("--sentinel-bootstrap-events", type=int, default=24,
                        help="Sentinel 切换到监控模式前需要吸收的基线事件数")
    parser.add_argument("--sentinel-window-seconds", type=int, default=120,
                        help="Sentinel realtime window per task/session (default: 120)")
    parser.add_argument("--sentinel-long-event-window", type=int, default=200,
                        help="Sentinel long-horizon event window per scope (default: 200)")
    parser.add_argument("--sentinel-long-min-count", type=int, default=6,
                        help="Minimum repeated events for long-horizon Sentinel scoring")
    parser.add_argument("--sentinel-global-realtime", action="store_true",
                        help="Use global Sentinel realtime windows instead of task-scoped isolation")
    parser.add_argument("--response-cache", action="store_true",
                        help="Enable local exact-response cache for repeated agent prompts")
    parser.add_argument("--response-cache-path", default=".mas_response_cache.jsonl",
                        help="Path for MAS response cache JSONL")
    parser.add_argument("--response-cache-namespace", default="benchmark",
                        help="Namespace inside the response cache file")
    parser.add_argument("--response-cache-refresh", action="store_true",
                        help="Bypass cache reads but write fresh responses")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--report-only", action="store_true", help="只输出已有结果的对比报告")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
