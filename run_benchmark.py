"""
统一 Benchmark 脚本 — GAIA 性能 + TAMAS 安全 对比测试

运行 4 组实验:
  1. GAIA  Baseline  (无 Guardian) — 衡量基线性能
  2. GAIA  Defended  (有 Guardian) — 衡量防御对性能的影响
  3. TAMAS Baseline  (无 Guardian) — 衡量基线安全能力
  4. TAMAS Defended  (有 Guardian) — 衡量防御对攻击抵抗的提升

用法:
    # 小规模试跑 (GAIA 前5题 + TAMAS 每组1条)
    python run_benchmark.py

    # 自定义规模
    python run_benchmark.py --gaia-num 10 --tamas-per-group 2

    # 全量
    python run_benchmark.py --gaia-all --tamas-full

    # 只跑 GAIA
    python run_benchmark.py --only-gaia

    # 只跑 TAMAS
    python run_benchmark.py --only-tamas

    # 只输出已有结果的对比报告
    python run_benchmark.py --report-only
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys, io, os
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from collections import defaultdict

from gaia_solver.config import get_text_client, API_KEY, MODEL_NAME, BASE_URL
from gaia_solver.solver import load_gaia_tasks, build_task_prompt, compare_answers, extract_final_answer
from gaia_solver import tools as gaia_tools

from mas.factory import create_gaia_team, create_tamas_team
from mas.metrics import MetricsLogger, MetricsCollector

from tamas_adapter.loader import load_all_tamas, extract_task_info, ATTACK_TYPES, SCENARIOS
from tamas_adapter.prompt_builder import build_clean_prompt, build_attack_prompt
from tamas_adapter.evaluator import evaluate_output
from tamas_adapter.tools import (
    get_tamas_function_tools, get_tool_call_log, clear_tool_call_log,
)
from tamas_adapter.guardian import Guardian, Action


# ============================================================
# 配置
# ============================================================

GAIA_BASELINE_OUTPUT = "benchmark_gaia_baseline.jsonl"
GAIA_DEFENDED_OUTPUT = "benchmark_gaia_defended.jsonl"
TAMAS_BASELINE_OUTPUT = "benchmark_tamas_baseline.jsonl"
TAMAS_DEFENDED_OUTPUT = "benchmark_tamas_defended.jsonl"


# ============================================================
# GAIA 测试
# ============================================================

async def run_gaia_single(task: dict, team, verbose: bool = True) -> dict:
    """用 MASTeam 求解单个 GAIA 任务"""
    task_id = task.get("task_id", "")
    ground_truth = task.get("Final answer", "")
    question = task.get("Question", "")
    file_name = task.get("file_name", "")

    prompt = build_task_prompt(task)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[GAIA] {task_id[:16]}...")
        print(f"  Q: {question[:100]}...")
        if file_name:
            print(f"  File: {file_name}")
        print(f"{'='*60}")

    start = time.time()
    predicted = "NO_ANSWER"

    try:
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=task_id, expected_answer=ground_truth),
            timeout=300,
        )
        predicted = result.get("answer", "NO_ANSWER") or "NO_ANSWER"
    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        if verbose:
            print("  [TIMEOUT]")
    except Exception as e:
        predicted = f"ERROR: {str(e)[:200]}"
        if verbose:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - start
    is_correct = compare_answers(predicted, ground_truth)

    result_dict = {
        "task_id": task_id,
        "question": question[:200],
        "file_name": file_name,
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "elapsed": round(elapsed, 1),
    }

    if verbose:
        status = "✓" if is_correct else "✗"
        print(f"  {status} Predicted: {predicted[:80]}")
        print(f"    Expected: {ground_truth[:80]}")
        print(f"    Time: {elapsed:.1f}s")

    # 清理浏览器
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


async def run_gaia_benchmark(
    tasks: List[dict],
    output_path: str,
    use_guardian: bool = False,
    verbose: bool = True,
):
    """运行 GAIA benchmark (baseline 或 defended)"""
    mode_name = "Defended" if use_guardian else "Baseline"
    print(f"\n{'#'*70}")
    print(f"  GAIA {mode_name} — {len(tasks)} tasks")
    print(f"  Output: {output_path}")
    print(f"{'#'*70}")

    client = get_text_client()

    # Guardian (仅 defended 模式)
    guardian = None
    if use_guardian:
        guardian = Guardian(scenario="gaia", strict_mode=False)
        # 创建一个有 Guardian 保护的 GAIA 团队
        # 复用 create_tamas_team 的 Guardian 包裹逻辑，不传 domain_tools
        team = create_tamas_team(
            scenario="gaia",
            domain_tools=None,
            guardian=guardian,
            client=client,
            max_rounds=3,
            verbose=False,
            worker_timeout=120,
        )
    else:
        team = create_gaia_team(client=client, max_rounds=3, verbose=False)

    # 清空输出
    if os.path.exists(output_path):
        os.remove(output_path)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n--- [{i+1}/{len(tasks)}] ---")
        r = await run_gaia_single(task, team, verbose=verbose)
        results.append(r)

        # 逐条保存
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


# ============================================================
# TAMAS 测试
# ============================================================

async def run_tamas_single(
    task_info: dict,
    mode: str,
    client,
    use_guardian: bool = False,
    verbose: bool = True,
) -> dict:
    """用 MASTeam 架构求解单个 TAMAS 任务"""
    task_id = task_info["id"]
    scenario = task_info["scenario"]
    attack_type = task_info["attack_type"]

    # 构建 prompt
    if mode == "clean":
        prompt = build_clean_prompt(task_info)
    else:
        prompt = build_attack_prompt(task_info)

    if verbose:
        q = task_info['clean_query'] if mode == 'clean' else task_info['attack_query']
        print(f"  [{mode.upper()}] {task_id} | {attack_type}/{scenario}")
        print(f"  Q: {q[:100]}...")

    # 获取 TAMAS 领域工具 (始终包含恶意工具 — 真实测试)
    domain_tools = get_tamas_function_tools(
        scenario=scenario,
        include_malicious=True,
        include_benign=True,
    )

    # 清空工具调用日志
    clear_tool_call_log()

    # 创建团队
    guardian = None
    if use_guardian:
        guardian = Guardian(scenario=scenario, strict_mode=False)
        # Gate 0: 预扫描工具描述
        tool_descs = [{"name": t.name, "description": t.description} for t in domain_tools]
        guardian.scan_tool_descriptions(tool_descs, verbose=False)

    team = create_tamas_team(
        scenario=scenario,
        domain_tools=domain_tools,
        guardian=guardian,
        client=client,
        max_rounds=2,
        verbose=False,
        worker_timeout=90,
        max_tool_calls_per_worker=8,
    )

    start = time.time()
    predicted = "NO_ANSWER"
    full_output = ""

    try:
        result = await asyncio.wait_for(
            team.run(task=prompt, task_id=task_id),
            timeout=300,
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

        # 附加工具调用日志
        tool_log = get_tool_call_log()
        if tool_log:
            tool_log_str = "; ".join(tool_log)
            full_output += f"\n[ToolLog] TOOL_CALLS: {tool_log_str}"

        # 附加 Guardian 审计日志
        if guardian:
            full_output += f"\n[Guardian Audit Log] {guardian.get_log().summary()}"

    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        full_output = "Task timed out after 300s"
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

    result_dict = {
        "id": task_id,
        "mode": mode,
        "attack_type": attack_type,
        "scenario": scenario,
        "clean_query": task_info["clean_query"][:200],
        "predicted_answer": predicted[:500],
        "full_output": full_output[:2000],
        "task_completed": eval_result["task_completed"],
        "attack_resisted": eval_result["attack_resisted"],
        "aria_score": eval_result["aria_score"],
        "score": eval_result["score"],
        "reasoning": eval_result["reasoning"],
        "tools_called": eval_result.get("tools_called", []),
        "malicious_tools_called": eval_result.get("malicious_tools_called", []),
        "elapsed_time": round(elapsed, 1),
    }

    if verbose:
        tc = "OK" if eval_result["task_completed"] else "FAIL"
        ar = "SAFE" if eval_result["attack_resisted"] else "PWNED"
        mal = eval_result.get("malicious_tools_called", [])
        mal_str = f" | MalTools: {','.join(mal)}" if mal else ""
        print(f"    → Task:{tc} Resist:{ar} ARIA:{eval_result['aria_score']} "
              f"Score:{eval_result['score']:.2f} Time:{elapsed:.1f}s{mal_str}")

    # 清理浏览器
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


async def run_tamas_benchmark(
    all_data: List[dict],
    output_path: str,
    use_guardian: bool = False,
    verbose: bool = True,
):
    """运行 TAMAS benchmark (baseline 或 defended)"""
    mode_name = "Defended" if use_guardian else "Baseline"
    total = len(all_data)
    print(f"\n{'#'*70}")
    print(f"  TAMAS {mode_name} — {total} tasks × 2 modes = {total * 2} inferences")
    print(f"  Guardian: {'ON' if use_guardian else 'OFF'}")
    print(f"  Output: {output_path}")
    print(f"{'#'*70}")

    client = get_text_client()

    if os.path.exists(output_path):
        os.remove(output_path)

    for i, item in enumerate(all_data):
        task_info = extract_task_info(item)
        print(f"\n--- [{i+1}/{total}] {task_info['attack_type']}/{task_info['scenario']} ---")

        for mode in ["clean", "attack"]:
            r = await run_tamas_single(
                task_info, mode=mode, client=client,
                use_guardian=use_guardian, verbose=verbose,
            )
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# 对比报告
# ============================================================

def print_gaia_comparison():
    """打印 GAIA baseline vs defended 对比"""
    print(f"\n{'='*70}")
    print(f"  GAIA Benchmark 对比报告")
    print(f"{'='*70}")

    for label, path in [("Baseline", GAIA_BASELINE_OUTPUT), ("Defended", GAIA_DEFENDED_OUTPUT)]:
        if not os.path.exists(path):
            print(f"\n  [{label}] 文件不存在: {path}")
            continue

        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        if not results:
            print(f"\n  [{label}] 无数据")
            continue

        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct"))
        errors = sum(1 for r in results if str(r.get("predicted_answer", "")).startswith("ERROR"))
        timeouts = sum(1 for r in results if r.get("predicted_answer") == "TIMEOUT")
        avg_time = sum(r.get("elapsed", 0) for r in results) / total

        print(f"\n  [{label}] ({path})")
        print(f"    Total:     {total}")
        print(f"    Correct:   {correct}/{total} = {correct/total*100:.1f}%")
        print(f"    Errors:    {errors}")
        print(f"    Timeouts:  {timeouts}")
        print(f"    Avg Time:  {avg_time:.1f}s")

    # 如果两个都有数据，打印对比
    if os.path.exists(GAIA_BASELINE_OUTPUT) and os.path.exists(GAIA_DEFENDED_OUTPUT):
        bl = [json.loads(l) for l in open(GAIA_BASELINE_OUTPUT, encoding="utf-8") if l.strip()]
        df = [json.loads(l) for l in open(GAIA_DEFENDED_OUTPUT, encoding="utf-8") if l.strip()]

        if bl and df:
            bl_acc = sum(1 for r in bl if r.get("is_correct")) / len(bl) * 100
            df_acc = sum(1 for r in df if r.get("is_correct")) / len(df) * 100
            bl_time = sum(r.get("elapsed", 0) for r in bl) / len(bl)
            df_time = sum(r.get("elapsed", 0) for r in df) / len(df)

            print(f"\n  ┌{'─'*50}┐")
            print(f"  │  GAIA 对比:                                    │")
            print(f"  │  Accuracy:  Baseline {bl_acc:.1f}% → Defended {df_acc:.1f}%  (Δ{df_acc-bl_acc:+.1f}pp) │")
            print(f"  │  Avg Time:  {bl_time:.1f}s → {df_time:.1f}s  (Δ{df_time-bl_time:+.1f}s)│")
            print(f"  └{'─'*50}┘")

    print()


def print_tamas_comparison():
    """打印 TAMAS baseline vs defended 对比"""
    print(f"\n{'='*70}")
    print(f"  TAMAS Benchmark 对比报告")
    print(f"{'='*70}")

    for label, path in [("Baseline", TAMAS_BASELINE_OUTPUT), ("Defended", TAMAS_DEFENDED_OUTPUT)]:
        if not os.path.exists(path):
            print(f"\n  [{label}] 文件不存在: {path}")
            continue

        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        if not results:
            print(f"\n  [{label}] 无数据")
            continue

        clean = [r for r in results if r["mode"] == "clean"]
        attack = [r for r in results if r["mode"] == "attack"]

        print(f"\n  [{label}] ({path})")

        if clean:
            c_done = sum(1 for r in clean if r["task_completed"])
            c_score = sum(r["score"] for r in clean) / len(clean)
            print(f"    [CLEAN]  完成: {c_done}/{len(clean)} ({c_done/len(clean)*100:.1f}%)  Avg Score: {c_score:.3f}")

        if attack:
            a_done = sum(1 for r in attack if r["task_completed"])
            a_resist = sum(1 for r in attack if r["attack_resisted"])
            a_score = sum(r["score"] for r in attack) / len(attack)
            print(f"    [ATTACK] 完成: {a_done}/{len(attack)} ({a_done/len(attack)*100:.1f}%)  "
                  f"抵抗: {a_resist}/{len(attack)} ({a_resist/len(attack)*100:.1f}%)  "
                  f"Avg Score: {a_score:.3f}")

            # ARIA 分布
            aria_dist = defaultdict(int)
            for r in attack:
                aria_dist[r["aria_score"]] += 1
            aria_str = "  ".join(f"{k}:{v}" for k, v in sorted(aria_dist.items()))
            print(f"    ARIA 分布: {aria_str}")

            # 按攻击类型
            print(f"\n    {'Attack':<16} {'抵抗率':>10} {'Avg Score':>10} {'ARIA_4':>8}")
            print(f"    {'-'*46}")
            by_atk = defaultdict(list)
            for r in attack:
                by_atk[r["attack_type"]].append(r)
            for at, rs in sorted(by_atk.items()):
                resist = sum(1 for r in rs if r["attack_resisted"])
                avg_s = sum(r["score"] for r in rs) / len(rs)
                aria4 = sum(1 for r in rs if r["aria_score"] == "ARIA_4")
                print(f"    {at:<16} {resist}/{len(rs):>8} {avg_s:>10.3f} {aria4:>8}")

    # 两者对比
    if os.path.exists(TAMAS_BASELINE_OUTPUT) and os.path.exists(TAMAS_DEFENDED_OUTPUT):
        bl = [json.loads(l) for l in open(TAMAS_BASELINE_OUTPUT, encoding="utf-8") if l.strip()]
        df = [json.loads(l) for l in open(TAMAS_DEFENDED_OUTPUT, encoding="utf-8") if l.strip()]

        bl_attack = [r for r in bl if r["mode"] == "attack"]
        df_attack = [r for r in df if r["mode"] == "attack"]
        bl_clean = [r for r in bl if r["mode"] == "clean"]
        df_clean = [r for r in df if r["mode"] == "clean"]

        if bl_attack and df_attack:
            bl_resist = sum(1 for r in bl_attack if r["attack_resisted"]) / len(bl_attack) * 100
            df_resist = sum(1 for r in df_attack if r["attack_resisted"]) / len(df_attack) * 100
            bl_aria4 = sum(1 for r in bl_attack if r["aria_score"] == "ARIA_4") / len(bl_attack) * 100
            df_aria4 = sum(1 for r in df_attack if r["aria_score"] == "ARIA_4") / len(df_attack) * 100

            bl_c_done = sum(1 for r in bl_clean if r["task_completed"]) / len(bl_clean) * 100 if bl_clean else 0
            df_c_done = sum(1 for r in df_clean if r["task_completed"]) / len(df_clean) * 100 if df_clean else 0

            print(f"\n  ┌{'─'*58}┐")
            print(f"  │  TAMAS 对比:                                              │")
            print(f"  │  Attack 抵抗率:  Baseline {bl_resist:.1f}% → Defended {df_resist:.1f}% (Δ{df_resist-bl_resist:+.1f}pp) │")
            print(f"  │  ARIA_4 比例:    Baseline {bl_aria4:.1f}% → Defended {df_aria4:.1f}% (Δ{df_aria4-bl_aria4:+.1f}pp) │")
            print(f"  │  Clean 完成率:   Baseline {bl_c_done:.1f}% → Defended {df_c_done:.1f}% (Δ{df_c_done-bl_c_done:+.1f}pp) │")
            print(f"  └{'─'*58}┘")

    print()


def print_full_report():
    """打印完整对比报告"""
    print(f"\n{'#'*70}")
    print(f"  MAS Benchmark 完整对比报告")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'#'*70}")

    print_gaia_comparison()
    print_tamas_comparison()


# ============================================================
# 主入口
# ============================================================

async def main_async(args):
    """异步主函数"""
    if args.report_only:
        print_full_report()
        return

    print(f"\n{'#'*70}")
    print(f"  MAS Benchmark")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {BASE_URL}")
    print(f"{'#'*70}")

    # ---- GAIA 测试 ----
    if not args.only_tamas:
        tasks = load_gaia_tasks()
        if args.gaia_all:
            gaia_tasks = tasks
        else:
            gaia_tasks = tasks[:args.gaia_num]

        print(f"\n[GAIA] 将测试 {len(gaia_tasks)} 题 × 2 轮 (Baseline + Defended)")

        # Baseline
        await run_gaia_benchmark(
            gaia_tasks, GAIA_BASELINE_OUTPUT,
            use_guardian=False, verbose=not args.quiet,
        )

        # Defended
        await run_gaia_benchmark(
            gaia_tasks, GAIA_DEFENDED_OUTPUT,
            use_guardian=True, verbose=not args.quiet,
        )

    # ---- TAMAS 测试 ----
    if not args.only_gaia:
        all_data = load_all_tamas()

        # 按组限制
        if not args.tamas_full:
            per_group = args.tamas_per_group
            filtered = []
            count_map = {}
            for item in all_data:
                key = f"{item['_attack_type']}_{item['_scenario']}"
                count_map[key] = count_map.get(key, 0) + 1
                if count_map[key] <= per_group:
                    filtered.append(item)
            all_data = filtered

        # 进一步截断
        if args.tamas_max > 0 and len(all_data) > args.tamas_max:
            all_data = all_data[:args.tamas_max]

        print(f"\n[TAMAS] 将测试 {len(all_data)} 条 × 2 模式 × 2 轮 = {len(all_data)*4} 次推理")

        # Baseline
        await run_tamas_benchmark(
            all_data, TAMAS_BASELINE_OUTPUT,
            use_guardian=False, verbose=not args.quiet,
        )

        # Defended
        await run_tamas_benchmark(
            all_data, TAMAS_DEFENDED_OUTPUT,
            use_guardian=True, verbose=not args.quiet,
        )

    # ---- 输出对比报告 ----
    print_full_report()


def main():
    parser = argparse.ArgumentParser(description="MAS Benchmark: GAIA + TAMAS 对比测试")

    # GAIA 参数
    parser.add_argument("--gaia-num", type=int, default=5,
                        help="GAIA 测试题数 (默认 5)")
    parser.add_argument("--gaia-all", action="store_true",
                        help="GAIA 全量 53 题")

    # TAMAS 参数
    parser.add_argument("--tamas-per-group", type=int, default=1,
                        help="每种攻击+场景组合测几条 (默认 1)")
    parser.add_argument("--tamas-full", action="store_true",
                        help="TAMAS 全量 300 条")
    parser.add_argument("--tamas-max", type=int, default=0,
                        help="TAMAS 最大任务数 (0=不限制, 默认 0)")

    # 控制
    parser.add_argument("--only-gaia", action="store_true",
                        help="只跑 GAIA 测试")
    parser.add_argument("--only-tamas", action="store_true",
                        help="只跑 TAMAS 测试")
    parser.add_argument("--quiet", action="store_true",
                        help="安静模式")
    parser.add_argument("--report-only", action="store_true",
                        help="只输出已有结果的对比报告")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
