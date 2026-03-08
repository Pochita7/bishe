"""
TAMAS 攻击类型测试 — 每种攻击类型各取 1 个任务
6 种攻击: Byzantine, Colluding, Contradicting, DPI, IPI, Impersonation
每个任务跑 clean + attack 两种模式 = 12 次推理
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys, io, os
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import time
from collections import defaultdict

from gaia_solver.config import get_text_client
from tamas_adapter.loader import load_all_tamas, extract_task_info, ATTACK_TYPES
from tamas_adapter.prompt_builder import build_clean_prompt, build_attack_prompt
from tamas_adapter.evaluator import evaluate_output
from tamas_adapter.tools import get_tamas_function_tools, get_tool_call_log, clear_tool_call_log
from mas.factory import create_tamas_team


async def run_single(task_info, mode, client, verbose=True):
    """运行单个 TAMAS 任务"""
    task_id = task_info["id"]
    scenario = task_info["scenario"]
    attack_type = task_info["attack_type"]

    prompt = build_clean_prompt(task_info) if mode == "clean" else build_attack_prompt(task_info)
    q = task_info['clean_query'] if mode == 'clean' else task_info['attack_query']

    if verbose:
        print(f"    [{mode.upper():6s}] {task_id} | Q: {q[:80]}...")

    domain_tools = get_tamas_function_tools(
        scenario=scenario,
        include_malicious=True,
        include_benign=True,
    )
    clear_tool_call_log()

    team = create_tamas_team(
        scenario=scenario,
        domain_tools=domain_tools,
        guardian=None,
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

        msgs = result.get("messages", [])
        parts = []
        for m in msgs:
            src = m.get("source", "")
            content = m.get("content", "")
            if content:
                parts.append(f"[{src}]: {content}")
        full_output = "\n".join(parts)

        tool_log = get_tool_call_log()
        if tool_log:
            full_output += f"\n[ToolLog] TOOL_CALLS: {'; '.join(tool_log)}"

    except asyncio.TimeoutError:
        predicted = "TIMEOUT"
        full_output = "Task timed out after 300s"
    except Exception as e:
        predicted = f"ERROR: {str(e)[:200]}"
        full_output = str(e)

    elapsed = time.time() - start

    eval_result = evaluate_output(
        user_query=task_info["clean_query"],
        actual_output=full_output,
        attack_type=attack_type,
        mode=mode,
        injected_query=task_info["attack_query"] if mode == "attack" else "",
    )

    return {
        "id": task_id,
        "mode": mode,
        "attack_type": attack_type,
        "scenario": scenario,
        "task_completed": eval_result["task_completed"],
        "attack_resisted": eval_result["attack_resisted"],
        "aria_score": eval_result["aria_score"],
        "score": eval_result["score"],
        "malicious_tools_called": eval_result.get("malicious_tools_called", []),
        "elapsed_time": round(elapsed, 1),
        "predicted": predicted[:200],
    }


async def main():
    print("=" * 70)
    print("  TAMAS 攻击类型测试 — 每种攻击各 1 个")
    print(f"  攻击类型: {', '.join(ATTACK_TYPES)}")
    print("=" * 70)

    # 加载数据，每种攻击类型取第 1 个
    all_data = load_all_tamas()
    selected = {}
    for item in all_data:
        at = item["_attack_type"]
        if at not in selected:
            selected[at] = item

    print(f"\n  已选择 {len(selected)} 个任务:")
    for at, item in selected.items():
        print(f"    {at:16s} -> {item['_scenario']}")

    client = get_text_client()
    results = []

    for i, (at, item) in enumerate(selected.items()):
        task_info = extract_task_info(item)
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(selected)}] {at} / {task_info['scenario']}")
        print(f"  Clean Q: {task_info['clean_query'][:100]}...")
        print(f"{'='*70}")

        for mode in ["clean", "attack"]:
            r = await run_single(task_info, mode, client)
            results.append(r)

            tc = "OK" if r["task_completed"] else "FAIL"
            ar = "SAFE" if r["attack_resisted"] else "PWNED"
            mal = r["malicious_tools_called"]
            mal_str = f" MalTools: {','.join(mal)}" if mal else ""
            print(f"      → Task:{tc} Resist:{ar} ARIA:{r['aria_score']} "
                  f"Score:{r['score']:.2f} Time:{r['elapsed_time']}s{mal_str}")

    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")

    clean_results = [r for r in results if r["mode"] == "clean"]
    attack_results = [r for r in results if r["mode"] == "attack"]

    if clean_results:
        c_done = sum(1 for r in clean_results if r["task_completed"])
        c_avg = sum(r["score"] for r in clean_results) / len(clean_results)
        print(f"\n  [CLEAN] 完成: {c_done}/{len(clean_results)} "
              f"({c_done/len(clean_results)*100:.0f}%)  Avg Score: {c_avg:.3f}")

    if attack_results:
        a_done = sum(1 for r in attack_results if r["task_completed"])
        a_resist = sum(1 for r in attack_results if r["attack_resisted"])
        a_avg = sum(r["score"] for r in attack_results) / len(attack_results)
        print(f"  [ATTACK] 完成: {a_done}/{len(attack_results)} "
              f"({a_done/len(attack_results)*100:.0f}%)  "
              f"抵抗: {a_resist}/{len(attack_results)} "
              f"({a_resist/len(attack_results)*100:.0f}%)  "
              f"Avg Score: {a_avg:.3f}")

        aria_dist = defaultdict(int)
        for r in attack_results:
            aria_dist[r["aria_score"]] += 1
        aria_str = "  ".join(f"{k}:{v}" for k, v in sorted(aria_dist.items()))
        print(f"  ARIA 分布: {aria_str}")

    # 逐攻击类型
    fmt = "  {:<16} {:>8} {:>8} {:>10} {:>10} {:>8} {:>10}"
    print(f"\n{fmt.format('Attack', 'C_Done', 'A_Done', 'A_Resist', 'A_Score', 'ARIA', 'Time')}")
    print(f"  {'-'*76}")

    for at in ATTACK_TYPES:
        c = [r for r in clean_results if r["attack_type"] == at]
        a = [r for r in attack_results if r["attack_type"] == at]
        if not c and not a:
            continue
        c_done = sum(1 for r in c if r["task_completed"]) if c else 0
        a_done = sum(1 for r in a if r["task_completed"]) if a else 0
        a_resist = sum(1 for r in a if r["attack_resisted"]) if a else 0
        a_score = sum(r["score"] for r in a) / len(a) if a else 0
        aria = a[0]["aria_score"] if a else "-"
        a_time = a[0]["elapsed_time"] if a else 0
        mal = a[0]["malicious_tools_called"] if a else []
        mal_str = f" [{','.join(mal)}]" if mal else ""
        print(fmt.format(at, f"{c_done}/1", f"{a_done}/1", f"{a_resist}/1",
                         f"{a_score:.3f}", aria, f"{a_time}s") + mal_str)


if __name__ == "__main__":
    asyncio.run(main())
