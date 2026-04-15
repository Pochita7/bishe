"""
Token 统计修复验证脚本
=========================================
选取 5 个含较多工具调用的 GAIA 任务运行，对比：
  - 修复前记录的 token（从旧 benchmark JSONL）
  - estimate_tokens.py 的估算值
  - 修复后实际记录的 token

用法:
    python test_token_fix.py
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

from gaia_solver.config import get_text_client
from gaia_solver.solver import load_gaia_tasks, build_task_prompt, compare_answers
from mas.factory import create_gaia_team


# 选取的 5 个任务 ID（工具调用多，方便验证 token 累计修复效果）
TEST_TASK_IDS = [
    "cabe07ed-9eca-40ea-8",   # 19 tools, 7 agents, correct
    "840bfca7-4f7b-481a-8",   # 19 tools, 6 agents, correct
    "0383a3ee-47a7-41a4-b",   # 15 tools, 5 agents, correct
    "4b6bb5f7-f634-410e-8",   # 14 tools, 4 agents, correct
    "23dd907f-1261-4488-b",   # 17 tools, 7 agents, correct
]


def load_old_benchmark():
    """加载旧 benchmark 结果作为对比基线"""
    old_data = {}
    path = "benchmark_gaia_baseline.jsonl"
    if not os.path.exists(path):
        return old_data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            old_data[d["task_id"]] = d
    return old_data


def estimate_tokens_for_task(old_record):
    """用 estimate_tokens.py 的公式估算真实 token"""
    agent_calls = old_record.get("total_agent_calls", 0)
    tool_calls = old_record.get("total_tool_calls", 0)
    old_inp = old_record.get("input_tokens", 0)
    old_out = old_record.get("output_tokens", 0)

    if agent_calls > 0:
        tools_per_agent = tool_calls / agent_calls
    else:
        tools_per_agent = 0

    input_multiplier = 1 + tools_per_agent * 0.6
    est_inp = int(old_inp * input_multiplier)
    est_out = old_out + tool_calls * 80
    est_total = est_inp + est_out
    return est_inp, est_out, est_total


async def run_test():
    # 加载 GAIA 数据
    all_tasks = load_gaia_tasks()
    old_benchmark = load_old_benchmark()

    # 筛选测试任务（用前缀匹配）
    test_tasks = []
    for task in all_tasks:
        tid = task.get("task_id", "")
        for short_id in TEST_TASK_IDS:
            if tid.startswith(short_id):
                test_tasks.append(task)
                break

    if len(test_tasks) < 5:
        # 如果前缀匹配不够，尝试工具调用最多的5个任务
        print(f"[WARN] 前缀匹配只找到 {len(test_tasks)} 个任务，改用全部数据中前5个")
        matched_ids = {t["task_id"] for t in test_tasks}
        # 从 old_benchmark 中找 tool_calls 最多的，排除已匹配的
        candidates = [
            (tid, d) for tid, d in old_benchmark.items()
            if tid not in matched_ids
        ]
        candidates.sort(key=lambda x: x[1].get("total_tool_calls", 0), reverse=True)
        for tid, _ in candidates:
            if len(test_tasks) >= 5:
                break
            for task in all_tasks:
                if task.get("task_id", "") == tid:
                    test_tasks.append(task)
                    break

    print(f"\n{'='*80}")
    print(f"  Token 修复验证 — 运行 {len(test_tasks)} 个 GAIA 任务")
    print(f"{'='*80}")

    # 创建 team (Baseline，无 Guardian)
    client = get_text_client()
    team = create_gaia_team(client=client, max_rounds=3, verbose=True)

    results = []
    for i, task in enumerate(test_tasks):
        task_id = task.get("task_id", "")
        question = task.get("Question", "")
        ground_truth = task.get("Final answer", "")
        prompt = build_task_prompt(task)

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(test_tasks)}] {task_id[:24]}...")
        print(f"  Q: {question[:100]}...")
        print(f"{'='*70}")

        start = time.time()
        predicted = "NO_ANSWER"
        metrics = None

        try:
            result = await asyncio.wait_for(
                team.run(task=prompt, task_id=task_id, expected_answer=ground_truth),
                timeout=300,
            )
            predicted = result.get("answer", "NO_ANSWER") or "NO_ANSWER"
            metrics = result.get("metrics", None)
        except asyncio.TimeoutError:
            predicted = "TIMEOUT"
            print("  [TIMEOUT]")
        except Exception as e:
            predicted = f"ERROR: {str(e)[:200]}"
            print(f"  [ERROR] {e}")

        elapsed = time.time() - start
        is_correct = compare_answers(predicted, ground_truth)

        # 收集修复后的 token 数据
        new_inp = metrics.input_tokens if metrics else 0
        new_out = metrics.output_tokens if metrics else 0
        new_total = metrics.total_tokens if metrics else 0
        new_tools = metrics.total_tool_calls if metrics else 0
        new_agents = metrics.total_agent_calls if metrics else 0

        # 旧数据
        old_record = old_benchmark.get(task_id, {})
        old_inp = old_record.get("input_tokens", 0)
        old_out = old_record.get("output_tokens", 0)
        old_total = old_record.get("total_tokens", 0)
        old_tools = old_record.get("total_tool_calls", 0)

        # 估算值
        est_inp, est_out, est_total = estimate_tokens_for_task(old_record) if old_record else (0, 0, 0)

        record = {
            "task_id": task_id,
            "is_correct": is_correct,
            "elapsed": round(elapsed, 1),
            "new_tool_calls": new_tools,
            "new_agent_calls": new_agents,
            "old_input_tokens": old_inp,
            "old_output_tokens": old_out,
            "old_total_tokens": old_total,
            "old_tool_calls": old_tools,
            "est_input_tokens": est_inp,
            "est_output_tokens": est_out,
            "est_total_tokens": est_total,
            "new_input_tokens": new_inp,
            "new_output_tokens": new_out,
            "new_total_tokens": new_total,
        }
        results.append(record)

        status = "✓" if is_correct else "✗"
        print(f"\n  {status} Answer: {predicted[:80]}")
        print(f"    Expected: {ground_truth[:80]}")
        print(f"    Time: {elapsed:.1f}s | Tools: {new_tools} | Agents: {new_agents}")
        print(f"    Token 对比:")
        print(f"      旧记录:  input={old_inp:>7,}  output={old_out:>6,}  total={old_total:>8,}")
        print(f"      估算值:  input={est_inp:>7,}  output={est_out:>6,}  total={est_total:>8,}")
        print(f"      修复后:  input={new_inp:>7,}  output={new_out:>6,}  total={new_total:>8,}")
        if old_total > 0:
            ratio_new = new_total / old_total
            print(f"      修复/旧 = {ratio_new:.2f}x")
        if est_total > 0 and new_total > 0:
            accuracy = 1 - abs(new_total - est_total) / new_total
            print(f"      估算准确度 = {accuracy:.1%}")

        # 清理浏览器
        try:
            from gaia_solver.tools import _cleanup_browser
            _cleanup_browser()
        except Exception:
            pass

    # ========================================
    # 汇总报告
    # ========================================
    print(f"\n\n{'='*80}")
    print(f"  Token 修复验证汇总")
    print(f"{'='*80}")

    print(f"\n{'task_id':<26} {'旧total':>8} {'估算total':>8} {'修复total':>8} {'修/旧':>6} {'估算准确度':>8}")
    print("-" * 80)

    sum_old = sum_est = sum_new = 0
    for r in results:
        tid = r["task_id"][:24]
        ot = r["old_total_tokens"]
        et = r["est_total_tokens"]
        nt = r["new_total_tokens"]
        sum_old += ot
        sum_est += et
        sum_new += nt

        ratio = nt / ot if ot > 0 else 0
        acc = (1 - abs(nt - et) / nt) * 100 if nt > 0 else 0
        print(f"{tid:<26} {ot:>8,} {et:>8,} {nt:>8,} {ratio:>5.2f}x {acc:>7.1f}%")

    print("-" * 80)
    ratio_total = sum_new / sum_old if sum_old > 0 else 0
    acc_total = (1 - abs(sum_new - sum_est) / sum_new) * 100 if sum_new > 0 else 0
    print(f"{'合计':<26} {sum_old:>8,} {sum_est:>8,} {sum_new:>8,} {ratio_total:>5.2f}x {acc_total:>7.1f}%")

    print(f"\n结论:")
    print(f"  - Token 修复倍率: 修复后 / 旧记录 = {ratio_total:.2f}x")
    print(f"  - 估算公式准确度: {acc_total:.1f}%")

    # 保存结果
    output_path = "token_fix_test_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  详细结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_test())
