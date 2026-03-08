"""
诊断脚本: 测试 GAIA Level 1 各类型题目各一道
  - 纯知识检索: #46 (NASA grant, 网页检索)
  - 逻辑/推理: #12 (土豆泥计算, 数学推理)
  - 文件处理(Excel): #52 (快餐店销售额)
  - 多媒体(音频): #31 (购物清单 MP3)
  - 代码执行: #35 (Python 代码输出)
"""
import asyncio
import json
import time
import sys

sys.path.insert(0, ".")

# 选取的题目索引(0-based)
SELECTED = [45, 11, 51, 30, 34]  
LABELS = ["知识检索", "逻辑推理", "文件处理(Excel)", "多媒体(音频)", "代码执行"]

async def main():
    from mas.factory import create_gaia_team
    from gaia_solver.solver import build_task_prompt, compare_answers

    # 加载数据
    with open("gaia_data/gaia_level1.jsonl", "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    results = []

    for idx, (task_idx, label) in enumerate(zip(SELECTED, LABELS)):
        task = all_tasks[task_idx]
        task_id = task["task_id"][:8]
        question = task["Question"][:120]
        ground_truth = task.get("Final answer", "")
        file_name = task.get("file_name", "")

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(SELECTED)}] 类型: {label}")
        print(f"  task_id: {task_id}")
        print(f"  Q: {question}...")
        print(f"  Expected: {ground_truth}")
        if file_name:
            print(f"  File: {file_name}")
        print(f"{'='*70}")

        # 每题新建一个team，避免状态干扰
        team = create_gaia_team(verbose=True)
        prompt = build_task_prompt(task)

        start = time.time()
        predicted = "NO_ANSWER"
        status = ""

        try:
            result = await asyncio.wait_for(
                team.run(task=prompt, task_id=task_id, expected_answer=ground_truth),
                timeout=300,
            )
            predicted = result.get("answer", "NO_ANSWER") or "NO_ANSWER"
            elapsed = time.time() - start
            is_correct = compare_answers(predicted, ground_truth)
            status = "CORRECT" if is_correct else "WRONG"
            
            m = result.get("metrics")
            tool_info = m.tool_calls if m else {}
            tokens = m.total_tokens if m else 0
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            predicted = "TIMEOUT"
            status = "TIMEOUT"
            tool_info = {}
            tokens = 0
        except Exception as e:
            elapsed = time.time() - start
            predicted = f"ERROR: {str(e)[:100]}"
            status = "ERROR"
            tool_info = {}
            tokens = 0

        r = {
            "label": label,
            "task_id": task_id,
            "status": status,
            "predicted": predicted[:80],
            "expected": ground_truth[:80],
            "elapsed": round(elapsed, 1),
            "tokens": tokens,
            "tools": tool_info,
        }
        results.append(r)
        
        print(f"\n  >>> [{status}] {elapsed:.1f}s | ans={predicted[:60]} | expect={ground_truth[:60]}")

    # 汇总
    print(f"\n\n{'='*70}")
    print(f"{'汇总':^70}")
    print(f"{'='*70}")
    fmt = "{:<18} {:<10} {:<8} {:>8} {:>8}  {}"
    print(fmt.format("类型", "状态", "task_id", "耗时", "tokens", "预测 / 期望"))
    print("-" * 90)
    correct = 0
    for r in results:
        print(fmt.format(
            r["label"],
            r["status"],
            r["task_id"],
            f"{r['elapsed']}s",
            str(r["tokens"]),
            f"{r['predicted'][:30]} / {r['expected'][:30]}"
        ))
        if r["status"] == "CORRECT":
            correct += 1
    print(f"\n正确率: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
