"""
评估脚本 - 分析 GAIA 求解结果
"""
import json
import os
from collections import Counter


def evaluate_results(results_path: str = "gaia_results.jsonl"):
    """
    评估求解结果，生成详细的统计报告。
    """
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    errors = sum(1 for r in results if r.get("predicted_answer", "").startswith("ERROR"))
    no_answer = sum(1 for r in results if r.get("predicted_answer") == "NO_ANSWER")

    print(f"{'='*60}")
    print(f"GAIA Level 1 Evaluation Report")
    print(f"{'='*60}")
    print(f"Total tasks:      {total}")
    print(f"Correct:           {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect:         {total - correct - errors - no_answer}")
    print(f"Errors:            {errors}")
    print(f"No Answer:         {no_answer}")
    print(f"{'='*60}")

    # 按附件类型统计
    print(f"\nBy Attachment Type:")
    with_file = [r for r in results if r.get("file_name")]
    without_file = [r for r in results if not r.get("file_name")]

    if without_file:
        wf_correct = sum(1 for r in without_file if r.get("is_correct"))
        print(f"  No attachment:   {wf_correct}/{len(without_file)} ({wf_correct/len(without_file)*100:.1f}%)")

    if with_file:
        f_correct = sum(1 for r in with_file if r.get("is_correct"))
        print(f"  With attachment: {f_correct}/{len(with_file)} ({f_correct/len(with_file)*100:.1f}%)")

        # 按文件扩展名
        ext_results = {}
        for r in with_file:
            ext = r.get("file_name", "").split(".")[-1].lower()
            if ext not in ext_results:
                ext_results[ext] = {"total": 0, "correct": 0}
            ext_results[ext]["total"] += 1
            if r.get("is_correct"):
                ext_results[ext]["correct"] += 1

        for ext, stats in sorted(ext_results.items()):
            print(f"    .{ext}: {stats['correct']}/{stats['total']}")

    # 打印错误的题目
    print(f"\n--- Incorrect Tasks ---")
    for r in results:
        if not r.get("is_correct"):
            print(f"  ID: {r['task_id']}")
            print(f"  Q:  {r['question'][:100]}...")
            print(f"  Predicted: {r['predicted_answer']}")
            print(f"  Expected:  {r['ground_truth']}")
            print()


if __name__ == "__main__":
    evaluate_results()
