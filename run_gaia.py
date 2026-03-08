"""
GAIA Solver 入口脚本
========================================
基于 Microsoft Agent Framework + MetaGPT SOP 思想的多智能体系统
用于求解 GAIA benchmark Level 1 任务

已从 AutoGen v0.7 迁移到 agent-framework (Microsoft Agent Framework)

使用方法:
    # 测试单题
    python run_gaia.py --single 0

    # 求解前 N 题
    python run_gaia.py --num 5

    # 求解全部 Level 1
    python run_gaia.py --all

    # 评估已有结果
    python run_gaia.py --evaluate

环境变量:
    ARK_API_KEY       - 火山引擎 API Key (必需，或使用默认值)
    OPENAI_BASE_URL   - API 地址 (可选)
    GAIA_MODEL        - 文本模型接入点 ID (默认 DeepSeek-V3.2)
    GAIA_VISION_MODEL - 视觉模型接入点 ID (默认 Doubao-Seed-1.8)
"""
# 在所有其他导入之前先抑制无关警告
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys
import io
# Fix GBK encoding issues on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_solver.config import API_KEY, MODEL_NAME, GAIA_LEVEL1_PATH, BASE_URL, VISION_MODEL
from gaia_solver.solver import load_gaia_tasks, solve_single_task, solve_all_level1
from gaia_solver.evaluate import evaluate_results


def main():
    parser = argparse.ArgumentParser(description="GAIA Level 1 Solver")
    parser.add_argument("--single", type=int, default=None,
                        help="求解单个任务 (指定任务索引，从 0 开始)")
    parser.add_argument("--num", type=int, default=None,
                        help="求解前 N 个任务")
    parser.add_argument("--all", action="store_true",
                        help="求解所有 Level 1 任务")
    parser.add_argument("--evaluate", action="store_true",
                        help="评估已有结果")
    parser.add_argument("--output", type=str, default="gaia_results.jsonl",
                        help="结果输出文件路径")
    parser.add_argument("--max-round", type=int, default=15,
                        help="每个任务的最大对话轮数")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑，跳过已完成的题目")
    parser.add_argument("--quiet", action="store_true",
                        help="安静模式，减少输出")

    args = parser.parse_args()

    # 检查 API Key
    if not args.evaluate:
        if not API_KEY:
            print("ERROR: API Key 未设置!")
            print("请运行: $env:ARK_API_KEY='your-api-key'")
            print("或者:   $env:OPENAI_API_KEY='sk-your-key'")
            sys.exit(1)
        print(f"Text Model:   {MODEL_NAME}")
        print(f"Vision Model: {VISION_MODEL}")
        print(f"API Base: {BASE_URL}")
        print(f"Data:  {GAIA_LEVEL1_PATH}")
        print(f"Output: {args.output}")

    # 评估模式
    if args.evaluate:
        evaluate_results(args.output)
        return

    # 求解模式
    if args.single is not None:
        # 求解单个任务
        tasks = load_gaia_tasks()
        if args.single < 0 or args.single >= len(tasks):
            print(f"ERROR: 任务索引 {args.single} 超出范围 (0-{len(tasks)-1})")
            sys.exit(1)
        task = tasks[args.single]
        result = solve_single_task(task, max_turns=args.max_round, verbose=not args.quiet)
        
        # Save single task result to output file
        import json
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        status = "YES" if result['is_correct'] else "NO"
        print(f"\nCorrect: {status}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")

    elif args.num:
        # 求解前 N 题
        solve_all_level1(
            max_tasks=args.num,
            max_turns=args.max_round,
            verbose=not args.quiet,
            output_path=args.output,
            resume=args.resume,
        )

    elif args.all:
        # 求解全部
        solve_all_level1(
            max_tasks=None,
            max_turns=args.max_round,
            verbose=not args.quiet,
            output_path=args.output,
            resume=args.resume,
        )

    else:
        parser.print_help()
        print("\n示例:")
        print("  python run_gaia.py --single 0     # 测试第一题")
        print("  python run_gaia.py --num 5         # 前 5 题")
        print("  python run_gaia.py --all           # 全部 53 题")
        print("  python run_gaia.py --evaluate      # 评估结果")


if __name__ == "__main__":
    main()
