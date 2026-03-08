"""
TAMAS 测试入口脚本

用法:
    # 快速测试（DPI + education，每组1条，验证流程通畅）
    python -m tamas_adapter.run_test --quick

    # 测试特定攻击类型和场景
    python -m tamas_adapter.run_test --attacks DPI IPI --scenarios education finance

    # 每种组合测 2 条
    python -m tamas_adapter.run_test --max-per-group 2

    # 全量测试
    python -m tamas_adapter.run_test --full

    # 使用 LLM 评估（更准确但消耗 API）
    python -m tamas_adapter.run_test --quick --llm-eval

    # 断点续跑
    python -m tamas_adapter.run_test --resume

    # 只打印已有结果的报告
    python -m tamas_adapter.run_test --report-only
"""
import argparse
import asyncio
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamas_adapter.runner import run_comparison, print_report
from tamas_adapter.loader import ATTACK_TYPES, SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="TAMAS MAS 安全测试")
    parser.add_argument("--attacks", nargs="+", default=None,
                        help=f"攻击类型 (可选: {ATTACK_TYPES})")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help=f"场景 (可选: {SCENARIOS})")
    parser.add_argument("--max-per-group", type=int, default=None,
                        help="每种 attack+scenario 组合最多测几条")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Agent 最大对话轮数 (默认 10)")
    parser.add_argument("--output", type=str, default="tamas_results.jsonl",
                        help="输出文件路径")
    parser.add_argument("--llm-eval", action="store_true",
                        help="使用 LLM 评估（更准确但消耗 API）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试（DPI+education, 1条）")
    parser.add_argument("--full", action="store_true",
                        help="全量测试（所有攻击+场景）")
    parser.add_argument("--quiet", action="store_true",
                        help="减少输出")
    parser.add_argument("--report-only", action="store_true",
                        help="只打印已有结果的报告，不运行测试")
    parser.add_argument("--guardian", action="store_true",
                        help="启用 Guardian 防火墙（中心化策略执行点）")

    args = parser.parse_args()

    # 只打印报告
    if args.report_only:
        print_report(args.output)
        return

    # 快速模式
    if args.quick:
        args.attacks = args.attacks or ["DPI"]
        args.scenarios = args.scenarios or ["education"]
        args.max_per_group = args.max_per_group or 1

    # 全量模式
    if args.full:
        args.attacks = ATTACK_TYPES
        args.scenarios = SCENARIOS
        args.max_per_group = None

    print(f"TAMAS MAS 安全测试")
    print(f"  攻击类型: {args.attacks or 'ALL'}")
    print(f"  场景:     {args.scenarios or 'ALL'}")
    print(f"  每组上限: {args.max_per_group or 'ALL'}")
    print(f"  评估方式: {'LLM' if args.llm_eval else '规则'}")
    print(f"  Guardian: {'启用' if args.guardian else '关闭'}")
    print(f"  输出文件: {args.output}")
    print()

    asyncio.run(run_comparison(
        attack_types=args.attacks,
        scenarios=args.scenarios,
        max_tasks_per_group=args.max_per_group,
        max_turns=args.max_turns,
        use_llm_eval=args.llm_eval,
        verbose=not args.quiet,
        output_path=args.output,
        resume=args.resume,
        enable_guardian=args.guardian,
    ))


if __name__ == "__main__":
    main()
