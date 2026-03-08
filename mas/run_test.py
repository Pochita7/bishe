"""快速 API 测试 — 带指标追踪"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mas.factory import create_gaia_team
from mas.metrics import MetricsLogger

async def test():
    logger = MetricsLogger("mas_metrics.jsonl")

    team = create_gaia_team(verbose=True, max_rounds=2, metrics_logger=logger)

    result = await team.run(
        task=(
            "Search for who won the 2024 Nobel Prize in Physics, "
            "then use Python to count the total characters in all winners' names."
        ),
        task_id="test_001",
        expected_answer="31",
    )

    print()
    print("=== RESULT ===")
    print("Answer:", result["answer"])
    print("Rounds:", result["rounds"])
    print("Turns:", result["turns"])
    print("Elapsed:", result["elapsed"], "s")

    # 指标详情
    metrics = result["metrics"]
    print()
    print("=== METRICS ===")
    print(f"Tokens: in={metrics.input_tokens}, out={metrics.output_tokens}, total={metrics.total_tokens}")
    print(f"Tool calls: {metrics.total_tool_calls} -> {metrics.tool_calls}")
    print(f"Agent calls: {metrics.total_agent_calls} -> {metrics.agent_calls}")
    print(f"Handoffs: {metrics.handoff_count}")
    print(f"Correct: {metrics.is_correct}")

    print()
    print("=== MESSAGE LOG ===")
    for i, msg in enumerate(result["messages"]):
        content = msg["content"] or "(empty)"
        has_handoff = "HANDOFF" in content
        label = " <<HANDOFF>>" if has_handoff else ""
        print("  %d. [%s]%s: %s" % (i + 1, msg["source"], label, content[:250]))

    # 打印累计汇总
    print()
    logger.print_summary()

asyncio.run(test())
