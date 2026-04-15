"""
MAS 工作流程演示 — 用一个简单任务展示完整的多智能体协作过程

任务: "144的平方根是多少？"
预期流程: Planner 规划 → CodeExecutor 计算 → Planner 审查 → FINAL_ANSWER
"""
import asyncio
import sys
import time

# ============================================================
# 演示配置
# ============================================================

async def demo():
    print("=" * 70)
    print("  MAS 多智能体系统 — 工作流程演示")
    print("=" * 70)

    # ------ Step 0: 创建团队 ------
    print("\n[Step 0] 创建 MAS 团队...")
    from mas.factory import create_gaia_team
    team = create_gaia_team(max_rounds=3, verbose=True)

    # ------ 运行任务 ------
    task = "What is the square root of 144?"
    print(f"\n{'=' * 70}")
    print(f"  任务: {task}")
    print(f"  预期答案: 12")
    print(f"{'=' * 70}")

    result = await team.run(
        task=task,
        task_id="demo_001",
        expected_answer="12",
    )

    # ------ 结果展示 ------
    print(f"\n{'=' * 70}")
    print(f"  工作流程执行完毕")
    print(f"{'=' * 70}")
    print(f"  最终答案: {result['answer']}")
    print(f"  规划轮数: {result['rounds']}")
    print(f"  总对话轮: {result['turns']}")
    print(f"  总耗时:   {result['elapsed']}s")

    # 详细消息流
    print(f"\n{'=' * 70}")
    print(f"  详细消息流 (谁说了什么)")
    print(f"{'=' * 70}")
    for i, msg in enumerate(result['messages'], 1):
        source = msg['source']
        content = msg['content']
        # 截断过长内容
        if len(content) > 500:
            content = content[:500] + "...(truncated)"
        print(f"\n  [{i}] {source}:")
        for line in content.split('\n'):
            print(f"      {line}")

    # Metrics 详情
    metrics = result.get('metrics')
    if metrics:
        print(f"\n{'=' * 70}")
        print(f"  指标详情 (Metrics)")
        print(f"{'=' * 70}")
        print(f"  Input tokens:  {metrics.input_tokens}")
        print(f"  Output tokens: {metrics.output_tokens}")
        print(f"  Total tokens:  {metrics.total_tokens}")
        print(f"  Tool calls:    {metrics.tool_calls}")
        print(f"  Agent calls:   {metrics.agent_calls}")
        print(f"  Handoffs:      {metrics.handoff_count}")

    return result


if __name__ == "__main__":
    result = asyncio.run(demo())
    print(f"\n✓ 演示完成 — 答案: {result['answer']}")
