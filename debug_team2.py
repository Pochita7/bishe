"""
调试脚本 2：用 GAIA 实际任务测试 team
"""
import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_solver.config import get_model_client, GAIA_LEVEL1_PATH
from gaia_solver.solver import load_gaia_tasks, build_task_prompt


async def test_gaia_with_roundrobin():
    """用 RoundRobinGroupChat 替代 SelectorGroupChat 测试"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

    client = get_model_client()

    # 简单的两阶段 Agent
    solver = AssistantAgent(
        name="Solver",
        model_client=client,
        system_message="""You solve GAIA benchmark tasks step by step.
For calculations, show your work.
For knowledge questions, explain your reasoning.
After solving, state your answer as: FINAL_ANSWER: <exact answer>
Keep answers concise - just the number, word, or short phrase asked for.""",
    )

    checker = AssistantAgent(
        name="Checker",
        model_client=client,
        system_message="""You verify solutions to GAIA tasks.
Check the Solver's work for errors.
If correct, repeat the answer as: FINAL_ANSWER: <exact answer>
If wrong, explain the error so Solver can fix it.""",
    )

    termination = TextMentionTermination("FINAL_ANSWER") | MaxMessageTermination(10)

    team = RoundRobinGroupChat(
        participants=[solver, checker],
        termination_condition=termination,
    )

    # 加载第一个任务
    tasks = load_gaia_tasks()
    task = tasks[0]
    prompt = build_task_prompt(task)
    
    print(f"Task: {task['Question'][:100]}...")
    print(f"Ground truth: {task.get('Final answer', '?')}")
    print(f"Prompt length: {len(prompt)} chars")
    print()

    try:
        result = await team.run(task=prompt)
        print(f"\nMessages count: {len(result.messages)}")
        for msg in result.messages:
            src = getattr(msg, 'source', 'unknown')
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"\n[{src}]: {content[:500]}")

        # 检查 stop_reason
        if hasattr(result, 'stop_reason'):
            print(f"\nStop reason: {result.stop_reason}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


async def test_gaia_with_selector_debug():
    """用 SelectorGroupChat 测试但打印选择过程"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_core.tools import FunctionTool

    client = get_model_client()

    # 简单工具
    def calculate(expression: str) -> str:
        """Evaluate a math expression and return the result."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message="You analyze tasks and create plans. Be brief.",
        description="Analyzes tasks and creates plans.",
    )

    executor = AssistantAgent(
        name="Executor",
        model_client=client,
        system_message="You execute plans using tools. Say FINAL_ANSWER: <answer> when done.",
        tools=[FunctionTool(calculate, description="Evaluate a math expression.")],
        description="Executes plans using tools.",
    )

    termination = TextMentionTermination("FINAL_ANSWER") | MaxMessageTermination(8)

    # 极简 selector prompt
    selector_prompt = """Choose the next speaker from {participants}.
- Planner goes first to analyze the task
- Executor goes next to solve and give FINAL_ANSWER
Return ONLY the name.
{roles}"""

    team = SelectorGroupChat(
        participants=[planner, executor],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
    )

    tasks = load_gaia_tasks()
    task = tasks[0]
    prompt = build_task_prompt(task)

    print(f"\n{'='*50}")
    print("TEST: SelectorGroupChat with GAIA task")
    print(f"{'='*50}")
    
    try:
        result = await team.run(task=prompt)
        print(f"\nMessages count: {len(result.messages)}")
        for msg in result.messages:
            src = getattr(msg, 'source', 'unknown')
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"\n[{src}]: {content[:500]}")
        if hasattr(result, 'stop_reason'):
            print(f"\nStop reason: {result.stop_reason}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("Test 1: RoundRobinGroupChat with GAIA task")
    await test_gaia_with_roundrobin()
    
    print("\n\nTest 2: SelectorGroupChat with GAIA task")
    await test_gaia_with_selector_debug()


if __name__ == "__main__":
    asyncio.run(main())
