"""
调试脚本：测试 SelectorGroupChat 是否正常工作
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_solver.config import get_model_client, API_KEY, MODEL_NAME, BASE_URL


async def test_simple_chat():
    """测试 1: 简单的单 Agent 对话"""
    from autogen_agentchat.agents import AssistantAgent

    print("="*50)
    print("TEST 1: Single Agent Chat")
    print("="*50)

    client = get_model_client()
    agent = AssistantAgent(
        name="TestAgent",
        model_client=client,
        system_message="You are a helpful assistant. Always answer concisely.",
    )

    from autogen_core import CancellationToken
    response = await agent.on_messages(
        [
            # 使用 ChatMessage
        ],
        CancellationToken(),
    )
    # 直接用 run
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination

    team = RoundRobinGroupChat(
        participants=[agent],
        termination_condition=MaxMessageTermination(3),
    )
    result = await team.run(task="What is 2+2? Answer with just the number.")
    print(f"Messages count: {len(result.messages)}")
    for msg in result.messages:
        src = getattr(msg, 'source', 'unknown')
        content = msg.content if hasattr(msg, 'content') else str(msg)
        print(f"  [{src}]: {content[:200]}")
    print()


async def test_selector_group_chat():
    """测试 2: SelectorGroupChat 最简版"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

    print("="*50)
    print("TEST 2: SelectorGroupChat (simple)")
    print("="*50)

    client = get_model_client()

    agent_a = AssistantAgent(
        name="Solver",
        model_client=client,
        system_message="You solve math problems. Show your work, then say FINAL_ANSWER: <answer>",
        description="Solves math problems.",
    )
    agent_b = AssistantAgent(
        name="Checker",
        model_client=client,
        system_message="You check math solutions. If correct, repeat the answer as FINAL_ANSWER: <answer>",
        description="Checks solutions for correctness.",
    )

    termination = TextMentionTermination("FINAL_ANSWER") | MaxMessageTermination(6)

    selector_prompt = """Select the next speaker from {participants}.
If no one has solved the problem yet, select 'Solver'.
If Solver has provided an answer, select 'Checker'.
Only return the role name, nothing else.
{roles}"""

    team = SelectorGroupChat(
        participants=[agent_a, agent_b],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
    )

    try:
        result = await team.run(task="What is 15 * 37? Give the exact number.")
        print(f"Messages count: {len(result.messages)}")
        for msg in result.messages:
            src = getattr(msg, 'source', 'unknown')
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"  [{src}]: {content[:300]}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


async def test_round_robin():
    """测试 3: RoundRobinGroupChat (不依赖 LLM 选择)"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

    print("="*50)
    print("TEST 3: RoundRobinGroupChat")
    print("="*50)

    client = get_model_client()

    planner = AssistantAgent(
        name="Planner",
        model_client=client,
        system_message="You plan how to solve problems. Create a brief plan with 2-3 steps.",
    )
    solver = AssistantAgent(
        name="Solver",
        model_client=client,
        system_message="You execute plans and solve problems. Say FINAL_ANSWER: <answer> when done.",
    )

    termination = TextMentionTermination("FINAL_ANSWER") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        participants=[planner, solver],
        termination_condition=termination,
    )

    try:
        result = await team.run(task="What is 15 * 37? Give the exact number.")
        print(f"Messages count: {len(result.messages)}")
        for msg in result.messages:
            src = getattr(msg, 'source', 'unknown')
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"  [{src}]: {content[:300]}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


async def main():
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {BASE_URL}")
    print(f"Key: {API_KEY[:10]}..." if API_KEY else "Key: NOT SET")
    print()

    # 先测试最简单的
    await test_simple_chat()

    # 再测试 RoundRobin（不需要 LLM 选择 next speaker）
    await test_round_robin()

    # 最后测试 SelectorGroupChat
    await test_selector_group_chat()


if __name__ == "__main__":
    asyncio.run(main())
