"""Debug script to understand agent-framework workflow output structure"""
import asyncio
from agent_framework import Agent, Message, FunctionTool
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatBuilder


async def main():
    client = OpenAIChatClient(
        model_id="ep-20260212204648-nrlx2",
        api_key="d9452cdf-f0ec-41f7-9029-115170830afc",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    def add(a: int, b: int) -> str:
        """Add two numbers and return result."""
        return str(a + b)

    agent1 = Agent(
        client=client,
        name="Calculator",
        instructions="You are a calculator. Use the add tool to compute 2+3, then say FINAL_ANSWER: <result>",
        description="Computes math",
        tools=[FunctionTool(name="add", description="Add two integers", func=add)],
        default_options={"temperature": 0},
    )

    agent2 = Agent(
        client=client,
        name="Checker",
        instructions="Check the Calculator's answer. If correct, say FINAL_ANSWER: <answer>. Otherwise say FAIL.",
        description="Checks answers",
        default_options={"temperature": 0},
    )

    def selector(state) -> str:
        conv = state.conversation
        agents_spoke = []
        for msg in conv:
            a = getattr(msg, "author_name", "") or ""
            if a in ("Calculator", "Checker"):
                agents_spoke.append(a)
        if not agents_spoke or agents_spoke[-1] != "Calculator":
            return "Calculator"
        return "Checker"

    def term_cond(conv) -> bool:
        for msg in reversed(conv[-3:]):
            author = getattr(msg, "author_name", "") or ""
            text = msg.text if hasattr(msg, "text") else ""
            if author == "Checker" and "FINAL_ANSWER" in (text or "").upper():
                return True
        return False

    workflow = GroupChatBuilder(
        participants=[agent1, agent2],
        selection_func=selector,
        max_rounds=6,
        termination_condition=term_cond,
    ).build()

    result = await workflow.run("What is 2 + 3?")

    print("=== Result type:", type(result))
    print("=== Result is list?", isinstance(result, list))

    # check get_outputs
    outputs = result.get_outputs()
    print(f"\n=== get_outputs() returned {len(outputs)} items")
    for i, out in enumerate(outputs):
        print(f"  output[{i}] type={type(out)}, is_list={isinstance(out, list)}")
        if isinstance(out, list):
            print(f"  output[{i}] has {len(out)} messages")
            for j, msg in enumerate(out):
                print(f"    msg[{j}] type={type(msg)}")
                print(f"      author_name={getattr(msg, 'author_name', 'N/A')}")
                print(f"      role={getattr(msg, 'role', 'N/A')}")
                txt = msg.text if hasattr(msg, 'text') else 'NO .text'
                print(f"      text={repr(txt[:100])}")
                if hasattr(msg, 'contents') and msg.contents:
                    for k, c in enumerate(msg.contents):
                        print(f"      contents[{k}] type={getattr(c, 'type', '?')}, text={repr(getattr(c, 'text', None))[:80] if getattr(c, 'text', None) else repr(c)[:80]}")
        else:
            print(f"  output[{i}] = {repr(out)[:200]}")

    # Also iterate events
    print("\n=== Direct iteration over result (events):")
    for i, event in enumerate(result):
        print(f"  event[{i}] type='{event.type}', executor_id={getattr(event, 'executor_id', 'N/A')}")
        if event.type == "output":
            data = event.data
            print(f"    data type={type(data)}, is_list={isinstance(data, list)}")
            if isinstance(data, list):
                for j, msg in enumerate(data[:3]):
                    txt = msg.text if hasattr(msg, 'text') else str(msg)
                    print(f"    data[{j}] author={getattr(msg, 'author_name', '?')}, text={repr(txt[:80])}")


asyncio.run(main())
