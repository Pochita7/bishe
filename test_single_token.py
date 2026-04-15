"""Quick single-task test to measure real new/old token ratio."""
import asyncio, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_solver.config import get_text_client
from gaia_solver.solver import load_gaia_tasks, build_task_prompt, compare_answers
from mas.factory import create_gaia_team

async def test():
    tasks = load_gaia_tasks()
    target = tasks[0]
    
    client = get_text_client()
    team = create_gaia_team(client=client, max_rounds=3, verbose=True)
    prompt = build_task_prompt(target)
    
    tid = target['task_id']
    print(f"Task: {tid[:24]}")
    print(f"Q: {target['Question'][:100]}")
    print()
    
    result = await asyncio.wait_for(
        team.run(task=prompt, task_id=tid, expected_answer=target['Final answer']),
        timeout=300,
    )
    m = result['metrics']
    print(f"\n====== NEW (fixed) SUMMARY ======")
    print(f"input_tokens={m.input_tokens}  output_tokens={m.output_tokens}  total={m.total_tokens}")
    print(f"agent_calls total={m.total_agent_calls}  tool_calls total={m.total_tool_calls}")
    print(f"agents: {m.agent_calls}")
    print(f"tools: {m.tool_calls}")

    old = None
    with open('benchmark_gaia_baseline.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip())
            if d['task_id'] == tid:
                old = d
                break
    if old:
        print(f"\n====== OLD (buggy) record ======")
        print(f"input_tokens={old['input_tokens']}  output_tokens={old['output_tokens']}  total={old['total_tokens']}")
        print(f"agent_calls: {old['agent_calls']}")
        print(f"tool_calls: {old['tool_calls']}")
        ratio = m.total_tokens / old['total_tokens'] if old['total_tokens'] > 0 else 0
        inp_ratio = m.input_tokens / old['input_tokens'] if old['input_tokens'] > 0 else 0
        out_ratio = m.output_tokens / old['output_tokens'] if old['output_tokens'] > 0 else 0
        print(f"\nRatio (new/old): total={ratio:.2f}x  input={inp_ratio:.2f}x  output={out_ratio:.2f}x")

asyncio.run(test())
