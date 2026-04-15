"""MAS Workflow Demo - Run a real GAIA task to show multi-agent collaboration"""
import asyncio
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


async def demo():
    with open('gaia_data/gaia_level1.jsonl', encoding='utf-8') as f:
        tasks = [json.loads(l) for l in f]

    simple_tasks = [t for t in tasks if not t.get('file_name') and len(t['Question']) < 200]
    target = None
    for t in simple_tasks:
        if 'Mercedes Sosa' in t['Question']:
            target = t
            break
    if not target:
        target = simple_tasks[0]

    task_id = target['task_id']
    question = target['Question']
    answer = target['Final answer']

    print("=" * 70)
    print("  MAS Workflow Demo - Real GAIA Dataset Task")
    print("=" * 70)
    print(f"\n  Task ID:  {task_id}")
    print(f"  Question: {question}")
    print(f"  Expected: {answer}")
    print("=" * 70)

    print("\n[Step 0] Creating MAS Team (Planner + 5 Workers, 15 tools)...")
    from mas.factory import create_gaia_team
    team = create_gaia_team(max_rounds=3, verbose=True)

    print(f"\n{'='*70}")
    print(f"  Starting execution...")
    print(f"{'='*70}")

    result = await team.run(
        task=question,
        task_id=task_id,
        expected_answer=answer,
    )

    print(f"\n{'='*70}")
    print(f"  Execution Result")
    print(f"{'='*70}")
    print(f"  Final Answer:    {result['answer']}")
    print(f"  Expected:        {answer}")
    print(f"  Correct:         {'YES' if str(result['answer']).strip() == str(answer).strip() else 'NO'}")
    print(f"  Planning Rounds: {result['rounds']}")
    print(f"  Total Turns:     {result['turns']}")
    print(f"  Elapsed:         {result['elapsed']}s")

    print(f"\n{'='*70}")
    print(f"  Full Message Flow (Multi-Agent Collaboration)")
    print(f"{'='*70}")
    for i, msg in enumerate(result['messages'], 1):
        source = msg['source']
        content = msg['content']
        if len(content) > 800:
            content = content[:800] + "\n      ...(truncated)"
        print(f"\n  --- [{i}] {source} ---")
        for line in content.split('\n'):
            print(f"  | {line}")

    metrics = result.get('metrics')
    if metrics:
        print(f"\n{'='*70}")
        print(f"  Performance Metrics")
        print(f"{'='*70}")
        print(f"  Input tokens:   {metrics.input_tokens}")
        print(f"  Output tokens:  {metrics.output_tokens}")
        print(f"  Total tokens:   {metrics.total_tokens}")
        print(f"  Tool calls:     {metrics.tool_calls}")
        print(f"  Agent calls:    {metrics.agent_calls}")
        print(f"  Handoffs:       {metrics.handoff_count}")

    return result


if __name__ == "__main__":
    result = asyncio.run(demo())
    correct = str(result['answer']).strip() == '3'
    print(f"\nDemo finished. Answer: {result['answer']} | Correct: {correct}")
