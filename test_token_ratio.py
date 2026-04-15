"""
测量修复后 GAIA 任务的真实 token 消耗
==============================================
跑 10 个 GAIA 任务(修复后)，直接测量每任务真实 token。
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import asyncio, sys, io, os, json, time
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_solver.config import get_text_client
from gaia_solver.solver import load_gaia_tasks, build_task_prompt, compare_answers
from mas.factory import create_gaia_team

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line.strip()))
                except: pass
    return data

async def main():
    tasks = load_gaia_tasks()
    
    # Pick 10 tasks evenly spread across the dataset
    n = len(tasks)
    step = max(1, n // 10)
    selected = [tasks[i] for i in range(0, n, step)][:10]
    
    client = get_text_client()
    team = create_gaia_team(client=client, max_rounds=3, verbose=False)
    
    results = []
    for i, task in enumerate(selected):
        tid = task['task_id']
        prompt = build_task_prompt(task)
        print(f"[{i+1}/10] {tid[:24]}...")
        
        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                team.run(task=prompt, task_id=tid, expected_answer=task.get('Final answer', '')),
                timeout=300,
            )
            m = result['metrics']
            elapsed = time.time() - t0
            rec = {
                'task_id': tid,
                'input_tokens': m.input_tokens,
                'output_tokens': m.output_tokens,
                'total_tokens': m.total_tokens,
                'tool_calls': m.total_tool_calls,
                'agent_calls': m.total_agent_calls,
                'elapsed': round(elapsed, 1),
                'status': 'ok',
            }
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            print(f"  TIMEOUT ({elapsed:.0f}s)")
            rec = {'task_id': tid, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                   'tool_calls': 0, 'agent_calls': 0, 'elapsed': round(elapsed, 1), 'status': 'timeout'}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            rec = {'task_id': tid, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                   'tool_calls': 0, 'agent_calls': 0, 'elapsed': round(elapsed, 1), 'status': 'error'}
        
        results.append(rec)
        if rec['status'] == 'ok':
            print(f"  inp={rec['input_tokens']:,} out={rec['output_tokens']:,} total={rec['total_tokens']:,} tools={rec['tool_calls']} agents={rec['agent_calls']} {elapsed:.0f}s")
        
        try:
            from gaia_solver.tools import _cleanup_browser
            _cleanup_browser()
        except: pass
    
    # Summary
    ok = [r for r in results if r['status'] == 'ok']
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(ok)}/{len(results)} tasks completed")
    print(f"{'='*70}")
    
    if ok:
        total_inp = sum(r['input_tokens'] for r in ok)
        total_out = sum(r['output_tokens'] for r in ok)
        total_tok = sum(r['total_tokens'] for r in ok)
        total_tools = sum(r['tool_calls'] for r in ok)
        total_agents = sum(r['agent_calls'] for r in ok)
        
        avg_inp = total_inp / len(ok)
        avg_out = total_out / len(ok)
        avg_tok = total_tok / len(ok)
        avg_tools = total_tools / len(ok)
        avg_agents = total_agents / len(ok)
        
        print(f"\nPer-task averages (修复后真实值):")
        print(f"  input_tokens:  {avg_inp:,.0f}")
        print(f"  output_tokens: {avg_out:,.0f}")
        print(f"  total_tokens:  {avg_tok:,.0f}")
        print(f"  tool_calls:    {avg_tools:.1f}")
        print(f"  agent_calls:   {avg_agents:.1f}")
        
        print(f"\nTotals across {len(ok)} tasks:")
        print(f"  input:  {total_inp:,}")
        print(f"  output: {total_out:,}")
        print(f"  total:  {total_tok:,}")
        
        # Extrapolate to full 53-task benchmark
        print(f"\nExtrapolated to 53 GAIA tasks:")
        ext_total = avg_tok * 53
        ext_inp = avg_inp * 53
        ext_out = avg_out * 53
        print(f"  est total tokens: {ext_total:,.0f}")
        print(f"  est input:  {ext_inp:,.0f}")
        print(f"  est output: {ext_out:,.0f}")
        
        # Cost calculation
        cost_t1 = ext_inp / 1e6 * 1.0 + ext_out / 1e6 * 1.5
        print(f"  est cost (tier1): {cost_t1:.2f} RMB")
        print(f"  est cost/task:    {cost_t1/53:.4f} RMB")
    
    with open('token_ratio_test_results.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nSaved to token_ratio_test_results.jsonl")

asyncio.run(main())
