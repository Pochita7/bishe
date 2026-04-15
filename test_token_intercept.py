"""
精确测量：在 OpenAI HTTP 层拦截每一次 API 调用的真实 token 消耗，
与 agent_framework 最终报告的 usage_details 对比，找出丢失的 token。
"""

import asyncio
import sys
import json
import functools
sys.path.insert(0, '.')

from gaia_solver.config import get_text_client
from gaia_solver.solver import load_gaia_tasks, build_task_prompt
from mas.factory import create_gaia_team

# ============================================================
# 全局拦截器：monkeypatch openai 底层 create 方法
# ============================================================
_api_calls = []  # 记录每一次底层 API 调用

def patch_openai_client(client):
    """
    在 OpenAI AsyncClient 的 chat.completions.create 上套一层 wrapper，
    记录每一次 API 调用的 raw usage 数据。
    """
    # client 是 agent_framework 的 OpenAIChatClient
    # 它内部有 _client 属性指向 openai.AsyncOpenAI
    inner = getattr(client, '_client', None) or getattr(client, 'client', None)
    if inner is None:
        # 尝试找到内层 client
        for attr_name in dir(client):
            attr = getattr(client, attr_name, None)
            if hasattr(attr, 'chat'):
                inner = attr
                break
    
    if inner is None:
        print("[WARN] Cannot find inner openai client to patch")
        return
    
    print(f"[PATCH] Found inner client: {type(inner).__name__}")
    
    original_create = inner.chat.completions.create
    
    @functools.wraps(original_create)
    async def intercepted_create(*args, **kwargs):
        resp = await original_create(*args, **kwargs)
        
        usage = resp.usage
        call_info = {
            'model': resp.model,
            'prompt_tokens': usage.prompt_tokens if usage else None,
            'completion_tokens': usage.completion_tokens if usage else None,
            'total_tokens': usage.total_tokens if usage else None,
            'has_tool_calls': any(
                c.message.tool_calls for c in resp.choices if hasattr(c, 'message') and c.message.tool_calls
            ) if resp.choices else False,
            'usage_is_none': usage is None,
        }
        
        # 检查 prompt_tokens_details (cached tokens)
        if usage and hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            ptd = usage.prompt_tokens_details
            call_info['cached_tokens'] = getattr(ptd, 'cached_tokens', 0) or 0
        
        _api_calls.append(call_info)
        
        n = len(_api_calls)
        inp = call_info['prompt_tokens'] or 0
        out = call_info['completion_tokens'] or 0
        tool = '🔧' if call_info['has_tool_calls'] else '💬'
        cached = call_info.get('cached_tokens', 0)
        print(f"  [API #{n:>3}] {tool} inp={inp:>7,} out={out:>5,} cached={cached:>6,} model={call_info['model']}")
        
        return resp
    
    inner.chat.completions.create = intercepted_create
    print("[PATCH] OpenAI create method patched successfully")


async def run_single_task(task, team):
    """运行单个 GAIA 任务，返回 (框架报告的 usage, API 拦截的总 usage)"""
    global _api_calls
    _api_calls = []  # 清空
    
    prompt = build_task_prompt(task)
    
    result = await asyncio.wait_for(
        team.run(task=prompt, task_id=task['task_id'], expected_answer=task['Final answer']),
        timeout=300,
    )
    
    # 框架报告的 token
    m = result['metrics']
    framework_inp = m.input_tokens
    framework_out = m.output_tokens
    
    # API 拦截的真实 token
    api_inp = sum(c['prompt_tokens'] or 0 for c in _api_calls)
    api_out = sum(c['completion_tokens'] or 0 for c in _api_calls)
    api_cached = sum(c.get('cached_tokens', 0) for c in _api_calls)
    n_calls = len(_api_calls)
    n_none = sum(1 for c in _api_calls if c['usage_is_none'])
    
    return {
        'task_id': task['task_id'],
        'n_api_calls': n_calls,
        'n_usage_none': n_none,
        'api_input': api_inp,
        'api_output': api_out,
        'api_cached': api_cached,
        'framework_input': framework_inp,
        'framework_output': framework_out,
        'ratio_input': api_inp / framework_inp if framework_inp > 0 else float('inf'),
        'ratio_output': api_out / framework_out if framework_out > 0 else float('inf'),
        'answer': result.get('answer', ''),
        'is_correct': result.get('is_correct', None),
    }


async def main():
    tasks = load_gaia_tasks()
    
    # 取 5 个均匀分布的任务
    n = len(tasks)
    indices = [int(i * n / 5) for i in range(5)]
    selected = [tasks[i] for i in indices]
    
    client = get_text_client()
    
    # Patch the client BEFORE creating agents
    patch_openai_client(client)
    
    team = create_gaia_team(client=client, max_rounds=3, verbose=True)
    
    # 也 patch vision client（如果存在）
    from gaia_solver.config import get_vision_client
    try:
        vclient = get_vision_client()
        patch_openai_client(vclient)
    except Exception:
        pass
    
    results = []
    for i, task in enumerate(selected):
        print(f"\n{'='*70}")
        print(f"Task {i+1}/5: {task['task_id'][:12]}...")
        print(f"{'='*70}")
        
        try:
            r = await run_single_task(task, team)
            results.append(r)
            
            print(f"\n  --- Task Summary ---")
            print(f"  API calls: {r['n_api_calls']} (usage=None: {r['n_usage_none']})")
            print(f"  API  total: inp={r['api_input']:>8,}  out={r['api_output']:>6,}  cached={r['api_cached']:>6,}")
            print(f"  Framework:  inp={r['framework_input']:>8,}  out={r['framework_output']:>6,}")
            print(f"  Ratio:      inp={r['ratio_input']:.2f}x  out={r['ratio_output']:.2f}x")
            print(f"  Correct: {r['is_correct']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({len(results)} tasks)")
    print(f"{'='*70}")
    
    if results:
        total_api_inp = sum(r['api_input'] for r in results)
        total_api_out = sum(r['api_output'] for r in results)
        total_api_cached = sum(r['api_cached'] for r in results)
        total_fw_inp = sum(r['framework_input'] for r in results)
        total_fw_out = sum(r['framework_output'] for r in results)
        total_api_calls = sum(r['n_api_calls'] for r in results)
        total_none = sum(r['n_usage_none'] for r in results)
        
        print(f"  Total API calls: {total_api_calls} (usage=None: {total_none})")
        print(f"  API  total:      inp={total_api_inp:>10,}  out={total_api_out:>8,}  cached={total_api_cached:>8,}")
        print(f"  Framework total: inp={total_fw_inp:>10,}  out={total_fw_out:>8,}")
        print(f"  Ratio:           inp={total_api_inp/total_fw_inp:.3f}x  out={total_api_out/total_fw_out:.3f}x")
        print(f"  OVERALL ratio:   {(total_api_inp+total_api_out)/(total_fw_inp+total_fw_out):.3f}x")
        
        print(f"\n  Per-task details:")
        for r in results:
            tid = r['task_id'][:12]
            print(f"    {tid}  calls={r['n_api_calls']:>2}  "
                  f"API_inp={r['api_input']:>7,} fw_inp={r['framework_input']:>7,} ratio={r['ratio_input']:.2f}x  "
                  f"API_out={r['api_output']:>5,} fw_out={r['framework_output']:>5,} ratio={r['ratio_output']:.2f}x")
        
        # 保存详细结果
        with open('token_intercept_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to token_intercept_results.json")

asyncio.run(main())
