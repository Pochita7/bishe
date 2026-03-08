"""诊断脚本: 单独运行 Mercedes Sosa 任务，打印详细时间信息"""
import asyncio
import time
import sys

sys.path.insert(0, ".")

async def main():
    from mas.factory import create_gaia_team

    team = create_gaia_team(verbose=True)
    
    question = (
        "How many studio albums were published by Mercedes Sosa "
        "between 2000 and 2009 (included)? "
        "You can use the latest 2022 version of english wikipedia."
    )
    
    print(f"=" * 70)
    print(f"[DEBUG] 开始运行 Mercedes Sosa 任务")
    print(f"  max_rounds={team.max_rounds}, worker_timeout={team.worker_timeout}, "
          f"max_tool_calls={team.max_tool_calls_per_worker}")
    print(f"=" * 70)
    
    start = time.time()
    try:
        result = await asyncio.wait_for(
            team.run(task=question, task_id="debug-mercedes", expected_answer="3"),
            timeout=300,
        )
        elapsed = time.time() - start
        print(f"\n{'=' * 70}")
        print(f"[RESULT] 耗时: {elapsed:.1f}s")
        print(f"  答案: {result.get('answer', 'NO_ANSWER')}")
        print(f"  轮数: {result.get('rounds', '?')}")
        print(f"  对话: {result.get('turns', '?')} turns")
        
        m = result.get("metrics")
        if m:
            print(f"  Tokens: in={m.input_tokens}, out={m.output_tokens}")
            print(f"  Tool calls: {m.total_tool_calls} ({m.tool_calls})")
            print(f"  Agent calls: {m.agent_calls}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"\n[TIMEOUT] 超时 ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[ERROR] {e} ({elapsed:.1f}s)")

if __name__ == "__main__":
    asyncio.run(main())
