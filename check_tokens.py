"""检查 GAIA benchmark 结果中的 token 统计"""
import json
import statistics

for label, path in [
    ("GAIA Baseline", "benchmark_gaia_baseline.jsonl"),
    ("GAIA Defended", "benchmark_gaia_defended.jsonl"),
]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    total_tokens_list = [r.get("total_tokens", 0) for r in rows]
    input_tokens_list = [r.get("input_tokens", 0) for r in rows]
    output_tokens_list = [r.get("output_tokens", 0) for r in rows]
    agent_calls_list = [r.get("total_agent_calls", 0) for r in rows]
    tool_calls_list = [r.get("total_tool_calls", 0) for r in rows]

    zero_token_tasks = sum(1 for t in total_tokens_list if t == 0)

    print(f"\n{'='*60}")
    print(f"  {label} ({len(rows)} tasks)")
    print(f"{'='*60}")
    print(f"  zero-token tasks: {zero_token_tasks}/{len(rows)}")
    print(f"  total_tokens:  sum={sum(total_tokens_list):,}  mean={statistics.mean(total_tokens_list):,.0f}  "
          f"median={statistics.median(total_tokens_list):,.0f}  max={max(total_tokens_list):,}")
    print(f"  input_tokens:  sum={sum(input_tokens_list):,}  mean={statistics.mean(input_tokens_list):,.0f}")
    print(f"  output_tokens: sum={sum(output_tokens_list):,}  mean={statistics.mean(output_tokens_list):,.0f}")
    print(f"  agent_calls:   mean={statistics.mean(agent_calls_list):.1f}  total={sum(agent_calls_list)}")
    print(f"  tool_calls:    mean={statistics.mean(tool_calls_list):.1f}  total={sum(tool_calls_list)}")

    # 每agent call的平均token
    total_calls = sum(agent_calls_list)
    if total_calls > 0:
        print(f"  tokens_per_agent_call: {sum(total_tokens_list)/total_calls:,.0f}")

    # 展示几个样例
    print(f"\n  前5个任务的 token 详情:")
    for r in rows[:5]:
        tid = r.get("task_id", "?")[:16]
        print(f"    {tid}: total={r.get('total_tokens',0):,}  "
              f"input={r.get('input_tokens',0):,}  output={r.get('output_tokens',0):,}  "
              f"agents={r.get('total_agent_calls',0)}  tools={r.get('total_tool_calls',0)}")
