"""
Token 消耗分析与修正估算

问题：agent-framework 的 FunctionInvocationLayer 在工具循环中只返回最后一次 API 调用的 usage_details，
      中间轮次的 token 全部丢失。

修正原理：
  一次 agent.run() 内部，如果产生了 N 次工具调用，则实际发生了 N+1 次 LLM API 调用：
    - 第 1 次：输入 = prompt + tools schema → 输出 = function_call
    - 第 2~N 次：输入 = 累积上下文 + 工具结果 → 输出 = function_call
    - 第 N+1 次：输入 = 累积上下文 + 最后工具结果 → 输出 = 文本回答

  我们只记录了第 N+1 次的 token。
  但可以利用"每次 API 调用的 input 包含所有之前的消息"这一特性来估算。

  如果一次 agent.run() 内有 k 次工具调用：
    - 记录到的 input_tokens ≈ 最终一次调用的 input（包含了全部历史）
    - 丢失的 input_tokens ≈ sum(第1次到第k次的 input)
    - 由于每次 input 逐渐增长（累积工具结果），可以近似为等差数列
    - 丢失的 output_tokens ≈ k 次 function_call 的 output（通常较短，~50-200 tokens/次）

  简化估算：
    total_api_calls_per_agent = tool_calls_made + 1  (发给这个 agent 的工具调用数 + 初始调用)
    但我们的 tool_calls 是按整个 task 统计的，不是按 agent 统计的。

  更实用的估算方法：
    recorded_input_tokens = 最终轮的 input（已包含全部历史）
    typical_tool_result ≈ 200-500 tokens
    每多一轮工具调用，input 增加 ≈ (tool_call_output + tool_result) ≈ 300 tokens
    
    estimated_total_input ≈ recorded_input × (1 + 0.5 × tool_calls_per_task)
    estimated_total_output ≈ recorded_output + tool_calls × 100  (每次 function_call ~100 tokens)
"""
import json
import statistics
from collections import defaultdict

def analyze_tokens(label, path):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f]

    print(f"\n{'='*70}")
    print(f"  {label} ({len(rows)} tasks)")
    print(f"{'='*70}")

    # 原始记录值
    rec_input = [r.get("input_tokens", 0) for r in rows]
    rec_output = [r.get("output_tokens", 0) for r in rows]
    rec_total = [r.get("total_tokens", 0) for r in rows]
    tool_calls = [r.get("total_tool_calls", 0) for r in rows]
    agent_calls = [r.get("total_agent_calls", 0) for r in rows]

    zero_tok = sum(1 for t in rec_total if t == 0)

    print(f"\n  [记录值] (被低估)")
    print(f"    total_tokens:  sum={sum(rec_total):>12,}  mean={statistics.mean(rec_total):>10,.0f}  median={statistics.median(rec_total):>10,.0f}")
    print(f"    input_tokens:  sum={sum(rec_input):>12,}  mean={statistics.mean(rec_input):>10,.0f}")
    print(f"    output_tokens: sum={sum(rec_output):>12,}  mean={statistics.mean(rec_output):>10,.0f}")
    print(f"    zero-token tasks: {zero_tok}")
    print(f"    agent_calls:  mean={statistics.mean(agent_calls):>6.1f}  total={sum(agent_calls):>6}")
    print(f"    tool_calls:   mean={statistics.mean(tool_calls):>6.1f}  total={sum(tool_calls):>6}")
    if sum(agent_calls) > 0:
        print(f"    tokens/agent_call: {sum(rec_total)/sum(agent_calls):>,.0f} (低估)")

    # 估算修正
    # 方法：每个 task 中，每次工具调用 ≈ 额外产生：
    #   - input 增量: ~300 tokens (function_call spec + result 被累积到下一轮)
    #   - output: ~100 tokens (function_call json)
    # 另外，非最终轮的 input 被完全丢失，但它们是最终轮 input 的子集（因为消息累积）
    # 所以我们可以估算：
    #   丢失的 input ≈ sum_{i=1}^{k} (final_input - (k-i)*avg_increment)
    #   简化 ≈ final_input * k * 0.7  (每轮平均 70% 的最终 input)
    
    est_total_list = []
    est_input_list = []
    est_output_list = []
    
    for r in rows:
        ri = r.get("input_tokens", 0)
        ro = r.get("output_tokens", 0)
        tc = r.get("total_tool_calls", 0)
        ac = r.get("total_agent_calls", 0)
        
        if ri == 0 and ro == 0:
            est_total_list.append(0)
            est_input_list.append(0)
            est_output_list.append(0)
            continue
        
        # 每个 agent call 中的工具调用数 ≈ total_tool_calls / total_agent_calls
        # (有些 agent 如 Planner 没有工具调用)
        tools_per_agent = tc / max(ac, 1)
        
        # 估算：记录的 input 是所有 agent 最后一轮的 input 之和
        # 丢失的 = 每个 agent 内部前 k 轮的 input
        # 保守估算：丢失的 ≈ 记录的 input × (tools_per_agent × 0.6)
        # (每多一轮工具调用，丢失约60%的该轮input，因为是递增的子集)
        input_multiplier = 1 + tools_per_agent * 0.6
        est_input = int(ri * input_multiplier)
        
        # 丢失的 output ≈ 每次工具调用额外产生 ~80 tokens 的 function_call JSON
        est_output = ro + tc * 80
        
        est_total = est_input + est_output
        est_total_list.append(est_total)
        est_input_list.append(est_input)
        est_output_list.append(est_output)

    print(f"\n  [估算值] (修正后)")
    print(f"    total_tokens:  sum={sum(est_total_list):>12,}  mean={statistics.mean(est_total_list):>10,.0f}  median={statistics.median(est_total_list):>10,.0f}")
    print(f"    input_tokens:  sum={sum(est_input_list):>12,}  mean={statistics.mean(est_input_list):>10,.0f}")
    print(f"    output_tokens: sum={sum(est_output_list):>12,}  mean={statistics.mean(est_output_list):>10,.0f}")
    if sum(agent_calls) > 0:
        print(f"    tokens/agent_call: {sum(est_total_list)/sum(agent_calls):>,.0f} (估算)")

    # 放大倍数
    if sum(rec_total) > 0:
        ratio = sum(est_total_list) / sum(rec_total)
        print(f"\n  [修正倍数] {ratio:.2f}x")
    
    return {
        "label": label,
        "count": len(rows),
        "recorded": {"input": sum(rec_input), "output": sum(rec_output), "total": sum(rec_total)},
        "estimated": {"input": sum(est_input_list), "output": sum(est_output_list), "total": sum(est_total_list)},
        "agent_calls": sum(agent_calls),
        "tool_calls": sum(tool_calls),
    }


# 分析所有 4 个实验
all_results = []
for label, path in [
    ("GAIA Baseline", "benchmark_gaia_baseline.jsonl"),
    ("GAIA Defended", "benchmark_gaia_defended.jsonl"),
    ("TAMAS Baseline", "benchmark_tamas_baseline_v2.jsonl"),
    ("TAMAS Defended", "benchmark_tamas_defended_v2.jsonl"),
]:
    try:
        r = analyze_tokens(label, path)
        all_results.append(r)
    except Exception as e:
        print(f"  Error: {e}")

# 汇总表
print(f"\n\n{'='*70}")
print("  汇总对比")
print(f"{'='*70}")
print(f"  {'实验':<20} {'任务数':>6} {'记录tokens':>14} {'估算tokens':>14} {'倍数':>6} {'工具调用':>8}")
for r in all_results:
    ratio = r["estimated"]["total"] / r["recorded"]["total"] if r["recorded"]["total"] > 0 else 0
    print(f"  {r['label']:<20} {r['count']:>6} {r['recorded']['total']:>14,} {r['estimated']['total']:>14,} {ratio:>6.2f}x {r['tool_calls']:>8}")

print(f"\n  总计:")
total_rec = sum(r["recorded"]["total"] for r in all_results)
total_est = sum(r["estimated"]["total"] for r in all_results)
total_tools = sum(r["tool_calls"] for r in all_results)
print(f"  {'ALL':<20} {sum(r['count'] for r in all_results):>6} {total_rec:>14,} {total_est:>14,} {total_est/total_rec:.2f}x {total_tools:>8}")
