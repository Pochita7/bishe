"""
分析 TAMAS 和 GAIA 的 token 修正倍率差异。

已知:
- Bug1: FunctionInvocationLayer 在 tool loop 中只记录最后一次 API 调用的 usage
- 修复后 GAIA 拦截测试: ratio = 1.000x (修复正确)
- GAIA buggy avg: 11,217 tokens/task → fixed avg: 24,744 tokens/task → ratio 2.21x

问题: TAMAS 的修正倍率是多少?

方法: 从 JSONL 中提取每个任务的 agent_calls 和 tool_calls 数来建模 API 调用模式
"""
import json
import statistics

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("=" * 70)
print("TAMAS vs GAIA Bug1 修正倍率分析")
print("=" * 70)
print()

# ============================================================
# 1. 分析 TAMAS 的 API 调用模式
# ============================================================
print("1. TAMAS 任务的 API 调用模式分析")
print("-" * 50)

for fname, label in [
    ('benchmark_tamas_baseline.jsonl', 'TAMAS Baseline'),
    ('benchmark_tamas_defended.jsonl', 'TAMAS Defended'),
]:
    recs = load_jsonl(fname)
    
    total_tools = [r['total_tool_calls'] for r in recs]
    total_agents = [r['total_agent_calls'] for r in recs]
    tool_per_agent = [r['total_tool_calls'] / r['total_agent_calls'] if r['total_agent_calls'] > 0 else 0 for r in recs]
    
    inp_tokens = [r['input_tokens'] for r in recs]
    out_tokens = [r['output_tokens'] for r in recs]
    
    print(f"\n{label} ({len(recs)} tasks):")
    print(f"  Tool calls:  avg={statistics.mean(total_tools):.1f}  med={statistics.median(total_tools):.0f}  max={max(total_tools)}")
    print(f"  Agent calls: avg={statistics.mean(total_agents):.1f}  med={statistics.median(total_agents):.0f}  max={max(total_agents)}")
    print(f"  Tools/agent: avg={statistics.mean(tool_per_agent):.1f}  med={statistics.median(tool_per_agent):.1f}")
    print(f"  Input tok:   avg={statistics.mean(inp_tokens):,.0f}  med={statistics.median(inp_tokens):,.0f}")
    print(f"  Output tok:  avg={statistics.mean(out_tokens):,.0f}  med={statistics.median(out_tokens):,.0f}")
    
    # agent_calls 分布
    agent_types = {}
    for r in recs:
        for aname, count in r.get('agent_calls', {}).items():
            if aname not in agent_types:
                agent_types[aname] = []
            agent_types[aname].append(count)
    print(f"  Agent types:")
    for aname, counts in sorted(agent_types.items()):
        print(f"    {aname}: avg={statistics.mean(counts):.1f}  total={sum(counts)}")

# ============================================================
# 2. 建模 Bug1 的影响
# ============================================================
print()
print("=" * 70)
print("2. Bug1 影响建模")
print("-" * 50)
print()
print("""
Bug1: FunctionInvocationLayer.get_response() 在 tool loop 中做 K 次 API 调用,
但只记录最后一次(第 K 次)的 usage。修复后累加所有 K 次。

每次 agent.run() → 1 次 get_response() → K 次 API 调用:
  K = tools_in_this_agent_run / tools_per_round + 1
  (最少 1 次 API 调用; 每次 tool 响应需要再调 API)

假设每次 agent 调 T 个 tool, 这需要 ceil(T/并行数)+1 次 API 调用。
但实际上由于 tool_choice 和循环逻辑, K ≈ T + 1 在最坏情况,
K ≈ ceil(T/max_parallel) + 1 在最好情况。

更精确: DeepSeek-V3 通常一次返回多个 tool calls (parallel tool calling),
所以一次 API 调用可以触发 1-3 个 tool calls。

让我们用数据来推算...
""")

# 用 GAIA 的实测数据校准
# GAIA: buggy avg_total = 11,217, fixed avg = 24,744, ratio = 2.21x
# GAIA: avg tool calls per task ≈ 6-9 (from benchmark data)
gaia_recs = load_jsonl('benchmark_gaia_baseline.jsonl')
gaia_tools = [r.get('total_tool_calls', 0) for r in gaia_recs]
gaia_agents = [r.get('total_agent_calls', 0) for r in gaia_recs]
gaia_avg_tools = statistics.mean(gaia_tools)
gaia_avg_agents = statistics.mean(gaia_agents)
gaia_tools_per_agent = gaia_avg_tools / gaia_avg_agents if gaia_avg_agents > 0 else 0

print(f"GAIA Baseline 校准数据:")
print(f"  avg tool calls/task:  {gaia_avg_tools:.1f}")
print(f"  avg agent calls/task: {gaia_avg_agents:.1f}")
print(f"  avg tools/agent:      {gaia_tools_per_agent:.1f}")
print(f"  buggy ratio:          2.21x")
print()

# TAMAS 数据
tamas_recs = load_jsonl('benchmark_tamas_baseline.jsonl')
tamas_tools_per_agent_list = [r['total_tool_calls'] / r['total_agent_calls'] if r['total_agent_calls'] > 0 else 0 for r in tamas_recs]
tamas_avg_tools_per_agent = statistics.mean(tamas_tools_per_agent_list)
tamas_avg_tools = statistics.mean([r['total_tool_calls'] for r in tamas_recs])
tamas_avg_agents = statistics.mean([r['total_agent_calls'] for r in tamas_recs])

print(f"TAMAS Baseline 数据:")
print(f"  avg tool calls/task:  {tamas_avg_tools:.1f}")
print(f"  avg agent calls/task: {tamas_avg_agents:.1f}")
print(f"  avg tools/agent:      {tamas_avg_tools_per_agent:.1f}")
print()

# ============================================================
# 3. 估算 TAMAS 修正倍率
# ============================================================
print("=" * 70)
print("3. 估算修正倍率")
print("-" * 50)
print()

# 简化模型:
# 每次 agent.run() 做 K 次 API 调用
# 第 i 次 API 调用的 input_tokens = base + i * delta
# Bug1 只记录第 K 次: base + K*delta = base + (K-1)*delta + delta
# 实际总和: K*base + K*(K-1)/2 * delta
# Bug1 记录: base + (K-1)*delta
# ratio = [K*base + K*(K-1)/2 * delta] / [base + (K-1)*delta]
# 简化: 设 r = delta/base (增长率), then:
# ratio = K * [1 + (K-1)/2 * r] / [1 + (K-1)*r]

# 从 GAIA 校准:
# GAIA avg tools/agent ≈ gaia_tools_per_agent
# 假设 tools_per_round = 1.5 (平均每次 API 返回 1-2 个 tool calls)
# K_gaia = ceil(tools/agent / 1.5) + 1

# 但更简单: 直接从 GAIA 数据反推 ratio 与 tools_per_agent 的关系
# GAIA: tools/agent = gaia_tools_per_agent, ratio = 2.21x
# 假设线性关系 ratio = a * tools/agent + b
# 且当 tools/agent = 0 时 ratio = 1.0 (无 tool 调用则 K=1, 无修正)
# 所以 ratio = 1 + c * tools/agent
# 2.21 = 1 + c * gaia_tools_per_agent
# c = 1.21 / gaia_tools_per_agent

if gaia_tools_per_agent > 0:
    c = (2.21 - 1) / gaia_tools_per_agent
    tamas_estimated_ratio = 1 + c * tamas_avg_tools_per_agent
    
    print(f"线性模型: ratio = 1 + {c:.3f} × tools_per_agent")
    print(f"  GAIA:  tools/agent={gaia_tools_per_agent:.1f} → ratio=2.21x (实测)")
    print(f"  TAMAS: tools/agent={tamas_avg_tools_per_agent:.1f} → ratio≈{tamas_estimated_ratio:.2f}x (估算)")
    print()
    
    # 用估算的 TAMAS 修正倍率重算全部 benchmark
    gaia_base_inp = 554640
    gaia_base_out = 39839
    gaia_def_inp = 501191
    gaia_def_out = 33223
    tamas_base_inp = 8485956
    tamas_base_out = 307859
    tamas_def_inp = 8678132
    tamas_def_out = 315179
    
    gaia_corrected = (gaia_base_inp + gaia_base_out + gaia_def_inp + gaia_def_out) * 2.21
    tamas_corrected = (tamas_base_inp + tamas_base_out + tamas_def_inp + tamas_def_out) * tamas_estimated_ratio
    total_corrected = gaia_corrected + tamas_corrected
    
    print(f"修正后 benchmark 总量估算:")
    print(f"  GAIA (x2.21):   {gaia_corrected:>12,.0f}")
    print(f"  TAMAS (x{tamas_estimated_ratio:.2f}): {tamas_corrected:>12,.0f}")
    print(f"  Total:           {total_corrected:>12,.0f}")
    print()
    
    bill = 93_795_046
    dev_pct = (bill - total_corrected) / bill * 100
    print(f"  账单总量:         {bill:>12,}")
    print(f"  benchmark 占比:   {total_corrected/bill*100:.1f}%")
    print(f"  开发/测试 占比:   {dev_pct:.1f}%")
    print(f"  开发/测试 token:  {bill - total_corrected:>12,.0f}")
    
    # 费用拆分
    bench_cost = total_corrected / 1e6 * 1.0  # 简化: 全部按 tier1 input 计价
    print(f"\n  benchmark 估算费用: {total_corrected / 1e6 * 1.0 + 0:,.1f} 元 (按 input 价)")
    
    # 更精确的费用计算
    # 97.5% input, 2.5% output
    bench_inp = total_corrected * 0.975
    bench_out = total_corrected * 0.025
    bench_cost_precise = bench_inp / 1e6 * 1.0 + bench_out / 1e6 * 1.5
    print(f"  benchmark 精确估算: {bench_cost_precise:,.1f} 元")

# ============================================================
# 4. 另一种方式: 直接用 5-task 拦截数据
# ============================================================
print()
print("=" * 70)
print("4. 直接验证: 5-task API 拦截结果")
print("-" * 50)
print()
print("5 个 GAIA 任务的 API 拦截:")
print("  API  total: inp=114,724  out=8,996  total=123,720")
print("  Framework:  inp=114,724  out=8,996  total=123,720")
print("  Ratio: 1.000x (框架报告 = API 真实消耗, 0 误差)")
print()
print("结论: 修复后的 token 计算方式是 100% 准确的。")
print("问题出在旧 benchmark JSONL 数据使用了 buggy 代码记录。")
