"""
分析真实 Token 消耗和费用
基于 Bug1 token 丢失模型 + 火山引擎官方定价
"""
import json
import math
from collections import Counter

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except:
                    pass
    return data


# ============================================================
# Part 1: 基础统计
# ============================================================
print("=" * 80)
print("Part 1: 各实验基础统计")
print("=" * 80)

benchmarks = [
    ('GAIA Baseline', 'benchmark_gaia_baseline.jsonl'),
    ('GAIA Defended', 'benchmark_gaia_defended.jsonl'),
    ('TAMAS Baseline', 'benchmark_tamas_baseline_v2.jsonl'),
    ('TAMAS Defended', 'benchmark_tamas_defended_v2.jsonl'),
]

all_results = {}

for label, path in benchmarks:
    data = load_jsonl(path)
    n = len(data)

    total_agent_calls = sum(d.get('total_agent_calls', 0) for d in data)
    total_tool_calls = sum(d.get('total_tool_calls', 0) for d in data)
    total_inp = sum(d.get('input_tokens', 0) for d in data)
    total_out = sum(d.get('output_tokens', 0) for d in data)
    total_tok = sum(d.get('total_tokens', 0) for d in data)

    planner_calls = 0
    worker_calls = 0
    for d in data:
        ac = d.get('agent_calls', {})
        for name, cnt in ac.items():
            if 'planner' in name.lower():
                planner_calls += cnt
            else:
                worker_calls += cnt

    avg_tools_per_worker = total_tool_calls / worker_calls if worker_calls > 0 else 0

    all_results[label] = {
        'data': data, 'n': n,
        'total_agent_calls': total_agent_calls,
        'planner_calls': planner_calls,
        'worker_calls': worker_calls,
        'total_tool_calls': total_tool_calls,
        'avg_tools_per_worker': avg_tools_per_worker,
        'rec_inp': total_inp, 'rec_out': total_out, 'rec_tok': total_tok,
    }

    rounds_dist = Counter(d.get('rounds', 0) for d in data)
    print(f"\n{label} ({n} tasks):")
    print(f"  agent_calls: {total_agent_calls} (planner={planner_calls}, worker={worker_calls})")
    print(f"  tool_calls: {total_tool_calls}")
    print(f"  avg tools/worker: {avg_tools_per_worker:.1f}")
    print(f"  recorded tokens: inp={total_inp:,} out={total_out:,} total={total_tok:,}")
    print(f"  rounds dist: {dict(rounds_dist)}")


# ============================================================
# Part 2: Bug1 Token 丢失建模
# ============================================================
print("\n")
print("=" * 80)
print("Part 2: Bug1 Token 丢失建模")
print("=" * 80)
print("""
Bug1 原理: FunctionInvocationLayer 在工具调用循环中只返回*最后一次*API调用的usage.
Worker agent的典型流程:
  API Call 1: [system + task] → model returns tool_calls (tool_use)
  API Call 2: [system + task + tool_calls + tool_results] → more tool_calls or final answer
  ...
  API Call K: [accumulated context] → final answer

Bug1只记录了 Call K 的 usage. 前面 K-1 次的都丢了.

关键: input tokens 随 K 增长 (context 累积), 所以丢失的不只是 1/K.
模型:
  - 每次 API call 的 input = base * (1 + growth_factor * call_index)  
  - growth_factor ≈ 0.3-0.5 (tool results 占 base 的 30-50%)
  - total_input / last_input = [K + g*K*(K-1)/2] / [1 + g*(K-1)]
  - total_output ≈ K * avg_output (每次API调用都产生output)
""")

# Pricing
T1_INP = 1.0   # input <=32k, 元/百万tokens
T1_OUT = 1.5   # output when input <=32k
T2_INP = 2.0   # input 32k-128k
T2_OUT = 3.0   # output when input 32k-128k

def model_real_tokens(data, growth_factor=0.4):
    """Model real tokens considering Bug1 loss.
    
    For each task:
    - Planner calls: 1 API call each (no tools), no loss
    - Worker calls: K API calls (1 initial + tool call rounds), Bug1 loss  
    
    K = 1 + number_of_tool_roundtrips
    For parallel tool calls in same round: counted as 1 roundtrip
    Estimate: K = 1 + ceil(tools_per_worker / parallel_factor)
    parallel_factor ≈ 2 (some tools get batched)
    """
    total_real_inp = 0
    total_real_out = 0
    
    for d in data:
        ac = d.get('agent_calls', {})
        total_tools = d.get('total_tool_calls', 0)
        rec_inp = d.get('input_tokens', 0)
        rec_out = d.get('output_tokens', 0)

        planner_count = sum(v for k, v in ac.items() if 'planner' in k.lower())
        worker_count = sum(v for k, v in ac.items() if 'planner' not in k.lower())
        total_ac = planner_count + worker_count

        if total_ac == 0:
            total_real_inp += rec_inp
            total_real_out += rec_out
            continue

        # Split recorded input between planner and workers
        # Planner gets a proportional share
        planner_share = planner_count / total_ac
        worker_share = worker_count / total_ac

        planner_rec_inp = rec_inp * planner_share
        worker_rec_inp = rec_inp * worker_share
        planner_rec_out = rec_out * planner_share
        worker_rec_out = rec_out * worker_share

        # Planner: no correction needed (1 API call, no tools)
        real_planner_inp = planner_rec_inp
        real_planner_out = planner_rec_out

        # Worker: correct for Bug1
        if worker_count > 0 and total_tools > 0:
            avg_wk_tools = total_tools / worker_count
            # K = number of API calls per worker run
            # DeepSeek supports parallel tool calls, 
            # but typically 1-2 tools per round
            parallel_factor = 2.0
            K = 1 + math.ceil(avg_wk_tools / parallel_factor)

            if K > 1:
                g = growth_factor
                # Input correction: total_real / recorded_last
                inp_numerator = K + g * K * (K - 1) / 2
                inp_denominator = 1 + g * (K - 1)
                inp_correction = inp_numerator / inp_denominator

                # Output correction: K calls total, only 1 recorded
                out_correction = K
            else:
                inp_correction = 1.0
                out_correction = 1.0

            real_worker_inp = worker_rec_inp * inp_correction
            real_worker_out = worker_rec_out * out_correction
        else:
            real_worker_inp = worker_rec_inp
            real_worker_out = worker_rec_out

        total_real_inp += real_planner_inp + real_worker_inp
        total_real_out += real_planner_out + real_worker_out

    return total_real_inp, total_real_out


def calc_cost(inp_tokens, out_tokens, tier2_frac=0.05):
    """Calculate cost with tiered pricing"""
    t1_frac = 1 - tier2_frac
    eff_inp_price = t1_frac * T1_INP + tier2_frac * T2_INP
    eff_out_price = t1_frac * T1_OUT + tier2_frac * T2_OUT
    return inp_tokens / 1e6 * eff_inp_price + out_tokens / 1e6 * eff_out_price


# Try different growth factors
print("\nSensitivity to growth_factor (tool result size / base context):")
print("-" * 90)
print(f"{'growth':<8} {'GAIA_BL':>10} {'GAIA_DF':>10} {'TAMAS_BL':>12} {'TAMAS_DF':>12} {'Total':>12} {'Cost':>10}")
print("-" * 90)

for gf in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    total_cost = 0
    row = [f"{gf:.1f}"]
    for label, path in benchmarks:
        data = all_results[label]['data']
        ri, ro = model_real_tokens(data, growth_factor=gf)
        cost = calc_cost(ri, ro)
        total_cost += cost
        row.append(f"{(ri+ro):,.0f}")
    row.append(f"{total_cost:.1f}")
    print(f"{row[0]:<8} {row[1]:>10} {row[2]:>10} {row[3]:>12} {row[4]:>12} {row[5]:>12}")


# ============================================================
# Part 3: 更精确的 K 值分析
# ============================================================
print("\n")
print("=" * 80)
print("Part 3: 分析每个任务的 K 值 (API calls per agent.run)")  
print("=" * 80)

# The FunctionInvocationLayer calls the LLM, checks for tool_calls,
# executes them, and loops. So K = 1 + number of rounds with tool_calls.
# We don't have this directly, but we can model from tool_calls per worker.

# Actually, we should look at it differently:
# The 'input_tokens' we recorded is from the LAST API call,
# which has the LARGEST input (accumulated context).
# For each worker call with T tools and K API calls:
#   Recorded input ≈ base + T * avg_tool_result_size
#   Real total input ≈ K * base + sum of accumulated tool results
#   = base*(K) + avg_tool_result*(0 + 1 + 2 + ... + (K-1)) per-round tools
# This is complex. Let's use a simpler direct model.

# Better approach: each API call's input ≈ recorded_input * ratio_table[k]
# where ratio_table[k] is the fraction of total context at step k.
# For K calls: step k has context proportion p_k = (base + k*delta) / (base + (K-1)*delta)
# Total input = sum_{k=0}^{K-1} (base + k*delta) = K*base + delta*K*(K-1)/2
# Recorded (last) = base + (K-1)*delta
# Therefore: true_total / recorded_last = [K*base + delta*K*(K-1)/2] / [base + (K-1)*delta]

# Let r = delta/base (growth rate per round)
# true/rec = K * [1 + r*(K-1)/2] / [1 + r*(K-1)]

# For GAIA, examine specific tasks:
print("\nGAIA Baseline - per task K estimation:")
print(f"{'task_id':<40} {'agents':>7} {'tools':>6} {'K_est':>6} {'rec_inp':>9} {'ratio':>7} {'real_inp':>10}")
gaia_data = all_results['GAIA Baseline']['data']
total_modeled = 0
total_recorded = 0
for d in sorted(gaia_data, key=lambda x: -x.get('total_tool_calls', 0))[:20]:
    tid = d['task_id'][:38]
    ac = d.get('agent_calls', {})
    planner_c = sum(v for k, v in ac.items() if 'planner' in k.lower())
    worker_c = sum(v for k, v in ac.items() if 'planner' not in k.lower())
    tools = d.get('total_tool_calls', 0)
    rec_inp = d.get('input_tokens', 0)
    
    if worker_c > 0:
        avg_t = tools / worker_c
        K = 1 + math.ceil(avg_t / 2)
        r = 0.4
        if K > 1:
            ratio = K * (1 + r * (K - 1) / 2) / (1 + r * (K - 1))
        else:
            ratio = 1.0
    else:
        K = 1
        ratio = 1.0
    
    # Worker share of input
    total_ac = planner_c + worker_c
    if total_ac > 0:
        worker_share = worker_c / total_ac
    else:
        worker_share = 0.5
    
    worker_inp = rec_inp * worker_share
    real_worker_inp = worker_inp * ratio
    planner_inp = rec_inp * (1 - worker_share)
    real_total_inp = planner_inp + real_worker_inp
    
    total_modeled += real_total_inp
    total_recorded += rec_inp
    print(f"{tid:<40} {total_ac:>7} {tools:>6} {K:>6} {rec_inp:>9,} {ratio:>7.2f} {real_total_inp:>10,.0f}")

print(f"\nTop 20 tasks: recorded={total_recorded:,} modeled={total_modeled:,.0f} ratio={total_modeled/total_recorded:.2f}x")


# ============================================================
# Part 4: 考虑 _select_worker_llm 的额外调用
# ============================================================
print("\n")
print("=" * 80)
print("Part 4: LLM Selector + 重试等额外 LLM 调用")
print("=" * 80)
print("""
额外、未被 agent framework 追踪的 LLM 调用:
1. _select_worker_llm: Plan解析失败时的备选LLM选择 (偶发)
2. browse_webpage: 每次调用 vision LLM (Doubao-Seed-1.8)
3. analyze_image: vision LLM
4. analyze_youtube_video: vision LLM  
5. 缓存命中: 火山引擎的缓存策略可能降低实际费用

Doubao-Seed-1.8 定价: input=0.8/M, output=8.0/M (<=32k tier1)
""")


# ============================================================
# Part 5: 最终估算
# ============================================================
print("=" * 80)
print("Part 5: 最终费用估算 (growth_factor sweep)")
print("=" * 80)
print()

# Check: what growth factor matches 150 RMB for total benchmark cost?
best_gf = None
for gf_x10 in range(1, 30):
    gf = gf_x10 / 10.0
    total_cost = 0
    for label, path in benchmarks:
        data = all_results[label]['data']
        ri, ro = model_real_tokens(data, growth_factor=gf)
        cost = calc_cost(ri, ro, tier2_frac=0.05)
        total_cost += cost
    if total_cost >= 150 and best_gf is None:
        best_gf = gf
        print(f"** growth_factor={gf:.1f} gives total benchmark cost={total_cost:.1f} RMB (first to exceed 150)")

# Now do final table with the best gf
print()
print(f"Using growth_factor = 0.4 (conservative default):")
print("-" * 80)
total_bench_cost = 0
for label, path in benchmarks:
    data = all_results[label]['data']
    ri, ro = model_real_tokens(data, growth_factor=0.4)
    cost = calc_cost(ri, ro, tier2_frac=0.05)
    total_bench_cost += cost
    rec = all_results[label]
    corr = (ri + ro) / (rec['rec_inp'] + rec['rec_out'])
    print(f"  {label:20s}: real_inp={ri:>12,.0f} real_out={ro:>10,.0f} cost={cost:>6.1f} RMB (correction={corr:.2f}x)")
print(f"  {'TOTAL':20s}: cost={total_bench_cost:.1f} RMB")
print()

# Check: what if Bug1 is worse than modeled? 
# Maybe parallel_factor is 1 (no parallel tool calls) → K = 1 + tools_per_worker
print("If DeepSeek doesn't do parallel tool calls (parallel_factor=1):")
print("-" * 80)

def model_real_tokens_pf1(data, growth_factor=0.4):
    total_real_inp = 0
    total_real_out = 0
    for d in data:
        ac = d.get('agent_calls', {})
        total_tools = d.get('total_tool_calls', 0)
        rec_inp = d.get('input_tokens', 0)
        rec_out = d.get('output_tokens', 0)
        planner_count = sum(v for k, v in ac.items() if 'planner' in k.lower())
        worker_count = sum(v for k, v in ac.items() if 'planner' not in k.lower())
        total_ac = planner_count + worker_count
        if total_ac == 0:
            total_real_inp += rec_inp
            total_real_out += rec_out
            continue
        planner_share = planner_count / total_ac
        worker_share = worker_count / total_ac
        planner_rec_inp = rec_inp * planner_share
        worker_rec_inp = rec_inp * worker_share
        planner_rec_out = rec_out * planner_share
        worker_rec_out = rec_out * worker_share
        real_planner_inp = planner_rec_inp
        real_planner_out = planner_rec_out
        if worker_count > 0 and total_tools > 0:
            avg_wk_tools = total_tools / worker_count
            # NO parallel: K = 1 + tools_per_worker
            K = 1 + round(avg_wk_tools)
            g = growth_factor
            if K > 1:
                inp_correction = K * (1 + g * (K - 1) / 2) / (1 + g * (K - 1))
                out_correction = K
            else:
                inp_correction = 1.0
                out_correction = 1.0
            real_worker_inp = worker_rec_inp * inp_correction
            real_worker_out = worker_rec_out * out_correction
        else:
            real_worker_inp = worker_rec_inp
            real_worker_out = worker_rec_out
        total_real_inp += real_planner_inp + real_worker_inp
        total_real_out += real_planner_out + real_worker_out
    return total_real_inp, total_real_out

total_bench_cost_pf1 = 0
for label, path in benchmarks:
    data = all_results[label]['data']
    ri, ro = model_real_tokens_pf1(data, growth_factor=0.4)
    cost = calc_cost(ri, ro, tier2_frac=0.10)  # more likely to hit tier2 with larger contexts
    total_bench_cost_pf1 += cost
    rec = all_results[label]
    corr = (ri + ro) / (rec['rec_inp'] + rec['rec_out'])
    print(f"  {label:20s}: real_inp={ri:>12,.0f} real_out={ro:>10,.0f} cost={cost:>6.1f} RMB (correction={corr:.2f}x)")
print(f"  {'TOTAL':20s}: cost={total_bench_cost_pf1:.1f} RMB")
print()

# The real question: what's the actual parallel factor?
# Let's check from the 5-task calibration data
print("=" * 80)
print("Part 6: 用5任务校准数据反推模型参数")
print("=" * 80)

cal_data = load_jsonl('token_fix_test_results.jsonl')
bm_data = {d['task_id']: d for d in load_jsonl('benchmark_gaia_baseline.jsonl')}

print(f"\n{'task_id':<25} {'old_tok':>8} {'new_tok':>8} {'ratio':>7} {'agents':>7} {'tools':>6} {'K_model':>8}")
for cd in cal_data:
    tid = cd['task_id']
    old_total = cd['old_total_tokens']
    new_total = cd['new_total_tokens']
    ratio = new_total / old_total if old_total > 0 else 0
    
    bm = bm_data.get(tid, {})
    agents = bm.get('total_agent_calls', 0)
    tools = bm.get('total_tool_calls', 0)
    
    ac = bm.get('agent_calls', {})
    planner_c = sum(v for k, v in ac.items() if 'planner' in k.lower())
    worker_c = sum(v for k, v in ac.items() if 'planner' not in k.lower())
    avg_t = tools / worker_c if worker_c > 0 else 0
    
    print(f"{tid[:24]:<25} {old_total:>8,} {new_total:>8,} {ratio:>7.2f} {agents:>7} {tools:>6} {avg_t:>8.1f}")

print()
print("IMPORTANT: ratio varies 0.93x to 2.44x because LLM takes different paths each run!")
print("The 1.60x average is NOT a reliable correction factor.")
print()

# Final: try to find what scenario totals 150 RMB
print("=" * 80)
print("Part 7: 反推 - 什么参数能达到 150 RMB benchmark 费用?")
print("=" * 80)

for pf_label, pf in [("parallel=2", 2), ("parallel=1.5", 1.5), ("parallel=1", 1)]:
    for gf in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        total = 0
        for label, path in benchmarks:
            data = all_results[label]['data']
            ti, to = 0, 0
            for d in data:
                ac = d.get('agent_calls', {})
                tt = d.get('total_tool_calls', 0)
                ri = d.get('input_tokens', 0)
                ro = d.get('output_tokens', 0)
                pc = sum(v for k, v in ac.items() if 'planner' in k.lower())
                wc = sum(v for k, v in ac.items() if 'planner' not in k.lower())
                ta = pc + wc
                if ta == 0:
                    ti += ri; to += ro; continue
                ps = pc / ta; ws = wc / ta
                pi = ri * ps; wi = ri * ws
                po = ro * ps; wo = ro * ws
                if wc > 0 and tt > 0:
                    at = tt / wc
                    K = 1 + math.ceil(at / pf)
                    g = gf
                    if K > 1:
                        ic = K * (1 + g*(K-1)/2) / (1 + g*(K-1))
                        oc = K
                    else:
                        ic = 1; oc = 1
                    wi *= ic; wo *= oc
                ti += pi + wi; to += po + wo
            total += calc_cost(ti, to, tier2_frac=0.10)
        if total >= 145:
            print(f"  {pf_label}, growth={gf:.1f}: {total:.1f} RMB  ✓")
        elif total >= 100:
            print(f"  {pf_label}, growth={gf:.1f}: {total:.1f} RMB")
