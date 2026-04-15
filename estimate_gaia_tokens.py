"""
精确估算 GAIA baseline 和 defended 的修正后 per-task token。

方法: 用 5-task API 拦截的精确数据 + benchmark JSONL 的旧(buggy)数据,
      推导 per-task 修正因子, 再应用到全部 53 个任务。
"""
import json
import statistics

# 1. 读取 5-task 拦截数据 (post-fix, verified 1.000x)
with open('token_intercept_results.json', 'r', encoding='utf-8') as f:
    intercept = json.load(f)

# 2. 读取 benchmark JSONL
def load_benchmark(path):
    tasks = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip())
            tasks[d['task_id']] = d
    return tasks

base_tasks = load_benchmark('benchmark_gaia_baseline.jsonl')
def_tasks = load_benchmark('benchmark_gaia_defended.jsonl')

# 3. 精确匹配: 拦截过的 task vs 旧 benchmark 记录
print("=" * 70)
print("Per-task 修正因子 (5 tasks with API intercept)")
print("=" * 70)
ratios = []
for r in intercept:
    tid = r['task_id']
    new_inp = r['api_input']
    new_out = r['api_output']
    new_total = new_inp + new_out
    
    if tid in base_tasks:
        old = base_tasks[tid]
        old_inp = old['input_tokens']
        old_out = old['output_tokens']
        old_total = old_inp + old_out
        
        ratio_inp = new_inp / old_inp if old_inp > 0 else 0
        ratio_out = new_out / old_out if old_out > 0 else 0
        ratio_total = new_total / old_total if old_total > 0 else 0
        
        n_tools = old.get('total_tool_calls', 0)
        n_agents = old.get('total_agent_calls', 0)
        tpa = n_tools / n_agents if n_agents > 0 else 0
        
        ratios.append({
            'task_id': tid[:12],
            'old_total': old_total,
            'new_total': new_total,
            'ratio': ratio_total,
            'ratio_inp': ratio_inp,
            'ratio_out': ratio_out,
            'tools': n_tools,
            'agents': n_agents,
            'tools_per_agent': tpa,
        })
        print(f"  {tid[:12]}  old={old_total:>6,}  new(fix)={new_total:>6,}  "
              f"ratio={ratio_total:.2f}x  tools={n_tools}  agents={n_agents}  t/a={tpa:.1f}")
    else:
        print(f"  {tid[:12]}  NOT IN BASELINE (different task set)")

print()

if not ratios:
    print("NO OVERLAPPING TASKS! 需要用不同方法估算。")
    print()
    
    # 没有重叠 → 用修复后实测的平均值 + 旧数据的比例关系
    # 方法: 已知修复后 GAIA 平均值，用 baseline/defended 在旧数据中的比例来推算
    
    # 5-task 平均 (所有都是 baseline 模式运行)
    avg_5task = sum(r['api_input'] + r['api_output'] for r in intercept) / len(intercept)
    print(f"5-task 拦截平均 (post-fix): {avg_5task:,.0f} tok/task")
    
    # 但这 5 个任务可能不代表全部 53 个
    # 更好：用全部 53 任务的旧数据作为权重
    baseline_total_old = sum(t['input_tokens'] + t['output_tokens'] for t in base_tasks.values())
    defended_total_old = sum(t['input_tokens'] + t['output_tokens'] for t in def_tasks.values())
    avg_base_old = baseline_total_old / len(base_tasks)
    avg_def_old = defended_total_old / len(def_tasks)
    
    print(f"Baseline 旧平均: {avg_base_old:,.0f}")
    print(f"Defended 旧平均: {avg_def_old:,.0f}")
    print(f"Defended/Baseline 旧比例: {avg_def_old / avg_base_old:.3f}")
    
    # ---- 方法 A: 用 10-task + 5-task 合并平均 ----
    # 10-task 结果: avg=22,000 (来自 token_ratio_test_results.jsonl)
    with open('token_ratio_test_results.jsonl', 'r', encoding='utf-8') as f:
        ratio_results = [json.loads(line) for line in f if line.strip()]
    
    avg_10task = statistics.mean(r['input_tokens'] + r['output_tokens'] for r in ratio_results)
    n_10 = len(ratio_results)
    
    # 合并 15 tasks
    all_measured = []
    for r in intercept:
        all_measured.append(r['api_input'] + r['api_output'])
    for r in ratio_results:
        all_measured.append(r['input_tokens'] + r['output_tokens'])
    
    avg_all = statistics.mean(all_measured)
    med_all = statistics.median(all_measured)
    std_all = statistics.stdev(all_measured) if len(all_measured) > 1 else 0
    
    print(f"\n10-task 平均: {avg_10task:,.0f} ({n_10} tasks)")
    print(f"5-task 拦截平均: {avg_5task:,.0f}")
    print(f"合并 {len(all_measured)} tasks: avg={avg_all:,.0f}  median={med_all:,.0f}  std={std_all:,.0f}")
    
    # ---- 方法 B: 用修正倍率 ----
    # 已知旧数据的 tools_per_agent 分布，校准修正倍率
    
    # 从 benchmark 数据中获取 per-task tools/agent
    base_tpa_list = []
    for t in base_tasks.values():
        na = t.get('total_agent_calls', 0)
        nt = t.get('total_tool_calls', 0)
        if na > 0:
            base_tpa_list.append(nt / na)
    
    print(f"\nBaseline tools/agent: avg={statistics.mean(base_tpa_list):.2f}  "
          f"med={statistics.median(base_tpa_list):.1f}")
    
    print()
    print("=" * 70)
    print("最终估算")
    print("=" * 70)
    
    # LLM 的非确定性导致每次运行结果不同
    # 但修正倍率是系统性的: 取决于 tool loop 中 API 调用次数
    # 最可靠的方法: 用大样本修复后测量的平均值
    
    # 假定修复后的平均 token 与旧数据呈固定比例关系
    # 从 5-task 拦截数据估算整体修正倍率
    # 但由于任务不重叠,无法直接计算
    
    # 最保守方法: 用全部 15 个测量值的平均作为 baseline
    # 然后用旧 defended/baseline 比例推算 defended
    
    baseline_avg_corrected = avg_all  # 15-task measured average
    ratio_def_base = avg_def_old / avg_base_old
    defended_avg_corrected = baseline_avg_corrected * ratio_def_base
    
    print(f"\n方法 (15-task 合并平均 + 比例推算):")
    print(f"  Baseline avg: {baseline_avg_corrected:,.0f} tokens/task")
    print(f"  Defended avg:  {defended_avg_corrected:,.0f} tokens/task")
    print(f"  (Defended = Baseline × {ratio_def_base:.3f})")
    
    # 费用
    for label, avg_tok in [("Baseline", baseline_avg_corrected), ("Defended", defended_avg_corrected)]:
        inp = avg_tok * 0.975
        out = avg_tok * 0.025 
        cost = inp / 1e6 * 1.0 + out / 1e6 * 1.5
        total_cost = cost * 53
        print(f"  {label}: {avg_tok:,.0f} tok/task, ≈{cost:.4f} 元/task, 53任务={total_cost:.2f} 元")

else:
    # 有重叠！
    avg_ratio = statistics.mean(r['ratio'] for r in ratios)
    med_ratio = statistics.median(r['ratio'] for r in ratios)
    print(f"平均修正倍率: {avg_ratio:.2f}x  中位: {med_ratio:.2f}x")
    
    # 对每个 benchmark 任务应用修正
    base_total_corrected = sum((t['input_tokens'] + t['output_tokens']) * avg_ratio for t in base_tasks.values())
    def_total_corrected = sum((t['input_tokens'] + t['output_tokens']) * avg_ratio for t in def_tasks.values())
    
    base_avg = base_total_corrected / len(base_tasks)
    def_avg = def_total_corrected / len(def_tasks)
    
    print(f"\n估算结果:")
    print(f"  Baseline avg: {base_avg:,.0f} tokens/task")
    print(f"  Defended avg:  {def_avg:,.0f} tokens/task")
