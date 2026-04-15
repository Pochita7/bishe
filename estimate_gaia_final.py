"""
最终估算: GAIA baseline & defended 平均 token/task

已验证: 修复后的 token 计算 = API 真实消耗 (5-task intercept: 1.000x)
因此修复后的测量值就是真实值。

数据源:
  - 5-task API 拦截 (精确验证)
  - 10-task post-fix 实测
  合计 15 个 baseline 模式测量

问题: 没有 defended 模式的实测数据
解决: GAIA 是干净任务, Guardian 只增加少量前置检查, 
      可用旧数据中 defended/baseline 比例估算
"""
import json
import statistics

# ============================================================
# 1. 汇总所有 post-fix 测量
# ============================================================

# 5-task API 拦截数据
with open('token_intercept_results.json', 'r') as f:
    intercept = json.load(f)
intercept_totals = [(r['api_input'], r['api_output']) for r in intercept]

# 10-task post-fix 数据
with open('token_ratio_test_results.jsonl', 'r') as f:
    ratio_data = [json.loads(line) for line in f if line.strip()]
ratio_totals = [(r['input_tokens'], r['output_tokens']) for r in ratio_data]

# 合并 (检查有无重复 task_id)
intercept_ids = {r['task_id'] for r in intercept}
ratio_ids = {r['task_id'] for r in ratio_data}
overlap = intercept_ids & ratio_ids
if overlap:
    print(f"WARNING: {len(overlap)} overlapping task_ids, deduplicating")
    # 保留 intercept 的 (更精确)
    ratio_totals_dedup = [(r['input_tokens'], r['output_tokens']) 
                          for r in ratio_data if r['task_id'] not in overlap]
    all_totals = intercept_totals + ratio_totals_dedup
else:
    all_totals = intercept_totals + ratio_totals

n = len(all_totals)
all_inp = [t[0] for t in all_totals]
all_out = [t[1] for t in all_totals]
all_total = [t[0] + t[1] for t in all_totals]

print("=" * 70)
print(f"Post-fix GAIA 测量汇总 ({n} tasks, baseline mode)")
print("=" * 70)

for vals, label in [(all_inp, "Input"), (all_out, "Output"), (all_total, "Total")]:
    avg = statistics.mean(vals)
    med = statistics.median(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    mn = min(vals)
    mx = max(vals)
    print(f"  {label:8s}: avg={avg:>8,.0f}  median={med:>7,.0f}  std={std:>7,.0f}  "
          f"range=[{mn:,}-{mx:,}]")

print()

# ============================================================
# 2. 旧 benchmark 数据
# ============================================================
def load_benchmark(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

base = load_benchmark('benchmark_gaia_baseline.jsonl')
defd = load_benchmark('benchmark_gaia_defended.jsonl')

base_avg_inp = statistics.mean(r['input_tokens'] for r in base)
base_avg_out = statistics.mean(r['output_tokens'] for r in base)
def_avg_inp = statistics.mean(r['input_tokens'] for r in defd)
def_avg_out = statistics.mean(r['output_tokens'] for r in defd)

base_avg = base_avg_inp + base_avg_out
def_avg = def_avg_inp + def_avg_out

print("旧 Benchmark (buggy) 平均:")
print(f"  Baseline: inp={base_avg_inp:>8,.0f}  out={base_avg_out:>5,.0f}  total={base_avg:>8,.0f}")
print(f"  Defended:  inp={def_avg_inp:>8,.0f}  out={def_avg_out:>5,.0f}  total={def_avg:>8,.0f}")
print(f"  Defended/Baseline = {def_avg/base_avg:.4f}")
print()

# ============================================================
# 3. 最终估算
# ============================================================
print("=" * 70)
print("最终估算")
print("=" * 70)
print()

# 方法说明:
# 由于 LLM 非确定性, 同一任务每次消耗差异很大(修正因子 0.63x~2.55x)
# 因此 "旧数据 × 修正因子" 不可靠
# 最稳健: 用大样本 post-fix 实测的统计量
#
# Baseline: 直接用 15-task 实测的 mean
# Defended: GAIA 是干净任务, Guardian 开启后:
#   - 增加 system_message 中的安全指令 (~200 tokens)
#   - 每个 Worker 输入增加安全提示 (~100 tokens)
#   - 但不会触发拦截(任务是干净的)
#   从旧数据看 defended 比 baseline 反而少了约 10% (因为随机性)
#   最合理: defended ≈ baseline (Guardian 对干净任务的开销 < 5%)

# 使用 median 比 mean 更稳健(不受极端值影响)
baseline_est = statistics.median(all_total)
baseline_est_mean = statistics.mean(all_total)

# 对于 defended, 用旧数据比例
ratio_def_base = def_avg / base_avg

print("方案 A (推荐, median):")
print(f"  Baseline avg token: {baseline_est:>8,.0f} tokens/task")
defended_est = baseline_est * ratio_def_base
print(f"  Defended avg token:  {defended_est:>8,.0f} tokens/task")
print(f"  (基于 {n} tasks median, defended 用旧数据比例 {ratio_def_base:.3f})")
print()

print("方案 B (mean):")
print(f"  Baseline avg token: {baseline_est_mean:>8,.0f} tokens/task")
defended_est_mean = baseline_est_mean * ratio_def_base
print(f"  Defended avg token:  {defended_est_mean:>8,.0f} tokens/task")
print()

# 不如直接说: 用 mean 吧, 因为总量 = mean * N
# 但 mean 受极值影响大

# 看看去掉最大最小值的 trimmed mean
if n >= 5:
    sorted_vals = sorted(all_total)
    # 去掉最大最小各 10%
    trim = max(1, n // 10)
    trimmed = sorted_vals[trim:-trim] if trim > 0 else sorted_vals
    trimmed_mean = statistics.mean(trimmed)
    
    print(f"方案 C (10% trimmed mean, 去极值):")
    print(f"  Baseline avg token: {trimmed_mean:>8,.0f} tokens/task")
    defended_trimmed = trimmed_mean * ratio_def_base
    print(f"  Defended avg token:  {defended_trimmed:>8,.0f} tokens/task")
    print(f"  (去掉最大/最小各 {trim} 个, 剩余 {len(trimmed)} tasks)")
    print()

# ============================================================
# 4. 最终推荐
# ============================================================
print("=" * 70)
print("最终推荐 (用于论文)")
print("=" * 70)
print()

# 使用 trimmed mean (最稳健)
if n >= 5:
    final_base = trimmed_mean
else:
    final_base = baseline_est_mean

final_def = final_base * ratio_def_base

# 输入输出比例 (从实测数据)
inp_ratio = statistics.mean(all_inp) / (statistics.mean(all_inp) + statistics.mean(all_out))
out_ratio = 1 - inp_ratio

final_base_inp = final_base * inp_ratio
final_base_out = final_base * out_ratio
final_def_inp = final_def * inp_ratio
final_def_out = final_def * out_ratio

print(f"  GAIA Baseline: {final_base:>8,.0f} tokens/task")
print(f"    (input: {final_base_inp:>8,.0f}  output: {final_base_out:>8,.0f})")
print()
print(f"  GAIA Defended:  {final_def:>8,.0f} tokens/task")
print(f"    (input: {final_def_inp:>8,.0f}  output: {final_def_out:>8,.0f})")
print()

# 费用
base_cost = final_base_inp / 1e6 * 1.0 + final_base_out / 1e6 * 1.5
def_cost = final_def_inp / 1e6 * 1.0 + final_def_out / 1e6 * 1.5
print(f"  Baseline 费用: {base_cost:.4f} 元/task × 53 = {base_cost * 53:.2f} 元")
print(f"  Defended 费用:  {def_cost:.4f} 元/task × 53 = {def_cost * 53:.2f} 元")
print(f"  GAIA 总计: {(base_cost + def_cost) * 53:.2f} 元")
print()

print("注: 数据来源")
print(f"  - {n} 个 GAIA 任务的修复后直接测量")
print(f"  - 修复准确性已通过 API 拦截验证 (ratio=1.000x)")
print(f"  - Defended 比例来自旧 benchmark 数据 ({ratio_def_base:.3f})")
print(f"  - 使用 10% trimmed mean 消除极端任务影响")

# ============================================================
# 5. 分布可视化 (文本)
# ============================================================
print()
print("=" * 70)
print("Token 分布 (15 tasks)")
print("=" * 70)
sorted_all = sorted(zip(all_total, range(len(all_total))))
for val, idx in sorted_all:
    bar = '█' * (val // 2000)
    print(f"  Task {idx+1:>2}: {val:>7,} |{bar}")
