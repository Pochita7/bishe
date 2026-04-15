"""分析火山引擎真实账单数据"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 火山引擎真实账单 (202603)
# DeepSeek-V3.2
ds_inp_32k  = 91421.580  # 千tokens
ds_out_32k  = 2339.732   # 千tokens
ds_inp_128k = 33.469     # 千tokens
ds_out_128k = 0.265      # 千tokens

ds_cost_inp_32k  = 181.83  # 元
ds_cost_out_32k  = 6.41    # 元
ds_cost_inp_128k = 0.13    # 元
ds_cost_out_128k = 0.00    # 元

# Doubao-Seed-1.8
db_inp = 46.788  # 千tokens
db_out = 12.799  # 千tokens
db_cost_inp = 0.02
db_cost_out = 0.07

print("=" * 70)
print("火山引擎真实账单 (2026年3月)")
print("=" * 70)
print()

ds_cost = ds_cost_inp_32k + ds_cost_out_32k + ds_cost_inp_128k + ds_cost_out_128k
db_cost = db_cost_inp + db_cost_out
total_cost = ds_cost + db_cost

print("DeepSeek-V3.2:")
print(f"  32k以内 输入: {ds_inp_32k*1000:>14,.0f} tok  {ds_cost_inp_32k:>8.2f} 元")
print(f"  32k以内 输出: {ds_out_32k*1000:>14,.0f} tok  {ds_cost_out_32k:>8.2f} 元")
print(f"  128k以内输入: {ds_inp_128k*1000:>14,.0f} tok  {ds_cost_inp_128k:>8.2f} 元")
print(f"  128k以内输出: {ds_out_128k*1000:>14,.0f} tok  {ds_cost_out_128k:>8.2f} 元")
print(f"  小计: {ds_cost:.2f} 元")
print()

print("Doubao-Seed-1.8 (视觉):")
print(f"  输入: {db_inp*1000:>10,.0f} tok  {db_cost_inp:.2f} 元")
print(f"  输出: {db_out*1000:>10,.0f} tok  {db_cost_out:.2f} 元")
print(f"  小计: {db_cost:.2f} 元")
print()

print(f"总费用: {total_cost:.2f} 元")
print()

# Token 汇总
ds_total_inp = (ds_inp_32k + ds_inp_128k) * 1000
ds_total_out = (ds_out_32k + ds_out_128k) * 1000
ds_total = ds_total_inp + ds_total_out

print("=" * 70)
print("Token 汇总")
print("=" * 70)
print(f"DS-V3.2 总输入: {ds_total_inp:>14,.0f} tokens ({ds_total_inp/1e6:.1f}M)")
print(f"DS-V3.2 总输出: {ds_total_out:>14,.0f} tokens ({ds_total_out/1e6:.1f}M)")
print(f"DS-V3.2 总计:   {ds_total:>14,.0f} tokens ({ds_total/1e6:.1f}M)")
print(f"输入占比: {ds_total_inp/ds_total*100:.1f}%")
print(f"输入费用占比: {(ds_cost_inp_32k+ds_cost_inp_128k)/ds_cost*100:.1f}%")
print()

# 对比 benchmark 记录
print("=" * 70)
print("与 4组 benchmark JSONL 记录对比")
print("=" * 70)
rec_inp = 18_219_919
rec_out = 696_100
rec_total = rec_inp + rec_out

print(f"Benchmark记录: inp={rec_inp:>12,}  out={rec_out:>8,}  total={rec_total:>12,}")
print(f"账单实际:      inp={ds_total_inp:>12,.0f}  out={ds_total_out:>12,.0f}  total={ds_total:>12,.0f}")
print()

# 账单是所有调用(benchmark + 开发调试)
# 输入修正 = 账单输入 / 记录输入
inp_ratio = ds_total_inp / rec_inp
out_ratio = ds_total_out / rec_out
total_ratio = ds_total / rec_total

print(f"如果benchmark占全部调用:")
print(f"  输入修正: {inp_ratio:.2f}x")
print(f"  输出修正: {out_ratio:.2f}x")
print(f"  总体修正: {total_ratio:.2f}x")
print()

# GAIA 实测值
gaia_measured_total = 22000
gaia_measured_inp = 20027
gaia_measured_out = 1972

print("=" * 70)
print("GAIA 每任务真实 token 消耗 (10任务实测)")
print("=" * 70)
print(f"input:  {gaia_measured_inp:,} tokens/task")
print(f"output: {gaia_measured_out:,} tokens/task")
print(f"total:  {gaia_measured_total:,} tokens/task")
print()

gaia_cost = gaia_measured_inp / 1e6 * 1.0 + gaia_measured_out / 1e6 * 1.5
print(f"GAIA 每任务费用: {gaia_cost:.4f} 元")
print(f"53 GAIA 任务预估: {gaia_cost * 53:.2f} 元")
print()

# 用实测倍率 1.96x 估算 benchmark 占比
est_bench_tokens = rec_total * 1.96
est_bench_cost = rec_inp * 1.96 / 1e6 * 1.0 + rec_out * 1.96 / 1e6 * 1.5
print("=" * 70)
print("Benchmark vs 开发调试 拆分")
print("=" * 70)
print(f"用1.96x估算benchmark token: {est_bench_tokens:,.0f}")
print(f"账单总token:                {ds_total:,.0f}")
print(f"差额(开发调试):              {ds_total - est_bench_tokens:,.0f}")
print(f"benchmark占比:               {est_bench_tokens/ds_total*100:.0f}%")
print()

# 更准确: 反推 benchmark 真实修正倍率
# 假设开发调试占 30% 的账单 (保守)
for dev_pct in [0, 10, 20, 30, 40]:
    bench_tokens = ds_total * (100 - dev_pct) / 100
    bench_ratio = bench_tokens / rec_total
    bench_cost = ds_cost * (100 - dev_pct) / 100
    print(f"  假设开发占{dev_pct}%: benchmark={bench_tokens:,.0f}tok ratio={bench_ratio:.2f}x cost={bench_cost:.1f}元")

print()
print("=" * 70)
print("最终结论")
print("=" * 70)
print()
print(f"1. 火山引擎账单总费用: {total_cost:.2f} 元")
print(f"   - DeepSeek-V3.2: {ds_cost:.2f} 元 (占 {ds_cost/total_cost*100:.0f}%)")
print(f"   - Doubao-Seed-1.8: {db_cost:.2f} 元 (占 {db_cost/total_cost*100:.1f}%)")
print()
print(f"2. DeepSeek-V3.2 总消耗: {ds_total/1e6:.1f}M tokens")
print(f"   - 输入: {ds_total_inp/1e6:.1f}M ({ds_total_inp/ds_total*100:.1f}%)")
print(f"   - 输出: {ds_total_out/1e6:.1f}M ({ds_total_out/ds_total*100:.1f}%)")
print(f"   - 99.96% 在 32k tier1 定价区间")
print()
print(f"3. GAIA 每任务 (修复后实测):")
print(f"   - 平均消耗: {gaia_measured_total:,} tokens/task")
print(f"   - 平均费用: {gaia_cost:.4f} 元/task")
print()
print(f"4. 账单 vs 记录修正倍率: {total_ratio:.2f}x (上限, 包含开发调试)")
print(f"   GAIA 实测修正倍率: 1.96x")
