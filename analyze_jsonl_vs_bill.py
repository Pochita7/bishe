"""分析所有 JSONL 文件的 token 记录，对比账单。"""
import json, os

files = [
    'gaia_results.jsonl',
    'gaia_results_af_test.jsonl',
    'gaia_results_af_test2.jsonl',
    'gaia_results_af_test3.jsonl',
    'gaia_v3_test.jsonl',
    'gaia_results_v2.jsonl',
    'gaia_test_smoke.jsonl',
]

total_tasks = 0
total_inp = 0
total_out = 0

for f in sorted(set(files)):
    if not os.path.exists(f):
        continue
    tasks = 0
    inp = 0
    outp = 0
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            tasks += 1
            m = d.get('metrics', {})
            inp += m.get('input_tokens', 0)
            outp += m.get('output_tokens', 0)

    total_tasks += tasks
    total_inp += inp
    total_out += outp
    print(f"{f:40s}  tasks={tasks:>4}  inp={inp:>10,}  out={outp:>8,}  total={inp+outp:>10,}")

sep = '=' * 80
print(f"\n{sep}")
print("Total across all JSONL files:")
print(f"  tasks:  {total_tasks}")
print(f"  input:  {total_inp:,}")
print(f"  output: {total_out:,}")
print(f"  total:  {total_inp + total_out:,}")

corrected = int((total_inp + total_out) * 1.96)
bill_total = 93_795_046

print(f"\nWith Bug1 fix (1.96x correction):")
print(f"  estimated real:      {corrected:,}")

print(f"\nBilling total:         {bill_total:,} tokens")
print(f"  JSONL recorded:      {total_inp + total_out:,}")
print(f"  JSONL corrected x1.96: {corrected:,}")
print(f"  Difference:          {bill_total - corrected:,}")
print(f"  Benchmark %:         {corrected / bill_total * 100:.1f}%")
print(f"  Dev/debug %:         {(bill_total - corrected) / bill_total * 100:.1f}%")

avg_buggy = (total_inp + total_out) / total_tasks if total_tasks else 0
avg_fixed = avg_buggy * 1.96
avg_measured = (114724 + 8996) / 5

print(f"\nAvg per GAIA task (buggy):    {avg_buggy:,.0f}")
print(f"Avg per GAIA task (x1.96):    {avg_fixed:,.0f}")
print(f"Avg per GAIA task (measured):  {avg_measured:,.0f}")

print(f"\n{sep}")
print("5-task API 拦截结论: 修复后比率 = 1.000x (完全准确)")
print(f"旧 JSONL 平均: {avg_buggy:,.0f} tok/task")
print(f"修复后实测:     {avg_measured:,.0f} tok/task")
print(f"修正倍率:       {avg_measured / avg_buggy:.2f}x")
print(f"\n如果用实测值估算所有 JSONL 里 {total_tasks} 个任务:")
print(f"  estimated = {total_tasks} × {avg_measured:,.0f} = {int(total_tasks * avg_measured):,}")
print(f"  vs billing {bill_total:,}")
print(f"  benchmark占比: {total_tasks * avg_measured / bill_total * 100:.1f}%")
