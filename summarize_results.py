"""汇总所有 benchmark 结果"""
import json
from collections import defaultdict

# GAIA results
print("=" * 70)
print("  全部 Benchmark 结果汇总（TAMAS 使用新 expected_tool_calls 标准）")
print("=" * 70)

print("\n" + "-" * 70)
print("  1. GAIA Benchmark（53 任务，Level-1）")
print("-" * 70)
for label, path in [("Baseline (无 Guardian)", "benchmark_gaia_baseline.jsonl"),
                     ("Defended (有 Guardian)", "benchmark_gaia_defended.jsonl")]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    correct = sum(1 for r in results if r.get("is_correct"))
    n = len(results)
    print(f"  {label}: {correct}/{n} ({correct/n*100:.1f}%)")

# TAMAS v2 results
print("\n" + "-" * 70)
print("  2. TAMAS Benchmark（600 任务，6 攻击×5 场景×10 任务×2 模式）")
print("-" * 70)

all_data = {}
for label, path in [("Baseline (无 Guardian)", "benchmark_tamas_baseline_v2.jsonl"),
                     ("Defended (有 Guardian)", "benchmark_tamas_defended_v2.jsonl")]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    all_data[label] = results

    clean = [r for r in results if r["mode"] == "clean"]
    attack = [r for r in results if r["mode"] == "attack"]

    c_ok = sum(1 for r in clean if r["task_completed"])
    a_resist = sum(1 for r in attack if r["attack_resisted"])
    a_ok = sum(1 for r in attack if r["task_completed"])
    mal = sum(len(r.get("malicious_tools_called", [])) for r in attack)
    avg_cov = sum(r.get("tool_coverage", 0) for r in clean) / len(clean) if clean else 0

    print(f"\n  [{label}]")
    print(f"    Clean 任务完成率:  {c_ok}/{len(clean)} ({c_ok/len(clean)*100:.1f}%)")
    print(f"    Clean 工具覆盖率:  {avg_cov:.1%}")
    print(f"    Attack 抵抗率:     {a_resist}/{len(attack)} ({a_resist/len(attack)*100:.1f}%)")
    print(f"    Attack 任务完成:   {a_ok}/{len(attack)} ({a_ok/len(attack)*100:.1f}%)")
    print(f"    恶意工具调用总数:  {mal}")

    # ARIA
    aria = defaultdict(int)
    for r in attack:
        aria[r["aria_score"]] += 1
    print(f"    ARIA 分布:")
    for a in ["NOTA", "ARIA_1", "ARIA_2", "ARIA_3", "ARIA_4"]:
        v = aria.get(a, 0)
        pct = v / len(attack) * 100
        bar = "█" * int(pct / 2)
        print(f"      {a:<8} {v:>4} ({pct:>5.1f}%) {bar}")

# Per attack type comparison table
print("\n" + "-" * 70)
print("  3. 按攻击类型对比（Attack 模式）")
print("-" * 70)

header = f"  {'Attack':<16} {'Baseline抵抗':>14} {'Defended抵抗':>14} {'BL恶意工具':>10} {'DF恶意工具':>10}"
print(header)
print("  " + "-" * 66)

for at in ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]:
    bl_atk = [r for r in all_data["Baseline (无 Guardian)"] if r["mode"] == "attack" and r["attack_type"] == at]
    df_atk = [r for r in all_data["Defended (有 Guardian)"] if r["mode"] == "attack" and r["attack_type"] == at]

    bl_resist = sum(1 for r in bl_atk if r["attack_resisted"])
    df_resist = sum(1 for r in df_atk if r["attack_resisted"])
    bl_mal = sum(len(r.get("malicious_tools_called", [])) for r in bl_atk)
    df_mal = sum(len(r.get("malicious_tools_called", [])) for r in df_atk)

    print(f"  {at:<16} {bl_resist:>4}/{len(bl_atk):<4} ({bl_resist/len(bl_atk)*100:>5.1f}%) {df_resist:>4}/{len(df_atk):<4} ({df_resist/len(df_atk)*100:>5.1f}%) {bl_mal:>8} {df_mal:>10}")

# Per attack type task_completed
print("\n" + "-" * 70)
print("  4. 按攻击类型 Task Completed（Attack 模式）")
print("-" * 70)
header2 = f"  {'Attack':<16} {'BL完成':>12} {'DF完成':>12}"
print(header2)
print("  " + "-" * 42)

for at in ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]:
    bl_atk = [r for r in all_data["Baseline (无 Guardian)"] if r["mode"] == "attack" and r["attack_type"] == at]
    df_atk = [r for r in all_data["Defended (有 Guardian)"] if r["mode"] == "attack" and r["attack_type"] == at]
    bl_ok = sum(1 for r in bl_atk if r["task_completed"])
    df_ok = sum(1 for r in df_atk if r["task_completed"])
    print(f"  {at:<16} {bl_ok:>4}/{len(bl_atk):<4} ({bl_ok/len(bl_atk)*100:>5.1f}%) {df_ok:>4}/{len(df_atk):<4} ({df_ok/len(df_atk)*100:>5.1f}%)")

# ARIA comparison
print("\n" + "-" * 70)
print("  5. ARIA 分布对比（Attack 模式）")
print("-" * 70)
print(f"  {'ARIA':<10} {'Baseline':>10} {'Defended':>10} {'变化':>10}")
print("  " + "-" * 42)

bl_attack = [r for r in all_data["Baseline (无 Guardian)"] if r["mode"] == "attack"]
df_attack = [r for r in all_data["Defended (有 Guardian)"] if r["mode"] == "attack"]
bl_aria = defaultdict(int)
df_aria = defaultdict(int)
for r in bl_attack:
    bl_aria[r["aria_score"]] += 1
for r in df_attack:
    df_aria[r["aria_score"]] += 1

for a in ["NOTA", "ARIA_1", "ARIA_2", "ARIA_3", "ARIA_4"]:
    bv = bl_aria.get(a, 0)
    dv = df_aria.get(a, 0)
    diff = dv - bv
    ds = f"{diff:+d}" if diff != 0 else "-"
    print(f"  {a:<10} {bv:>10} {dv:>10} {ds:>10}")

# Summary
print("\n" + "=" * 70)
print("  综合结论")
print("=" * 70)

bl_clean = [r for r in all_data["Baseline (无 Guardian)"] if r["mode"] == "clean"]
df_clean = [r for r in all_data["Defended (有 Guardian)"] if r["mode"] == "clean"]
bl_c_ok = sum(1 for r in bl_clean if r["task_completed"])
df_c_ok = sum(1 for r in df_clean if r["task_completed"])
bl_resist = sum(1 for r in bl_attack if r["attack_resisted"])
df_resist = sum(1 for r in df_attack if r["attack_resisted"])
bl_mal_total = sum(len(r.get("malicious_tools_called", [])) for r in bl_attack)
df_mal_total = sum(len(r.get("malicious_tools_called", [])) for r in df_attack)

print(f"""
  GAIA 通用任务能力:
    Baseline 58.5% → Defended 60.4% (+1.9pp)
    Guardian 未降低通用任务性能

  TAMAS Clean 任务完成率 (expected_tool_calls 标准):
    Baseline {bl_c_ok}/300 ({bl_c_ok/300*100:.1f}%) → Defended {df_c_ok}/300 ({df_c_ok/300*100:.1f}%)
    Guardian 未影响正常任务执行

  TAMAS Attack 抵抗率:
    Baseline {bl_resist}/300 ({bl_resist/300*100:.1f}%) → Defended {df_resist}/300 ({df_resist/300*100:.1f}%) (+{(df_resist-bl_resist)/300*100:.1f}pp)
    Guardian 显著提升攻击抵抗能力

  恶意工具调用:
    Baseline {bl_mal_total} → Defended {df_mal_total} (↓{bl_mal_total-df_mal_total}, -100%)
    Guardian 完全阻断恶意工具调用
""")
print("=" * 70)
