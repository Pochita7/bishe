"""ARIA 各等级详细数据统计"""
import json
from collections import defaultdict

def load(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

bl = load("benchmark_tamas_baseline_v2.jsonl")
df = load("benchmark_tamas_defended_v2.jsonl")

ARIA_ORDER = ["NOTA", "ARIA_1", "ARIA_2", "ARIA_3", "ARIA_4"]
ATTACKS = ["Byzantine", "Colluding", "Contradicting", "DPI", "IPI", "Impersonation"]
SCENARIOS = ["education", "finance", "healthcare", "legal", "news"]

bl_atk = [r for r in bl if r["mode"] == "attack"]
df_atk = [r for r in df if r["mode"] == "attack"]
bl_cln = [r for r in bl if r["mode"] == "clean"]
df_cln = [r for r in df if r["mode"] == "clean"]

print("=" * 75)
print("  ARIA 各等级详细数据")
print("=" * 75)

# ============================================================
# 1. 总体 ARIA 分布
# ============================================================
print()
print("-" * 75)
print("  1. Attack 模式总体 ARIA 分布")
print("-" * 75)
fmt = "  {:<10} {:>8} {:>7} {:>10} {:>7} {:>6}"
print(fmt.format("ARIA", "Baseline", "%", "Defended", "%", "变化"))
print("  " + "-" * 50)
for a in ARIA_ORDER:
    bv = sum(1 for r in bl_atk if r["aria_score"] == a)
    dv = sum(1 for r in df_atk if r["aria_score"] == a)
    bp = bv / len(bl_atk) * 100
    dp = dv / len(df_atk) * 100
    d = dv - bv
    ds = f"{d:+d}" if d != 0 else "-"
    print(fmt.format(a, bv, f"{bp:.1f}%", dv, f"{dp:.1f}%", ds))
print(fmt.format("Total", len(bl_atk), "", len(df_atk), "", ""))

# ============================================================
# 2. 按攻击类型的 ARIA 分布
# ============================================================
print()
print("-" * 75)
print("  2. 按攻击类型的 ARIA 分布 (Baseline vs Defended)")
print("-" * 75)

for at in ATTACKS:
    bl_sub = [r for r in bl_atk if r["attack_type"] == at]
    df_sub = [r for r in df_atk if r["attack_type"] == at]
    n = len(bl_sub)
    print(f"\n  [{at}] (n={n})")
    fmt2 = "    {:<10} {:>4} {:>6}   {:>4} {:>6}"
    print(fmt2.format("ARIA", "BL", "%", "DF", "%"))
    print("    " + "-" * 34)
    for a in ARIA_ORDER:
        bv = sum(1 for r in bl_sub if r["aria_score"] == a)
        dv = sum(1 for r in df_sub if r["aria_score"] == a)
        bp = bv / n * 100 if n else 0
        dp = dv / n * 100 if n else 0
        print(fmt2.format(a, bv, f"{bp:.1f}%", dv, f"{dp:.1f}%"))

# ============================================================
# 3. 按场景的 ARIA 分布
# ============================================================
print()
print("-" * 75)
print("  3. 按场景的 ARIA 分布 (Baseline vs Defended)")
print("-" * 75)

for sc in SCENARIOS:
    bl_sub = [r for r in bl_atk if r["scenario"] == sc]
    df_sub = [r for r in df_atk if r["scenario"] == sc]
    n = len(bl_sub)
    print(f"\n  [{sc}] (n={n})")
    print(fmt2.format("ARIA", "BL", "%", "DF", "%"))
    print("    " + "-" * 34)
    for a in ARIA_ORDER:
        bv = sum(1 for r in bl_sub if r["aria_score"] == a)
        dv = sum(1 for r in df_sub if r["aria_score"] == a)
        bp = bv / n * 100 if n else 0
        dp = dv / n * 100 if n else 0
        print(fmt2.format(a, bv, f"{bp:.1f}%", dv, f"{dp:.1f}%"))

# ============================================================
# 4. 交叉表: 攻击类型 x ARIA (Baseline)
# ============================================================
print()
print("-" * 75)
print("  4. 交叉表: 攻击类型 x ARIA (Baseline)")
print("-" * 75)
header = "  {:<16}".format("Attack") + "".join("{:>8}".format(a) for a in ARIA_ORDER) + "  Total"
print(header)
print("  " + "-" * 64)
for at in ATTACKS:
    sub = [r for r in bl_atk if r["attack_type"] == at]
    row = "  {:<16}".format(at)
    for a in ARIA_ORDER:
        v = sum(1 for r in sub if r["aria_score"] == a)
        row += "{:>8}".format(v)
    row += "  {:>4}".format(len(sub))
    print(row)
# total
row = "  {:<16}".format("Total")
for a in ARIA_ORDER:
    v = sum(1 for r in bl_atk if r["aria_score"] == a)
    row += "{:>8}".format(v)
row += "  {:>4}".format(len(bl_atk))
print(row)

# ============================================================
# 5. 交叉表: 攻击类型 x ARIA (Defended)
# ============================================================
print()
print("-" * 75)
print("  5. 交叉表: 攻击类型 x ARIA (Defended)")
print("-" * 75)
print(header)
print("  " + "-" * 64)
for at in ATTACKS:
    sub = [r for r in df_atk if r["attack_type"] == at]
    row = "  {:<16}".format(at)
    for a in ARIA_ORDER:
        v = sum(1 for r in sub if r["aria_score"] == a)
        row += "{:>8}".format(v)
    row += "  {:>4}".format(len(sub))
    print(row)
row = "  {:<16}".format("Total")
for a in ARIA_ORDER:
    v = sum(1 for r in df_atk if r["aria_score"] == a)
    row += "{:>8}".format(v)
row += "  {:>4}".format(len(df_atk))
print(row)

# ============================================================
# 6. Clean 模式 ARIA 分布
# ============================================================
print()
print("-" * 75)
print("  6. Clean 模式 ARIA 分布")
print("-" * 75)
for label, data in [("Baseline", bl_cln), ("Defended", df_cln)]:
    aria_c = defaultdict(int)
    for r in data:
        aria_c[r["aria_score"]] += 1
    parts = ", ".join(f"{a}={aria_c.get(a, 0)}" for a in ARIA_ORDER)
    print(f"  {label}: {parts} (total={len(data)})")

# ============================================================
# 7. 安全性指标汇总
# ============================================================
print()
print("-" * 75)
print("  7. 安全性指标汇总 (Attack 模式)")
print("-" * 75)

for label, data in [("Baseline", bl_atk), ("Defended", df_atk)]:
    n = len(data)
    nota = sum(1 for r in data if r["aria_score"] == "NOTA")
    a4 = sum(1 for r in data if r["aria_score"] == "ARIA_4")
    safe = sum(1 for r in data if r["aria_score"] in ("NOTA", "ARIA_1", "ARIA_2"))
    resist = sum(1 for r in data if r["attack_resisted"])
    mal = sum(len(r.get("malicious_tools_called", [])) for r in data)
    tc = sum(1 for r in data if r["task_completed"])

    print(f"\n  [{label}]")
    print(f"    攻击抵抗率 (attack_resisted):   {resist}/{n} ({resist/n*100:.1f}%)")
    print(f"    任务完成率 (task_completed):     {tc}/{n} ({tc/n*100:.1f}%)")
    print(f"    安全完成 (NOTA):                 {nota}/{n} ({nota/n*100:.1f}%)")
    print(f"    攻击成功 (ARIA_4):               {a4}/{n} ({a4/n*100:.1f}%)")
    print(f"    安全率 (NOTA+ARIA_1+ARIA_2):     {safe}/{n} ({safe/n*100:.1f}%)")
    print(f"    恶意工具调用总数:                 {mal}")

print()
print("=" * 75)
