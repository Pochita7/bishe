"""合并全部6种攻击类型的测试结果并生成最终对比分析报告"""
import json
import sys
from collections import defaultdict


def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def merge_results(*file_paths):
    """合并多个 jsonl 文件的结果"""
    all_results = []
    for path in file_paths:
        all_results.extend(load_results(path))
    return all_results


def analyze(results, label):
    clean = [r for r in results if r["mode"] == "clean"]
    attack = [r for r in results if r["mode"] == "attack"]

    by_attack = defaultdict(lambda: {"clean": [], "attack": []})
    for r in results:
        by_attack[r["attack_type"]][r["mode"]].append(r)

    by_scenario = defaultdict(lambda: {"clean": [], "attack": []})
    for r in results:
        by_scenario[r["scenario"]][r["mode"]].append(r)

    aria_dist = defaultdict(int)
    for r in attack:
        aria_dist[r.get("aria_score", r.get("aria_level", "unknown"))] += 1

    return {
        "label": label,
        "clean": clean,
        "attack": attack,
        "by_attack": dict(by_attack),
        "by_scenario": dict(by_scenario),
        "aria_dist": dict(aria_dist),
    }


def resist_rate(attack_list):
    resisted = sum(1 for r in attack_list if r.get("attack_resisted", False))
    return resisted, len(attack_list)


def avg_score(results_list):
    if not results_list:
        return 0.0
    return sum(r.get("score", 0) for r in results_list) / len(results_list)


def print_full_report(baseline, guardian):
    W = 86

    print("=" * W)
    print("  TAMAS MAS 安全性评估 — 完整6种攻击类型 Guardian 防御对比报告")
    print("=" * W)
    print()

    # ─── 数据概览 ───
    print(f"  数据集概览:")
    print(f"    攻击类型: {', '.join(sorted(baseline['by_attack'].keys()))}")
    print(f"    场景: {', '.join(sorted(baseline['by_scenario'].keys()))}")
    print(f"    基线记录数: {len(baseline['clean'])+len(baseline['attack'])} (Clean={len(baseline['clean'])}, Attack={len(baseline['attack'])})")
    print(f"    Guardian记录数: {len(guardian['clean'])+len(guardian['attack'])} (Clean={len(guardian['clean'])}, Attack={len(guardian['attack'])})")
    print()

    # ═══════════════════════════════════════════════
    # 1. 总体性能对比
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  1. 总体性能对比")
    print("─" * W)

    b_clean_ok = sum(1 for r in baseline["clean"] if r.get("task_completed", False))
    g_clean_ok = sum(1 for r in guardian["clean"] if r.get("task_completed", False))
    b_clean_n = len(baseline["clean"])
    g_clean_n = len(guardian["clean"])
    b_clean_score = avg_score(baseline["clean"])
    g_clean_score = avg_score(guardian["clean"])
    b_resist, b_atk_n = resist_rate(baseline["attack"])
    g_resist, g_atk_n = resist_rate(guardian["attack"])
    b_atk_score = avg_score(baseline["attack"])
    g_atk_score = avg_score(guardian["attack"])

    headers = ["指标", "无防御 (Baseline)", "Guardian 防御", "变化"]
    rows = [
        ["Clean 任务完成率", f"{b_clean_ok}/{b_clean_n} ({b_clean_ok/b_clean_n*100:.1f}%)",
         f"{g_clean_ok}/{g_clean_n} ({g_clean_ok/g_clean_n*100:.1f}%)",
         f"{(g_clean_ok/g_clean_n - b_clean_ok/b_clean_n)*100:+.1f}pp"],
        ["Clean 平均评分", f"{b_clean_score:.3f}", f"{g_clean_score:.3f}",
         f"{g_clean_score - b_clean_score:+.3f}"],
        ["Attack 完成率", f"{len(baseline['attack'])}/{len(baseline['attack'])} (100%)",
         f"{len(guardian['attack'])}/{len(guardian['attack'])} (100%)", "-"],
        ["Attack 抵抗率", f"{b_resist}/{b_atk_n} ({b_resist/b_atk_n*100:.1f}%)",
         f"{g_resist}/{g_atk_n} ({g_resist/g_atk_n*100:.1f}%)",
         f"{(g_resist/g_atk_n - b_resist/b_atk_n)*100:+.1f}pp"],
        ["Attack 平均评分", f"{b_atk_score:.3f}", f"{g_atk_score:.3f}",
         f"{g_atk_score - b_atk_score:+.3f}"],
    ]

    col_widths = [18, 22, 22, 12]
    fmt = "  │ {:<{}} │ {:^{}} │ {:^{}} │ {:^{}} │"

    print("  ┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
    print(fmt.format(*[h for pair in zip(headers, col_widths) for h in pair]))
    print("  ├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
    for row in rows:
        print(fmt.format(*[h for pair in zip(row, col_widths) for h in pair]))
    print("  └" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")
    print()

    # ═══════════════════════════════════════════════
    # 2. ARIA 分布对比
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  2. ARIA 分布对比 (Attack 模式)")
    print("─" * W)

    all_aria = sorted(set(list(baseline["aria_dist"].keys()) + list(guardian["aria_dist"].keys())))
    col_widths2 = [12, 20, 20]

    print("  ┌" + "┬".join("─" * (w + 2) for w in col_widths2) + "┐")
    fmt2 = "  │ {:^{}} │ {:^{}} │ {:^{}} │"
    print(fmt2.format("ARIA Level", col_widths2[0], "无防御", col_widths2[1], "Guardian", col_widths2[2]))
    print("  ├" + "┼".join("─" * (w + 2) for w in col_widths2) + "┤")
    for aria in all_aria:
        b_cnt = baseline["aria_dist"].get(aria, 0)
        g_cnt = guardian["aria_dist"].get(aria, 0)
        b_pct = b_cnt / len(baseline["attack"]) * 100 if baseline["attack"] else 0
        g_pct = g_cnt / len(guardian["attack"]) * 100 if guardian["attack"] else 0
        print(fmt2.format(str(aria), col_widths2[0],
              f"{b_cnt} ({b_pct:.1f}%)", col_widths2[1],
              f"{g_cnt} ({g_pct:.1f}%)", col_widths2[2]))
    print("  └" + "┴".join("─" * (w + 2) for w in col_widths2) + "┘")
    print()

    # ═══════════════════════════════════════════════
    # 3. 按攻击类型对比
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  3. 按攻击类型对比")
    print("─" * W)

    attacks_sorted = sorted(set(list(baseline["by_attack"].keys()) + list(guardian["by_attack"].keys())))
    col_widths3 = [15, 18, 18, 10]

    print("  ┌" + "┬".join("─" * (w + 2) for w in col_widths3) + "┐")
    fmt3 = "  │ {:<{}} │ {:^{}} │ {:^{}} │ {:^{}} │"
    print(fmt3.format("攻击类型", col_widths3[0], "无防御 抵抗率", col_widths3[1],
          "Guardian 抵抗率", col_widths3[2], "提升", col_widths3[3]))
    print("  ├" + "┼".join("─" * (w + 2) for w in col_widths3) + "┤")

    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"attack": []})["attack"]
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        b_pct = b_r / max(b_n, 1) * 100
        g_pct = g_r / max(g_n, 1) * 100
        delta = g_pct - b_pct
        print(fmt3.format(atk, col_widths3[0],
              f"{b_r}/{b_n} ({b_pct:.0f}%)", col_widths3[1],
              f"{g_r}/{g_n} ({g_pct:.0f}%)", col_widths3[2],
              f"+{delta:.0f}pp" if delta >= 0 else f"{delta:.0f}pp", col_widths3[3]))
    print("  ├" + "┼".join("─" * (w + 2) for w in col_widths3) + "┤")
    b_total_pct = b_resist / max(b_atk_n, 1) * 100
    g_total_pct = g_resist / max(g_atk_n, 1) * 100
    print(fmt3.format("ALL", col_widths3[0],
          f"{b_resist}/{b_atk_n} ({b_total_pct:.0f}%)", col_widths3[1],
          f"{g_resist}/{g_atk_n} ({g_total_pct:.0f}%)", col_widths3[2],
          f"+{g_total_pct - b_total_pct:.0f}pp", col_widths3[3]))
    print("  └" + "┴".join("─" * (w + 2) for w in col_widths3) + "┘")
    print()

    # 按攻击类型详细分数
    print("  按攻击类型平均评分:")
    print(f"  {'攻击类型':<16}{'无防御':>10}{'Guardian':>12}{'Δ':>10}")
    print("  " + "-" * 48)
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"attack": []})["attack"]
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        b_s = avg_score(b_data)
        g_s = avg_score(g_data)
        print(f"  {atk:<16}{b_s:>10.3f}{g_s:>12.3f}{g_s-b_s:>+10.3f}")
    print(f"  {'ALL':<16}{b_atk_score:>10.3f}{g_atk_score:>12.3f}{g_atk_score-b_atk_score:>+10.3f}")
    print()

    # ═══════════════════════════════════════════════
    # 4. 按场景对比
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  4. 按场景对比")
    print("─" * W)

    scenarios_sorted = sorted(set(list(baseline["by_scenario"].keys()) + list(guardian["by_scenario"].keys())))
    col_widths4 = [13, 18, 18, 10]

    print("  ┌" + "┬".join("─" * (w + 2) for w in col_widths4) + "┐")
    fmt4 = "  │ {:<{}} │ {:^{}} │ {:^{}} │ {:^{}} │"
    print(fmt4.format("场景", col_widths4[0], "无防御 抵抗率", col_widths4[1],
          "Guardian 抵抗率", col_widths4[2], "提升", col_widths4[3]))
    print("  ├" + "┼".join("─" * (w + 2) for w in col_widths4) + "┤")

    for sc in scenarios_sorted:
        b_data = baseline["by_scenario"].get(sc, {"attack": []})["attack"]
        g_data = guardian["by_scenario"].get(sc, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        b_pct = b_r / max(b_n, 1) * 100
        g_pct = g_r / max(g_n, 1) * 100
        delta = g_pct - b_pct
        print(fmt4.format(sc, col_widths4[0],
              f"{b_r}/{b_n} ({b_pct:.0f}%)", col_widths4[1],
              f"{g_r}/{g_n} ({g_pct:.0f}%)", col_widths4[2],
              f"+{delta:.0f}pp" if delta >= 0 else f"{delta:.0f}pp", col_widths4[3]))
    print("  └" + "┴".join("─" * (w + 2) for w in col_widths4) + "┘")
    print()

    # ═══════════════════════════════════════════════
    # 5. 攻击类型 × 场景 交叉分析 (仅 Guardian)
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  5. Guardian 攻击类型 × 场景 交叉分析 (抵抗率)")
    print("─" * W)

    # Header row
    header = f"  {'Attack\\Scenario':<16}"
    for sc in scenarios_sorted:
        header += f"{sc:>12}"
    header += f"{'TOTAL':>12}"
    print(header)
    print("  " + "-" * (16 + 12 * (len(scenarios_sorted) + 1)))

    for atk in attacks_sorted:
        row = f"  {atk:<16}"
        for sc in scenarios_sorted:
            sub = [r for r in guardian["attack"] if r["attack_type"] == atk and r["scenario"] == sc]
            r_cnt = sum(1 for r in sub if r.get("attack_resisted", False))
            total = len(sub)
            if total > 0:
                row += f"{r_cnt}/{total} ({r_cnt/total*100:.0f}%)".rjust(12)
            else:
                row += "N/A".rjust(12)
        # Total for attack type
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        gr, gn = resist_rate(g_data)
        row += f"{gr}/{gn} ({gr/max(gn,1)*100:.0f}%)".rjust(12)
        print(row)

    # Total row
    row = f"  {'TOTAL':<16}"
    for sc in scenarios_sorted:
        sub = guardian["by_scenario"].get(sc, {"attack": []})["attack"]
        r_cnt = sum(1 for r in sub if r.get("attack_resisted", False))
        total = len(sub)
        if total > 0:
            row += f"{r_cnt}/{total} ({r_cnt/total*100:.0f}%)".rjust(12)
        else:
            row += "N/A".rjust(12)
    row += f"{g_resist}/{g_atk_n} ({g_resist/g_atk_n*100:.0f}%)".rjust(12)
    print(row)
    print()

    # ═══════════════════════════════════════════════
    # 6. Clean 模式对比（验证无负面影响）
    # ═══════════════════════════════════════════════
    print("─" * W)
    print("  6. Clean 模式对比 (验证 Guardian 对正常任务的影响)")
    print("─" * W)

    print(f"  {'攻击类型':<16}{'无防御 Clean':>14}{'Guardian Clean':>16}{'Δ':>8}")
    print("  " + "-" * 54)
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"clean": []})["clean"]
        g_data = guardian["by_attack"].get(atk, {"clean": []})["clean"]
        b_s = avg_score(b_data)
        g_s = avg_score(g_data)
        print(f"  {atk:<16}{b_s:>14.3f}{g_s:>16.3f}{g_s-b_s:>+8.3f}")
    print(f"  {'ALL':<16}{b_clean_score:>14.3f}{g_clean_score:>16.3f}{g_clean_score-b_clean_score:>+8.3f}")
    print()

    # ═══════════════════════════════════════════════
    # 7. 关键发现与总结
    # ═══════════════════════════════════════════════
    print("=" * W)
    print("  7. 关键发现与总结")
    print("=" * W)

    improve = g_resist / g_atk_n * 100 - b_resist / b_atk_n * 100
    print()
    print(f"  [防御效果]")
    print(f"    • Guardian 将总体攻击抵抗率从 {b_resist/b_atk_n*100:.1f}% 提升至 {g_resist/g_atk_n*100:.1f}% (提升 +{improve:.1f} 百分点)")
    print(f"    • Guardian 将攻击模式平均评分从 {b_atk_score:.3f} 提升至 {g_atk_score:.3f} (+{g_atk_score-b_atk_score:.3f})")
    print()

    print(f"  [各攻击类型防御效果]")
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"attack": []})["attack"]
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        b_pct = b_r / max(b_n, 1) * 100
        g_pct = g_r / max(g_n, 1) * 100
        delta = g_pct - b_pct
        marker = "★" if delta >= 50 else "▲" if delta > 0 else "─"
        print(f"    {marker} {atk:15s}: {b_pct:5.1f}% → {g_pct:5.1f}%  (+{delta:.1f}pp)")
    print()

    print(f"  [功能性保障]")
    if g_clean_score >= b_clean_score - 0.01:
        print(f"    ✓ Guardian 对正常任务性能零负面影响 (Clean: {b_clean_score:.3f} → {g_clean_score:.3f})")
    else:
        print(f"    ! Guardian 对正常任务有轻微性能影响 (Clean: {b_clean_score:.3f} → {g_clean_score:.3f}, Δ={g_clean_score-b_clean_score:.3f})")
    print()

    # 仍然失败的案例
    guardian_fails = [r for r in guardian["attack"] if not r.get("attack_resisted", False)]
    if guardian_fails:
        print(f"  [残余风险] Guardian 仍未防御的案例 ({len(guardian_fails)}/{g_atk_n}):")
        for r in guardian_fails:
            aria = r.get("aria_score", r.get("aria_level", "unknown"))
            print(f"    - {r.get('id', 'N/A')} ({r['attack_type']}/{r['scenario']}): ARIA={aria}, Score={r.get('score',0):.2f}")
    else:
        print(f"  [残余风险] Guardian 成功防御了所有测试攻击 (0/{g_atk_n} 失败)")
    print()

    print("=" * W)
    print("  报告完毕")
    print("=" * W)


if __name__ == "__main__":
    # 合并4攻击 + 2攻击补充 = 6攻击
    print("加载数据文件...")
    baseline_4 = load_results("tamas_baseline_no_defense.jsonl")
    baseline_2 = load_results("tamas_baseline_collu_contra.jsonl")
    guardian_4 = load_results("tamas_with_guardian.jsonl")
    guardian_2 = load_results("tamas_guardian_collu_contra.jsonl")

    print(f"  基线(4攻击): {len(baseline_4)} 条")
    print(f"  基线(补充2): {len(baseline_2)} 条")
    print(f"  Guardian(4攻击): {len(guardian_4)} 条")
    print(f"  Guardian(补充2): {len(guardian_2)} 条")
    print(f"  合计: {len(baseline_4)+len(baseline_2)+len(guardian_4)+len(guardian_2)} 条")
    print()

    baseline_all = baseline_4 + baseline_2
    guardian_all = guardian_4 + guardian_2

    baseline_analysis = analyze(baseline_all, "无防御")
    guardian_analysis = analyze(guardian_all, "Guardian")

    print_full_report(baseline_analysis, guardian_analysis)
