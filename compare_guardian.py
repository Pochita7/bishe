"""对比 Guardian 防御前后的测试结果"""
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


def analyze(results, label):
    clean = [r for r in results if r["mode"] == "clean"]
    attack = [r for r in results if r["mode"] == "attack"]

    # 按攻击类型
    by_attack = defaultdict(lambda: {"clean": [], "attack": []})
    for r in results:
        by_attack[r["attack_type"]][r["mode"]].append(r)

    # 按场景
    by_scenario = defaultdict(lambda: {"clean": [], "attack": []})
    for r in results:
        by_scenario[r["scenario"]][r["mode"]].append(r)

    # ARIA 分布
    aria_dist = defaultdict(int)
    for r in attack:
        aria_dist[r["aria_score"]] += 1

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
    return sum(r["score"] for r in results_list) / len(results_list)


def print_comparison(baseline, guardian):
    print("=" * 80)
    print("  TAMAS MAS 安全测试 — Guardian 防御对比报告")
    print("=" * 80)
    print()

    # 总体对比
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        总体性能对比                                │")
    print("├─────────────┬──────────────────────┬──────────────────────┤")
    print("│   指标       │   无防御 (Baseline)  │   Guardian 防御       │")
    print("├─────────────┼──────────────────────┼──────────────────────┤")

    # Clean 任务完成率
    b_clean_ok = sum(1 for r in baseline["clean"] if r.get("task_completed", False))
    g_clean_ok = sum(1 for r in guardian["clean"] if r.get("task_completed", False))
    b_clean_n = len(baseline["clean"])
    g_clean_n = len(guardian["clean"])
    print(f"│ Clean完成率  │  {b_clean_ok}/{b_clean_n} ({b_clean_ok/b_clean_n*100:.1f}%)            │  {g_clean_ok}/{g_clean_n} ({g_clean_ok/g_clean_n*100:.1f}%)            │")

    # Clean 平均分
    b_clean_score = avg_score(baseline["clean"])
    g_clean_score = avg_score(guardian["clean"])
    print(f"│ Clean平均分  │  {b_clean_score:.3f}                │  {g_clean_score:.3f}                │")

    # Attack 抵抗率
    b_resist, b_atk_n = resist_rate(baseline["attack"])
    g_resist, g_atk_n = resist_rate(guardian["attack"])
    print(f"│ Attack抵抗率 │  {b_resist}/{b_atk_n} ({b_resist/b_atk_n*100:.1f}%)            │  {g_resist}/{g_atk_n} ({g_resist/g_atk_n*100:.1f}%)            │")

    # Attack 平均分
    b_atk_score = avg_score(baseline["attack"])
    g_atk_score = avg_score(guardian["attack"])
    delta = g_atk_score - b_atk_score
    print(f"│ Attack平均分 │  {b_atk_score:.3f}                │  {g_atk_score:.3f} ({'+' if delta >= 0 else ''}{delta:.3f})        │")
    print("└─────────────┴──────────────────────┴──────────────────────┘")
    print()

    # ARIA 分布对比
    print("┌───────────────────────────────────────────────────────────┐")
    print("│                   ARIA 分布对比 (Attack)                  │")
    print("├──────────┬────────────────────┬────────────────────┤")
    print("│  ARIA    │  无防御            │  Guardian           │")
    print("├──────────┼────────────────────┼────────────────────┤")
    all_aria = sorted(set(list(baseline["aria_dist"].keys()) + list(guardian["aria_dist"].keys())))
    for aria in all_aria:
        b_cnt = baseline["aria_dist"].get(aria, 0)
        g_cnt = guardian["aria_dist"].get(aria, 0)
        b_atk_total = len(baseline["attack"])
        g_atk_total = len(guardian["attack"])
        print(f"│  {aria:8s}│  {b_cnt:2d} ({b_cnt/b_atk_total*100:5.1f}%)         │  {g_cnt:2d} ({g_cnt/g_atk_total*100:5.1f}%)         │")
    print("└──────────┴────────────────────┴────────────────────┘")
    print()

    # 按攻击类型对比
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                        按攻击类型对比                                       │")
    print("├───────────────┬─────────────────────────┬──────────────────────────┤")
    print("│  Attack Type  │  无防御 (抵抗/总 | 均分) │  Guardian (抵抗/总 | 均分) │")
    print("├───────────────┼─────────────────────────┼──────────────────────────┤")
    attacks_sorted = sorted(set(list(baseline["by_attack"].keys()) + list(guardian["by_attack"].keys())))
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"attack": []})["attack"]
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        b_s = avg_score(b_data)
        g_s = avg_score(g_data)
        delta = g_s - b_s
        print(f"│  {atk:13s}│  {b_r:2d}/{b_n:2d} ({b_r/max(b_n,1)*100:5.1f}%) | {b_s:.3f}  │  {g_r:2d}/{g_n:2d} ({g_r/max(g_n,1)*100:5.1f}%) | {g_s:.3f}   │  Δ{'+' if delta>=0 else ''}{delta:.3f}")
    print("└───────────────┴─────────────────────────┴──────────────────────────┘")
    print()

    # 按场景对比
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                        按场景对比                                           │")
    print("├───────────────┬─────────────────────────┬──────────────────────────┤")
    print("│  Scenario     │  无防御 (抵抗/总 | 均分) │  Guardian (抵抗/总 | 均分) │")
    print("├───────────────┼─────────────────────────┼──────────────────────────┤")
    scenarios_sorted = sorted(set(list(baseline["by_scenario"].keys()) + list(guardian["by_scenario"].keys())))
    for sc in scenarios_sorted:
        b_data = baseline["by_scenario"].get(sc, {"attack": []})["attack"]
        g_data = guardian["by_scenario"].get(sc, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        b_s = avg_score(b_data)
        g_s = avg_score(g_data)
        delta = g_s - b_s
        print(f"│  {sc:13s}│  {b_r:2d}/{b_n:2d} ({b_r/max(b_n,1)*100:5.1f}%) | {b_s:.3f}  │  {g_r:2d}/{g_n:2d} ({g_r/max(g_n,1)*100:5.1f}%) | {g_s:.3f}   │  Δ{'+' if delta>=0 else ''}{delta:.3f}")
    print("└───────────────┴─────────────────────────┴──────────────────────────┘")
    print()

    # Clean 模式对比（验证 Guardian 不干扰正常任务）
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│              Clean 模式对比 (验证 Guardian 对正常任务的影响)                  │")
    print("├───────────────┬──────────────────┬──────────────────┤")
    print("│  Attack Type  │  无防御 (均分)    │  Guardian (均分)  │")
    print("├───────────────┼──────────────────┼──────────────────┤")
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"clean": []})["clean"]
        g_data = guardian["by_attack"].get(atk, {"clean": []})["clean"]
        b_s = avg_score(b_data)
        g_s = avg_score(g_data)
        print(f"│  {atk:13s}│  {b_s:.3f}           │  {g_s:.3f}           │")
    print("└───────────────┴──────────────────┴──────────────────┘")
    print()

    # 关键发现
    print("=" * 80)
    print("  关键发现")
    print("=" * 80)
    improve = g_resist/g_atk_n*100 - b_resist/b_atk_n*100
    print(f"  1. Guardian 将攻击抵抗率从 {b_resist/b_atk_n*100:.1f}% 提升至 {g_resist/g_atk_n*100:.1f}% (+{improve:.1f}pp)")
    print(f"  2. Guardian 将攻击模式平均分从 {b_atk_score:.3f} 提升至 {g_atk_score:.3f} (+{g_atk_score-b_atk_score:.3f})")
    print(f"  3. Clean 模式: 无防御 {b_clean_score:.3f} vs Guardian {g_clean_score:.3f}")
    if g_clean_score >= b_clean_score - 0.01:
        print(f"     → Guardian 对正常任务性能无负面影响 ✓")
    else:
        print(f"     → Guardian 对正常任务有轻微性能影响 ({g_clean_score-b_clean_score:.3f})")

    # 各攻击类型提升
    print(f"\n  按攻击类型抵抗率提升:")
    for atk in attacks_sorted:
        b_data = baseline["by_attack"].get(atk, {"attack": []})["attack"]
        g_data = guardian["by_attack"].get(atk, {"attack": []})["attack"]
        b_r, b_n = resist_rate(b_data)
        g_r, g_n = resist_rate(g_data)
        delta = g_r/max(g_n,1)*100 - b_r/max(b_n,1)*100
        print(f"    {atk:15s}: {b_r/max(b_n,1)*100:5.1f}% → {g_r/max(g_n,1)*100:5.1f}%  (+{delta:.1f}pp)")

    # 仍然失败的案例
    guardian_fails = [r for r in guardian["attack"] if not r.get("attack_resisted", False)]
    if guardian_fails:
        print(f"\n  Guardian 仍未防御的案例 ({len(guardian_fails)}个):")
        for r in guardian_fails:
            mal = r.get("malicious_tools_called", [])
            print(f"    - {r['id']} ({r['attack_type']}/{r['scenario']}): "
                  f"ARIA={r['aria_score']}, reason={r.get('reasoning','')[:60]}")
    print()


if __name__ == "__main__":
    baseline_path = "tamas_baseline_no_defense.jsonl"
    guardian_path = "tamas_with_guardian.jsonl"

    baseline_data = analyze(load_results(baseline_path), "无防御")
    guardian_data = analyze(load_results(guardian_path), "Guardian")

    print_comparison(baseline_data, guardian_data)
