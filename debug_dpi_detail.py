"""分析 DPI 攻击防御前后的 task_completed / attack_resisted / ARIA 分布"""
import json
from collections import Counter

for label, path in [
    ("Baseline", "benchmark_tamas_baseline_v2.jsonl"),
    ("Defended", "benchmark_tamas_defended_v2.jsonl"),
]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    dpi_attack = [r for r in rows if r.get("attack_type") == "DPI" and r.get("mode") == "attack"]
    dpi_clean  = [r for r in rows if r.get("attack_type") == "DPI" and r.get("mode") == "clean"]

    print(f"\n{'='*60}")
    print(f"  {label} — DPI Attack mode ({len(dpi_attack)} tasks)")
    print(f"{'='*60}")

    # ARIA 分布
    aria = Counter(r["aria_score"] for r in dpi_attack)
    print(f"  ARIA 分布: {dict(sorted(aria.items()))}")

    # task_completed / attack_resisted
    tc = sum(1 for r in dpi_attack if r.get("task_completed"))
    ar = sum(1 for r in dpi_attack if r.get("attack_resisted"))
    print(f"  task_completed: {tc}/{len(dpi_attack)} ({tc/len(dpi_attack)*100:.1f}%)")
    print(f"  attack_resisted: {ar}/{len(dpi_attack)} ({ar/len(dpi_attack)*100:.1f}%)")

    # 恶意工具调用
    mal = sum(1 for r in dpi_attack if r.get("malicious_tools_called"))
    print(f"  有恶意工具调用: {mal}/{len(dpi_attack)}")

    # 各 ARIA 细分
    for aria_val in sorted(set(r["aria_score"] for r in dpi_attack)):
        subset = [r for r in dpi_attack if r["aria_score"] == aria_val]
        tc_s = sum(1 for r in subset if r.get("task_completed"))
        ar_s = sum(1 for r in subset if r.get("attack_resisted"))
        print(f"\n  [{aria_val}] ({len(subset)} tasks):")
        print(f"    task_completed={tc_s}, attack_resisted={ar_s}")
        # 看 reasoning
        reasons = Counter(r.get("reasoning", "?") for r in subset)
        for reason, cnt in reasons.most_common(5):
            print(f"    reason: {reason} (×{cnt})")

    # Clean 模式对比
    print(f"\n  --- DPI Clean mode ({len(dpi_clean)} tasks) ---")
    tc_c = sum(1 for r in dpi_clean if r.get("task_completed"))
    aria_c = Counter(r["aria_score"] for r in dpi_clean)
    print(f"  task_completed: {tc_c}/{len(dpi_clean)}")
    print(f"  ARIA: {dict(sorted(aria_c.items()))}")

    # 展示几个具体的 ARIA_3 和 ARIA_4 案例
    print(f"\n  --- 典型案例 ---")
    for aria_val in ["ARIA_3", "ARIA_4", "ARIA_1"]:
        cases = [r for r in dpi_attack if r["aria_score"] == aria_val]
        for c in cases[:2]:
            print(f"\n  [{aria_val}] {c.get('task_id', '?')}")
            print(f"    task_completed={c.get('task_completed')}, attack_resisted={c.get('attack_resisted')}")
            print(f"    tools_called: {c.get('tools_called', [])}")
            print(f"    malicious_tools: {c.get('malicious_tools_called', [])}")
            print(f"    tool_coverage: {c.get('tool_coverage', 0):.1%}")
            print(f"    reasoning: {c.get('reasoning', '?')}")
            # 输出前 300 字符
            out = c.get("output", "")[:300]
            print(f"    output[:300]: {out}")
