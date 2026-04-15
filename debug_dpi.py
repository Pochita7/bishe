"""分析 DPI 攻击的工具调用细节"""
import json
from collections import Counter

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

for label, path in [("Baseline", "benchmark_tamas_baseline_v2.jsonl"),
                     ("Defended", "benchmark_tamas_defended_v2.jsonl")]:
    data = load(path)
    dpi_atk = [r for r in data if r["attack_type"] == "DPI" and r["mode"] == "attack"]

    print(f"{'='*70}")
    print(f"  {label} - DPI Attack (n={len(dpi_atk)})")
    print(f"{'='*70}")

    # ARIA distribution
    aria = Counter(r["aria_score"] for r in dpi_atk)
    print(f"  ARIA: {dict(sorted(aria.items()))}")

    # Malicious tool calls
    mal_total = sum(len(r.get("malicious_tools_called", [])) for r in dpi_atk)
    print(f"  Malicious tool calls total: {mal_total}")

    # ARIA_4 cases detail
    a4 = [r for r in dpi_atk if r["aria_score"] == "ARIA_4"]
    print(f"\n  ARIA_4 cases ({len(a4)}):")
    for r in a4:
        print(f"    id: {r['id']}")
        print(f"      tools_called: {r.get('tools_called', [])}")
        print(f"      malicious_tools_called: {r.get('malicious_tools_called', [])}")
        print(f"      reasoning: {r.get('reasoning', '')}")

    # All unique tools
    all_tools = set()
    for r in dpi_atk:
        for t in r.get("tools_called", []):
            all_tools.add(t)
    print(f"\n  All unique tools called: {sorted(all_tools)}")

    # Non-NOTA cases summary
    print(f"\n  Non-NOTA cases:")
    for r in dpi_atk:
        if r["aria_score"] != "NOTA":
            tc = r.get("tools_called", [])
            mc = r.get("malicious_tools_called", [])
            print(f"    {r['id']}: ARIA={r['aria_score']} | tools={tc} | mal={mc}")
            print(f"      reasoning: {r['reasoning'][:100]}")

    print()
