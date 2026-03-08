import json

print("=== GAIA Baseline ===")
bl = [json.loads(l) for l in open("benchmark_gaia_baseline.jsonl", encoding="utf-8") if l.strip()]
correct = sum(1 for r in bl if r["is_correct"])
print(f"  Accuracy: {correct}/{len(bl)} = {correct/len(bl)*100:.1f}%")
avg_t = sum(r["elapsed"] for r in bl)/len(bl)
avg_tok = sum(r.get("total_tokens",0) for r in bl)/len(bl)
avg_tl = sum(r.get("total_tool_calls",0) for r in bl)/len(bl)
print(f"  Avg time: {avg_t:.1f}s")
print(f"  Avg tokens: {avg_tok:.0f}")
print(f"  Avg tools: {avg_tl:.1f}")

print()
print("=== GAIA Defended ===")
df = [json.loads(l) for l in open("benchmark_gaia_defended.jsonl", encoding="utf-8") if l.strip()]
correct2 = sum(1 for r in df if r["is_correct"])
print(f"  Accuracy: {correct2}/{len(df)} = {correct2/len(df)*100:.1f}%")
avg_t2 = sum(r["elapsed"] for r in df)/len(df)
avg_tok2 = sum(r.get("total_tokens",0) for r in df)/len(df)
avg_tl2 = sum(r.get("total_tool_calls",0) for r in df)/len(df)
print(f"  Avg time: {avg_t2:.1f}s")
print(f"  Avg tokens: {avg_tok2:.0f}")
print(f"  Avg tools: {avg_tl2:.1f}")

print()
bl_acc = correct/len(bl)*100
df_acc = correct2/len(df)*100
print("=== GAIA 对比 ===")
print(f"  Accuracy: {bl_acc:.1f}% -> {df_acc:.1f}% (delta {df_acc-bl_acc:+.1f}pp)")
print(f"  Avg time: {avg_t:.1f}s -> {avg_t2:.1f}s (delta {avg_t2-avg_t:+.1f}s)")
print(f"  Avg tokens: {avg_tok:.0f} -> {avg_tok2:.0f} (delta {avg_tok2-avg_tok:+.0f})")
print(f"  Avg tools: {avg_tl:.1f} -> {avg_tl2:.1f}")
