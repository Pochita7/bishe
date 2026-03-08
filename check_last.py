import json

with open('gaia_results.jsonl', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

r = json.loads(lines[-1])
print(f"Q: {r['question'][:80]}...")
print(f"Pred: {r['predicted_answer']}")
print(f"GT: {r['ground_truth']}")
print(f"OK: {r['is_correct']}")
