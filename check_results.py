import json

results = []
with open('gaia_results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

correct = sum(1 for r in results if r['is_correct'])
total = len(results)
print(f"Total: {total}, Correct: {correct}, Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
for i, r in enumerate(results):
    status = "O" if r["is_correct"] else "X"
    pred = r["predicted_answer"]
    gt = r["ground_truth"]
    q = r["question"][:70]
    print(f"  [{i}] {status} | Pred: {pred} | GT: {gt} | Q: {q}...")
