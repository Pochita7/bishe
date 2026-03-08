"""Compare v1 and v2 GAIA results to see what changed."""
import json

def load_results(path):
    results = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            results[r['task_id']] = r
    return results

v1 = load_results('gaia_results.jsonl')
v2 = load_results('gaia_results_v2.jsonl')

print(f"V1: {sum(1 for r in v1.values() if r['is_correct'])}/{len(v1)} correct")
print(f"V2: {sum(1 for r in v2.values() if r['is_correct'])}/{len(v2)} correct")
print()

# New correct (wrong in v1, correct in v2)
new_correct = []
for tid in v2:
    if tid in v1 and not v1[tid]['is_correct'] and v2[tid]['is_correct']:
        new_correct.append(tid)

# Regressed (correct in v1, wrong in v2) 
regressed = []
for tid in v1:
    if tid in v2 and v1[tid]['is_correct'] and not v2[tid]['is_correct']:
        regressed.append(tid)

print(f"=== NEW CORRECT (v1 wrong -> v2 correct): {len(new_correct)} ===")
for tid in new_correct:
    q = v2[tid].get('question', '')[:100]
    pred = v2[tid].get('predicted_answer', '')
    gt = v2[tid].get('ground_truth', '')
    print(f"  {tid}")
    print(f"    Q: {q}")
    print(f"    Predicted: {pred}, GT: {gt}")
    print()

print(f"=== REGRESSED (v1 correct -> v2 wrong): {len(regressed)} ===")
for tid in regressed:
    q = v1[tid].get('question', '')[:100]
    v1_pred = v1[tid].get('predicted_answer', '')
    v2_pred = v2[tid].get('predicted_answer', '')
    gt = v1[tid].get('ground_truth', '')
    print(f"  {tid}")
    print(f"    Q: {q}")
    print(f"    V1 pred: {v1_pred}, V2 pred: {v2_pred}, GT: {gt}")
    print()

# Still wrong analysis
still_wrong = [tid for tid in v2 if tid in v1 and not v1[tid]['is_correct'] and not v2[tid]['is_correct']]
print(f"=== STILL WRONG: {len(still_wrong)} ===")
no_answer_count = 0
wrong_answer_count = 0
for tid in still_wrong:
    pred = v2[tid].get('predicted_answer', '')
    gt = v2[tid].get('ground_truth', '')
    q = v2[tid].get('question', '')[:80]
    if pred == 'NO_ANSWER':
        no_answer_count += 1
        reason = 'NO_ANSWER'
    else:
        wrong_answer_count += 1
        reason = f"wrong: {pred} vs {gt}"
    print(f"  {tid}: {reason}")
    print(f"    Q: {q}")

print(f"\n  Still NO_ANSWER: {no_answer_count}")
print(f"  Still Wrong Answer: {wrong_answer_count}")
