import json

results = {}
with open('gaia_results_v2.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        results[r['task_id']] = r

tasks = []
with open('gaia_data/gaia_level1.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        t = json.loads(line)
        t['_idx'] = i
        tasks.append(t)

print("Previously WRONG tasks:")
for t in tasks:
    tid = t['task_id']
    r = results.get(tid, {})
    if not r.get('is_correct', True):
        q = t['Question'][:80]
        fn = t.get('file_name', '')
        gt = r.get('ground_truth', '')[:30]
        pred = r.get('predicted_answer', '')[:30]
        yt = 'youtube' in t['Question'].lower()
        ftype = fn.split('.')[-1] if fn else 'none'
        print(f"  idx={t['_idx']:2d} | {tid[:12]} | file={ftype:4s} | YT={'Y' if yt else 'N'} | pred={pred:30s} | gt={gt}")
