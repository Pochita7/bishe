"""分析 GAIA 测试结果的失败模式"""
import json

results = []
with open('gaia_results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        results.append(json.loads(line.strip()))

tasks = []
with open('gaia_data/gaia_level1.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        tasks.append(json.loads(line.strip()))
task_map = {t['task_id']: t for t in tasks}

# Categorize failures
no_answer = []
format_close = []
wrong = []
correct = [r for r in results if r['is_correct']]

for r in results:
    if r['is_correct']:
        continue
    pred = r['predicted_answer'].strip().lower()
    gt = r['ground_truth'].strip().lower()
    if pred in ('no_answer', 'unable to determine', ''):
        no_answer.append(r)
    elif pred.replace(' ', '') == gt.replace(' ', '') or pred.replace(',', ', ') == gt:
        format_close.append(r)
    else:
        wrong.append(r)

print(f'=== Failure Analysis ===')
print(f'Correct: {len(correct)}/53 = {len(correct)/53*100:.1f}%')
print(f'NO_ANSWER/Unable: {len(no_answer)}')
print(f'Format-close (fixable): {len(format_close)}')
print(f'Wrong answer: {len(wrong)}')
print()

print('=== NO_ANSWER tasks ===')
for r in no_answer:
    t = task_map.get(r['task_id'], {})
    fn = t.get('file_name', '')
    q = r['question'][:100]
    gt = r['ground_truth']
    tag = fn.split('.')[-1] if fn else 'no-file'
    print(f'  [{tag}] GT={gt!r}')
    print(f'    Q: {q}')
    print()

print('=== Format-close tasks (should be correct) ===')
for r in format_close:
    print(f'  Pred={r["predicted_answer"]!r} GT={r["ground_truth"]!r}')

print()
print('=== Image/Audio/Video tasks ===')
for t in tasks:
    fn = t.get('file_name', '')
    ext = fn.split('.')[-1].lower() if fn else ''
    q = t.get('Question', '')
    has_url = 'youtube.com' in q or 'youtu.be' in q
    if ext in ('png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4') or has_url:
        tid = t['task_id']
        res = next((r for r in results if r['task_id'] == tid), None)
        status = 'OK' if res and res['is_correct'] else 'FAIL'
        pred = res['predicted_answer'] if res else '?'
        gt = t.get('Final answer', '')
        marker = 'IMG' if ext in ('png', 'jpg', 'jpeg') else ('AUD' if ext in ('mp3', 'wav') else 'VID')
        print(f'  [{status}] [{marker}] {fn or "URL"} Pred={pred!r} GT={gt!r}')
        print(f'    Q: {q[:100]}')
        print()

print('=== Wrong answers (sample) ===')
for r in wrong[:15]:
    t = task_map.get(r['task_id'], {})
    fn = t.get('file_name', '')
    tag = fn.split('.')[-1] if fn else 'no-file'
    print(f'  [{tag}] Pred={r["predicted_answer"]!r} GT={r["ground_truth"]!r}')
    print(f'    Q: {r["question"][:100]}')
    print()
