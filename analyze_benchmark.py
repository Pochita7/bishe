"""分析最近一次 benchmark 测试数据 (2026-03-01~02)"""
import json, statistics, collections

def load(f):
    with open(f, encoding='utf-8') as fp:
        return [json.loads(l) for l in fp]

# ==================== GAIA ====================
gb = load('benchmark_gaia_baseline.jsonl')
gd = load('benchmark_gaia_defended.jsonl')

gb_correct = sum(1 for t in gb if t.get('is_correct'))
gd_correct = sum(1 for t in gd if t.get('is_correct'))

print('=' * 65)
print('  GAIA BENCHMARK ANALYSIS (53 Level-1 Tasks)')
print('=' * 65)
print(f'  Baseline Accuracy : {gb_correct}/{len(gb)} = {gb_correct/len(gb)*100:.1f}%')
print(f'  Defended Accuracy : {gd_correct}/{len(gd)} = {gd_correct/len(gd)*100:.1f}%')
delta = gd_correct - gb_correct
print(f'  Delta             : {delta:+d} tasks ({delta/len(gb)*100:+.1f}pp)')

# Timing
gb_times = [t['elapsed'] for t in gb]
gd_times = [t['elapsed'] for t in gd]
print(f'\n  [Latency]')
print(f'  Baseline avg/med  : {statistics.mean(gb_times):.1f}s / {statistics.median(gb_times):.1f}s')
print(f'  Defended avg/med  : {statistics.mean(gd_times):.1f}s / {statistics.median(gd_times):.1f}s')
overhead_pct = (statistics.mean(gd_times) - statistics.mean(gb_times)) / statistics.mean(gb_times) * 100
print(f'  Time overhead     : {overhead_pct:+.1f}%')

# Token usage
gb_tok = [t['total_tokens'] for t in gb]
gd_tok = [t['total_tokens'] for t in gd]
tok_overhead = (statistics.mean(gd_tok) - statistics.mean(gb_tok)) / statistics.mean(gb_tok) * 100
print(f'\n  [Token Usage]')
print(f'  Baseline avg      : {statistics.mean(gb_tok):.0f} tokens')
print(f'  Defended avg      : {statistics.mean(gd_tok):.0f} tokens')
print(f'  Token overhead    : {tok_overhead:+.1f}%')

# Tool calls
gb_tc = [t['total_tool_calls'] for t in gb]
gd_tc = [t['total_tool_calls'] for t in gd]
print(f'\n  [Tool Calls]')
print(f'  Baseline avg      : {statistics.mean(gb_tc):.1f}')
print(f'  Defended avg      : {statistics.mean(gd_tc):.1f}')

# Agent calls / rounds
gb_ac = [t['total_agent_calls'] for t in gb]
gd_ac = [t['total_agent_calls'] for t in gd]
gb_rnd = [t['rounds'] for t in gb]
gd_rnd = [t['rounds'] for t in gd]
print(f'\n  [Orchestration]')
print(f'  Baseline avg rounds/agents : {statistics.mean(gb_rnd):.1f} / {statistics.mean(gb_ac):.1f}')
print(f'  Defended avg rounds/agents : {statistics.mean(gd_rnd):.1f} / {statistics.mean(gd_ac):.1f}')

# Per-task changes
print(f'\n  [Task-level Changes]')
gained = []
lost = []
for b, d in zip(gb, gd):
    if not b['is_correct'] and d['is_correct']:
        gained.append(b['task_id'][:16])
    elif b['is_correct'] and not d['is_correct']:
        lost.append(b['task_id'][:16])
print(f'  Gained (wrong->correct): {len(gained)}')
for t in gained:
    print(f'    + {t}...')
print(f'  Lost (correct->wrong)  : {len(lost)}')
for t in lost:
    print(f'    - {t}...')


# ==================== TAMAS ====================
tb = load('benchmark_tamas_baseline.jsonl')
td = load('benchmark_tamas_defended.jsonl')

print('\n' + '=' * 65)
print('  TAMAS BENCHMARK ANALYSIS (600 Tasks: 6 attacks x 5 domains x 10 each x 2 modes)')
print('=' * 65)

# Separate clean vs attack
tb_clean = [t for t in tb if t.get('mode') == 'clean']
tb_attack = [t for t in tb if t.get('mode') == 'attack']
td_clean = [t for t in td if t.get('mode') == 'clean']
td_attack = [t for t in td if t.get('mode') == 'attack']

print(f'  Total tasks       : {len(tb)} (clean={len(tb_clean)}, attack={len(tb_attack)})')

# Overall metrics
def aria_stats(tasks):
    scores = [t['score'] for t in tasks]
    completed = sum(1 for t in tasks if t.get('task_completed'))
    resisted = sum(1 for t in tasks if t.get('attack_resisted'))
    return {
        'avg_score': statistics.mean(scores) if scores else 0,
        'completed': completed,
        'completed_pct': completed / len(tasks) * 100 if tasks else 0,
        'resisted': resisted,
        'resisted_pct': resisted / len(tasks) * 100 if tasks else 0,
    }

print(f'\n  === Overall ===')
for label, data in [('Baseline', tb), ('Defended', td)]:
    s = aria_stats(data)
    print(f'  {label:10s}: avg_score={s["avg_score"]:.3f}  completed={s["completed_pct"]:.1f}%  resisted={s["resisted_pct"]:.1f}%')

# Clean task performance
print(f'\n  === Clean Tasks (no attack, task completion) ===')
for label, data in [('Baseline', tb_clean), ('Defended', td_clean)]:
    s = aria_stats(data)
    print(f'  {label:10s}: completed={s["completed"]}/{len(data)} ({s["completed_pct"]:.1f}%)  avg_score={s["avg_score"]:.3f}')

clean_delta = aria_stats(td_clean)['completed_pct'] - aria_stats(tb_clean)['completed_pct']
print(f'  Clean degradation : {clean_delta:+.1f}pp')

# Attack resistance
print(f'\n  === Attack Tasks (attack resistance) ===')
for label, data in [('Baseline', tb_attack), ('Defended', td_attack)]:
    s = aria_stats(data)
    print(f'  {label:10s}: resisted={s["resisted"]}/{len(data)} ({s["resisted_pct"]:.1f}%)  avg_score={s["avg_score"]:.3f}')

atk_delta = aria_stats(td_attack)['resisted_pct'] - aria_stats(tb_attack)['resisted_pct']
print(f'  Resistance gain   : {atk_delta:+.1f}pp')

# Per attack type breakdown
print(f'\n  === Per Attack Type ===')
attack_types = sorted(set(t['attack_type'] for t in tb))
print(f'  {"Attack":<16s} | {"Baseline Resist":>22s} | {"Defended Resist":>22s} | {"Delta":>8s}')
print(f'  {"-"*16} | {"-"*22} | {"-"*22} | {"-"*8}')
for at in attack_types:
    b_atk = [t for t in tb_attack if t['attack_type'] == at]
    d_atk = [t for t in td_attack if t['attack_type'] == at]
    b_res = sum(1 for t in b_atk if t['attack_resisted'])
    d_res = sum(1 for t in d_atk if t['attack_resisted'])
    b_r = b_res / len(b_atk) * 100 if b_atk else 0
    d_r = d_res / len(d_atk) * 100 if d_atk else 0
    print(f'  {at:<16s} | {b_res:>3d}/{len(b_atk):<3d} ({b_r:>5.1f}%)    | {d_res:>3d}/{len(d_atk):<3d} ({d_r:>5.1f}%)    | {d_r-b_r:>+6.1f}pp')

# Per scenario breakdown (clean)
print(f'\n  === Per Scenario (Clean Task Completion) ===')
scenarios = sorted(set(t['scenario'] for t in tb))
print(f'  {"Scenario":<14s} | {"Baseline":>10s} | {"Defended":>10s} | {"Delta":>8s}')
print(f'  {"-"*14} | {"-"*10} | {"-"*10} | {"-"*8}')
for sc in scenarios:
    b_sc = [t for t in tb_clean if t['scenario'] == sc]
    d_sc = [t for t in td_clean if t['scenario'] == sc]
    b_c = sum(1 for t in b_sc if t['task_completed']) / len(b_sc) * 100 if b_sc else 0
    d_c = sum(1 for t in d_sc if t['task_completed']) / len(d_sc) * 100 if d_sc else 0
    print(f'  {sc:<14s} | {b_c:>8.1f}% | {d_c:>8.1f}% | {d_c-b_c:>+6.1f}pp')

# Timing
tb_times = [t['elapsed_time'] for t in tb]
td_times = [t['elapsed_time'] for t in td]
t_overhead = (statistics.mean(td_times) - statistics.mean(tb_times)) / statistics.mean(tb_times) * 100
print(f'\n  === Performance Overhead ===')
print(f'  Baseline avg time : {statistics.mean(tb_times):.1f}s')
print(f'  Defended avg time : {statistics.mean(td_times):.1f}s')
print(f'  Time overhead     : {t_overhead:+.1f}%')

tb_tok = [t['total_tokens'] for t in tb]
td_tok = [t['total_tokens'] for t in td]
tok_o = (statistics.mean(td_tok) - statistics.mean(tb_tok)) / statistics.mean(tb_tok) * 100
print(f'  Baseline avg tok  : {statistics.mean(tb_tok):.0f}')
print(f'  Defended avg tok  : {statistics.mean(td_tok):.0f}')
print(f'  Token overhead    : {tok_o:+.1f}%')

# Tool call analysis
print(f'\n  === Malicious Tool Calls ===')
tb_mal = sum(len(t.get('malicious_tools_called', [])) for t in tb_attack)
td_mal = sum(len(t.get('malicious_tools_called', [])) for t in td_attack)
print(f'  Baseline malicious calls: {tb_mal}')
print(f'  Defended malicious calls: {td_mal}')
if tb_mal > 0:
    print(f'  Reduction: {(tb_mal - td_mal) / tb_mal * 100:.1f}%')

# Malicious tool breakdown
tb_mal_tools = collections.Counter()
td_mal_tools = collections.Counter()
for t in tb_attack:
    for m in t.get('malicious_tools_called', []):
        tb_mal_tools[m] += 1
for t in td_attack:
    for m in t.get('malicious_tools_called', []):
        td_mal_tools[m] += 1

if tb_mal_tools:
    print(f'\n  Baseline top malicious tools:')
    for tool, cnt in tb_mal_tools.most_common(10):
        print(f'    {tool}: {cnt}')
if td_mal_tools:
    print(f'\n  Defended top malicious tools:')
    for tool, cnt in td_mal_tools.most_common(10):
        print(f'    {tool}: {cnt}')

# ARIA score distribution
print(f'\n  === ARIA Score Distribution (Attack tasks) ===')
for label, data in [('Baseline', tb_attack), ('Defended', td_attack)]:
    aria_dist = collections.Counter(t.get('aria_score', 'N/A') for t in data)
    parts = ', '.join(f'{k}={v}' for k, v in sorted(aria_dist.items()))
    print(f'  {label:10s}: {parts}')

print('\n' + '=' * 65)
print('  ANALYSIS COMPLETE')
print('=' * 65)
