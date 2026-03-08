"""
全面分析 GAIA Level 1 最新测试结果
- 统计正确/错误/无答案
- 按错误类型分类
- 分析具体每题的出错原因
"""
import json
import os

RESULT_FILE = "gaia_results_v2.jsonl"
DATA_FILE = "gaia_data/gaia_level1.jsonl"

# 加载结果
results = {}
with open(RESULT_FILE, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        results[r['task_id']] = r

# 加载原始题目数据（获取完整问题文本和附件信息）
tasks = {}
with open(DATA_FILE, encoding='utf-8') as f:
    for line in f:
        t = json.loads(line)
        tasks[t['task_id']] = t

print(f"=" * 80)
print(f"GAIA Level 1 测试结果总结")
print(f"结果文件: {RESULT_FILE}")
print(f"总题数: {len(results)}")
print(f"=" * 80)

# 基本统计
correct = [r for r in results.values() if r['is_correct']]
wrong = [r for r in results.values() if not r['is_correct'] and r.get('predicted_answer', '') != 'NO_ANSWER']
no_answer = [r for r in results.values() if not r['is_correct'] and r.get('predicted_answer', '') == 'NO_ANSWER']

print(f"\n📊 基本统计:")
print(f"  ✅ 正确: {len(correct)}/{len(results)} ({len(correct)/len(results)*100:.1f}%)")
print(f"  ❌ 错误答案: {len(wrong)}/{len(results)} ({len(wrong)/len(results)*100:.1f}%)")
print(f"  ⏱️ 无答案: {len(no_answer)}/{len(results)} ({len(no_answer)/len(results)*100:.1f}%)")

# 按题目类型分析
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac'}
DOC_EXTS = {'.docx', '.doc'}
EXCEL_EXTS = {'.xlsx', '.xls'}
PPT_EXTS = {'.pptx', '.ppt'}
TEXT_EXTS = {'.txt', '.csv', '.json', '.py', '.jsonl'}

def classify_task(task):
    """根据附件类型和问题内容分类任务"""
    file_name = task.get('file_name', '')
    question = task.get('Question', '')
    categories = []
    
    if file_name:
        ext = os.path.splitext(file_name)[1].lower()
        if ext in IMAGE_EXTS:
            categories.append('图片分析')
        elif ext in AUDIO_EXTS:
            categories.append('音频处理')
        elif ext in DOC_EXTS:
            categories.append('Word文档')
        elif ext in EXCEL_EXTS:
            categories.append('Excel表格')
        elif ext in PPT_EXTS:
            categories.append('PPT演示')
        elif ext in TEXT_EXTS:
            categories.append('文本文件')
        else:
            categories.append(f'其他文件({ext})')
    
    if 'youtube.com' in question or 'youtu.be' in question:
        categories.append('YouTube视频')
    
    if not categories:
        # 判断是否需要搜索 vs 纯推理
        keywords_search = ['wikipedia', 'website', 'article', 'paper', 'journal', 'published', 'changelog']
        keywords_calc = ['calculate', 'how many', 'how long', 'what is the', 'what was the']
        
        q_lower = question.lower()
        if any(kw in q_lower for kw in keywords_search):
            categories.append('网络搜索')
        elif any(kw in q_lower for kw in keywords_calc):
            categories.append('知识+计算')
        else:
            categories.append('知识推理')
    
    return categories

# 分类统计
category_stats = {}
for tid, r in results.items():
    task = tasks.get(tid, {})
    cats = classify_task(task)
    for cat in cats:
        if cat not in category_stats:
            category_stats[cat] = {'correct': 0, 'wrong': 0, 'no_answer': 0, 'total': 0}
        category_stats[cat]['total'] += 1
        if r['is_correct']:
            category_stats[cat]['correct'] += 1
        elif r.get('predicted_answer', '') == 'NO_ANSWER':
            category_stats[cat]['no_answer'] += 1
        else:
            category_stats[cat]['wrong'] += 1

print(f"\n📋 按任务类型分类:")
for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['total'], reverse=True):
    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"  {cat}: {stats['correct']}/{stats['total']} 正确 ({acc:.0f}%) | 错误:{stats['wrong']} 无答案:{stats['no_answer']}")

# 错误原因分析
print(f"\n{'=' * 80}")
print(f"错误原因详细分析")
print(f"{'=' * 80}")

def analyze_error(result, task):
    """分析单个错误的原因"""
    pred = result.get('predicted_answer', '')
    gt = result.get('ground_truth', '')
    question = task.get('Question', '')
    file_name = task.get('file_name', '')
    
    reasons = []
    
    if pred == 'NO_ANSWER':
        reasons.append('执行超时/无法获取结果')
        return reasons
    
    # 检查是否是格式问题
    pred_norm = pred.lower().strip().replace(', ', ',').replace('  ', ' ')
    gt_norm = gt.lower().strip().replace(', ', ',').replace('  ', ' ')
    if pred_norm == gt_norm:
        reasons.append('格式差异（大小写/空格）')
        return reasons
    
    # 检查是否部分匹配
    if gt_norm in pred_norm or pred_norm in gt_norm:
        reasons.append('答案包含多余内容或不完整')
        return reasons
    
    # 检查是否是数字近似
    try:
        p, g = float(pred), float(gt)
        if abs(p - g) / max(abs(g), 1) < 0.2:
            reasons.append(f'数字近似但不精确 (预测{pred} vs 真实{gt})')
            return reasons
    except ValueError:
        pass
    
    # YouTube 视频相关
    if 'youtube.com' in question or 'youtu.be' in question:
        reasons.append('无法获取YouTube视频内容')
        return reasons
    
    # 图片相关
    if file_name:
        ext = os.path.splitext(file_name)[1].lower()
        if ext in IMAGE_EXTS:
            reasons.append('视觉模型分析不准确')
            return reasons
        if ext in AUDIO_EXTS:
            reasons.append('音频转录/理解不准确')
            return reasons
    
    # 特定网站内容
    specific_sites = ['wikipedia', 'cornell law', 'bbc', 'scikit-learn', 'github']
    if any(s in question.lower() for s in specific_sites):
        reasons.append('未能正确获取/解析特定网页内容')
        return reasons
    
    # 复杂推理
    if len(question) > 300 or 'riddle' in question.lower() or 'game' in question.lower():
        reasons.append('复杂推理/逻辑题理解不足')
        return reasons
    
    reasons.append('搜索结果不准确或知识不足')
    return reasons

# 错误分类汇总
error_categories = {}
print(f"\n--- ❌ 错误答案详情 ({len(wrong)}题) ---\n")
for i, r in enumerate(wrong, 1):
    tid = r['task_id']
    task = tasks.get(tid, {})
    question = task.get('Question', '')
    file_name = task.get('file_name', '')
    pred = r.get('predicted_answer', '')
    gt = r.get('ground_truth', '')
    reasons = analyze_error(r, task)
    
    for reason in reasons:
        error_categories[reason] = error_categories.get(reason, 0) + 1
    
    cats = classify_task(task)
    print(f"[{i}] Task: {tid[:12]}...")
    print(f"    类型: {', '.join(cats)}")
    print(f"    问题: {question[:120]}{'...' if len(question)>120 else ''}")
    if file_name:
        print(f"    附件: {file_name}")
    print(f"    预测: {pred[:100]}")
    print(f"    正确: {gt[:100]}")
    print(f"    错因: {'; '.join(reasons)}")
    print()

if no_answer:
    print(f"\n--- ⏱️ 无答案详情 ({len(no_answer)}题) ---\n")
    for i, r in enumerate(no_answer, 1):
        tid = r['task_id']
        task = tasks.get(tid, {})
        question = task.get('Question', '')
        print(f"[{i}] {tid[:12]}... : {question[:100]}...")
        print()

print(f"\n{'=' * 80}")
print(f"错误原因分布汇总")
print(f"{'=' * 80}")
for reason, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * count
    print(f"  {reason}: {count} {bar}")

# 正确答案概览
print(f"\n{'=' * 80}")
print(f"✅ 正确回答的题目 ({len(correct)}题)")
print(f"{'=' * 80}")
for i, r in enumerate(correct, 1):
    tid = r['task_id']
    task = tasks.get(tid, {})
    question = task.get('Question', '')[:80]
    pred = r.get('predicted_answer', '')
    cats = classify_task(task)
    print(f"  [{i}] {', '.join(cats)} | 答案: {pred} | 问题: {question}...")

print(f"\n{'=' * 80}")
print(f"总结")
print(f"{'=' * 80}")
print(f"正确率: {len(correct)}/{len(results)} = {len(correct)/len(results)*100:.1f}%")
print(f"主要失败原因:")
for reason, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  - {reason}: {count}题")
print(f"\n改进建议:")
print(f"  1. 增加工作流总超时时间（当前180s不够，很多Executor超时）")
print(f"  2. 优化搜索策略，减少无效搜索轮次")  
print(f"  3. 改进视觉模型使用（棋盘、分数图片等需要更精细的prompt）")
print(f"  4. 对YouTube视频类题目考虑使用字幕API")
print(f"  5. 增强Verifier的格式化能力，避免修改正确答案")
