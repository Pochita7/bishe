"""
核心求解逻辑 - 读取 GAIA 任务并调度多智能体系统求解
适配 Microsoft Agent Framework (agent-framework) 异步 API
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from mas.compat import ensure_utf8_stdio

ensure_utf8_stdio()

import asyncio
import json
import re
import os
import string
from typing import Optional

from gaia_solver.config import GAIA_LEVEL1_PATH, GAIA_ATTACHMENTS_DIR
from gaia_solver.agents import create_team
from gaia_solver import tools


def load_gaia_tasks(jsonl_path: str = GAIA_LEVEL1_PATH):
    """加载 GAIA Level 1 任务数据"""
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_task_prompt(task: dict) -> str:
    """
    根据 GAIA 任务构建精简 prompt。
    Token 优化：只在文件较小时预读，大文件留给工具读取。
    """
    question = task.get("Question", "")
    file_name = task.get("file_name", "")

    prompt_parts = [
        f"**Question:** {question}",
        (
            "\n**Answer format:** Return the exact value requested by the question. "
            "If it asks for how many thousand/million/billion units, return the number "
            "of those scaled units, not the raw unit count."
        ),
    ]

    if file_name:
        ext = file_name.split(".")[-1].lower()

        if ext in ("txt", "py", "csv", "json", "md"):
            content = tools.read_text_file(file_name)
            # 小文件直接嵌入，大文件只提到有附件
            if len(content) <= 2000:
                prompt_parts.append(f"\n**File ({file_name}):**\n```\n{content}\n```")
            else:
                prompt_parts.append(f"\n**Attachment:** {file_name} ({len(content)} chars, use read_text_file to read)")
        elif ext in ("xlsx", "xls"):
            content = tools.read_excel_file(file_name)
            if len(content) <= 2000:
                prompt_parts.append(f"\n**Excel ({file_name}):**\n{content}")
            else:
                prompt_parts.append(f"\n**Attachment:** {file_name} (Excel, use read_excel_file)")
        elif ext == "docx":
            content = tools.read_docx_file(file_name)
            if len(content) <= 2000:
                prompt_parts.append(f"\n**Doc ({file_name}):**\n{content}")
            else:
                prompt_parts.append(f"\n**Attachment:** {file_name} (Word, {len(content)} chars, use read_docx_file)")
        elif ext == "pptx":
            content = tools.read_pptx_file(file_name)
            if len(content) <= 2000:
                prompt_parts.append(f"\n**PPT ({file_name}):**\n{content}")
            else:
                prompt_parts.append(f"\n**Attachment:** {file_name} (PPT, use read_pptx_file)")
        elif ext in ("png", "jpg", "jpeg", "gif", "webp"):
            prompt_parts.append(f"\n**[IMAGE]** {file_name} - use analyze_image('{file_name}', '<question>')")
        elif ext in ("mp3", "wav", "m4a", "ogg"):
            prompt_parts.append(f"\n**[AUDIO]** {file_name} - use transcribe_audio('{file_name}')")

    return "\n".join(prompt_parts)


def _get_message_text(msg) -> str:
    """从 agent-framework Message 中提取文本内容"""
    if hasattr(msg, "text") and msg.text:
        return msg.text
    if hasattr(msg, "contents") and msg.contents:
        parts = []
        for c in msg.contents:
            if hasattr(c, "text") and c.text:
                parts.append(c.text)
            elif hasattr(c, "value") and c.value:
                parts.append(str(c.value))
        return " ".join(parts) if parts else ""
    if hasattr(msg, "content"):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    if isinstance(msg, dict):
        return msg.get("text", "") or msg.get("content", "")
    return ""


def extract_final_answer(messages) -> Optional[str]:
    """从聊天记录中提取 FINAL_ANSWER（兼容 agent-framework Message）
    优先提取 Verifier 的答案，其次提取任意角色的答案
    """
    # 第一遍：优先找 Verifier 的 FINAL_ANSWER
    for msg in reversed(messages):
        author = getattr(msg, "author_name", "") or ""
        content = _get_message_text(msg)
        if not content:
            continue
        if author == "Verifier":
            match = re.search(r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = answer.strip("`").strip("*").strip()
                return answer

    # 第二遍：找任意角色的 FINAL_ANSWER
    for msg in reversed(messages):
        content = _get_message_text(msg)
        if not content:
            continue
        match = re.search(r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.strip("`").strip("*").strip()
            return answer
    return None


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})+|\d+)(?:\.\d+)?")


def _single_numeric_value(text: str) -> Optional[float]:
    """Return the only numeric token in an answer, if there is exactly one."""
    tokens = _NUMERIC_TOKEN_RE.findall(text.replace("−", "-"))
    if len(tokens) != 1:
        return None
    try:
        return float(tokens[0].replace(",", ""))
    except ValueError:
        return None


def _numbers_match(predicted: float, expected: float) -> bool:
    if abs(predicted - expected) < 1e-6:
        return True
    if abs(round(predicted) - round(expected)) < 1e-6:
        return True
    return expected != 0 and abs(predicted - expected) / abs(expected) < 0.01


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """比较预测答案和标准答案（增强版，处理格式差异）"""
    if not predicted or not ground_truth:
        return False

    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()

    if pred == gt:
        return True

    # Numeric answers must be judged numerically before substring matching.
    # Otherwise "17" would incorrectly match "17000".
    pred_num = _single_numeric_value(pred)
    gt_num = _single_numeric_value(gt)
    if gt_num is not None and pred_num is not None:
        return _numbers_match(pred_num, gt_num)
    if gt_num is not None and pred_num is None and _NUMERIC_TOKEN_RE.search(pred):
        return False

    # 去除标点后比较
    pred_clean = pred.translate(str.maketrans("", "", string.punctuation))
    gt_clean = gt.translate(str.maketrans("", "", string.punctuation))
    if pred_clean == gt_clean:
        return True

    # 规范化空格和逗号: "b,e" == "b, e"
    import re
    pred_norm = re.sub(r'\s*,\s*', ', ', pred).strip()
    gt_norm = re.sub(r'\s*,\s*', ', ', gt).strip()
    if pred_norm == gt_norm:
        return True
    
    # 去除所有空格后比较 (处理列表格式差异)
    if pred.replace(' ', '') == gt.replace(' ', ''):
        return True

    # 去除标点+空格后比较 (处理分词差异, e.g. "THESE A GULL" vs "The seagull")
    if pred_clean.replace(' ', '') == gt_clean.replace(' ', ''):
        return True

    # ​列表元素匹配: 如果两个都是逗号分隔列表，比较排序后的元素集合
    if ',' in pred and ',' in gt:
        pred_items = sorted([x.strip() for x in pred.split(',') if x.strip()])
        gt_items = sorted([x.strip() for x in gt.split(',') if x.strip()])
        # 完全匹配
        if pred_items == gt_items:
            return True
        # 模糊匹配: 每个 gt 元素都能在某个 pred 元素中找到关键词
        if len(pred_items) == len(gt_items):
            matched = 0
            for g in gt_items:
                g_words = set(g.lower().split())
                for p in pred_items:
                    p_words = set(p.lower().split())
                    # 核心词重叠（去掉常见修饰词）
                    if g_words & p_words:
                        matched += 1
                        break
            if matched == len(gt_items):
                return True

    # 包含匹配: 标准答案包含在预测中
    if gt_clean in pred_clean:
        return True
    if pred_clean in gt_clean:
        return True

    # 数值比较
    try:
        pf = float(pred)
        gf = float(gt)
        if abs(pf - gf) < 1e-6:
            return True
        # 整数比较（四舍五入后相等）
        if abs(round(pf) - round(gf)) < 1e-6:
            return True
        # 相对误差 < 1%
        if gf != 0 and abs(pf - gf) / abs(gf) < 0.01:
            return True
    except ValueError:
        pass

    return False


async def solve_single_task_async(task: dict, max_turns: int = 15, verbose: bool = True) -> dict:
    """
    异步求解单个 GAIA 任务。

    Args:
        task: GAIA 任务字典
        max_turns: 最大对话轮数
        verbose: 是否打印详细信息
    Returns:
        dict: 包含 task_id, predicted_answer, ground_truth, is_correct
    """
    task_id = task.get("task_id", "unknown")
    question = task.get("Question", "")
    ground_truth = task.get("Final answer", "")
    file_name = task.get("file_name", "")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving Task: {task_id}")
        print(f"Question: {question[:150]}...")
        if file_name:
            print(f"Attachment: {file_name}")
        print(f"{'='*60}")

    prompt = build_task_prompt(task)

    # 检测是否有图片附件
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    has_image = False
    if file_name:
        import os
        ext = os.path.splitext(file_name)[1].lower()
        has_image = ext in IMAGE_EXTS

    # 创建多智能体工作流
    workflow = create_team(max_turns=max_turns, has_image=has_image)

    # 运行工作流对话（带重试）
    max_retries = 2
    predicted_answer = "NO_ANSWER"
    
    for attempt in range(max_retries + 1):
        try:
            # 设置超时（单个任务最多 7 分钟，给 Executor 多轮+Verifier 留够时间）
            # SOPWorkflow.run() 返回 SOPWorkflowResult
            result = await asyncio.wait_for(workflow.run(prompt), timeout=600)
            
            # 从 result 中提取消息（dict 列表）
            all_messages = []
            try:
                outputs = result.get_outputs()
                for output in outputs:
                    if isinstance(output, list):
                        all_messages.extend(output)
                    else:
                        all_messages.append(output)
            except Exception:
                pass

            # 从对话消息中提取 FINAL_ANSWER
            predicted_answer = extract_final_answer(all_messages)
            if predicted_answer is None:
                predicted_answer = "NO_ANSWER"

            # verbose 输出已在 run_sop_workflow 中完成
            break  # 成功，退出重试循环

        except (asyncio.TimeoutError, asyncio.CancelledError):
            print(f"Task timed out")
            predicted_answer = "NO_ANSWER"
            break
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries and ("Connection" in error_str or "timeout" in error_str.lower()):
                print(f"API error (attempt {attempt+1}/{max_retries+1}), retrying in 5s: {error_str[:100]}")
                await asyncio.sleep(5)
                workflow = create_team(max_turns=max_turns)  # 重新创建 workflow
                continue
            print(f"Error during workflow run: {e}")
            import traceback
            traceback.print_exc()
            predicted_answer = f"ERROR: {error_str}"
            break

    is_correct = compare_answers(predicted_answer, ground_truth)

    result_dict = {
        "task_id": task_id,
        "question": question,
        "file_name": file_name,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
    }

    if verbose:
        print(f"\n--- Result ---")
        print(f"Predicted: {predicted_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {'YES' if is_correct else 'NO'}")

    # 清理浏览器资源
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


def solve_single_task(task: dict, max_turns: int = 15, verbose: bool = True) -> dict:
    """同步版本的单任务求解（内部使用 asyncio.run）"""
    return asyncio.run(solve_single_task_async(task, max_turns, verbose))


async def solve_all_level1_async(
    max_tasks: Optional[int] = None,
    max_turns: int = 15,
    verbose: bool = True,
    output_path: str = "gaia_results.jsonl",
    resume: bool = False,
):
    """异步批量求解所有 Level 1 任务，支持断点续跑"""
    tasks = load_gaia_tasks()
    if max_tasks:
        tasks = tasks[:max_tasks]

    # 断点续跑：读取已完成的 task_id
    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done_ids.add(r.get("task_id"))
        print(f"[Resume] 已完成 {len(done_ids)} 题，跳过这些题目")

    remaining = [t for t in tasks if t.get("task_id") not in done_ids]
    total_all = len(tasks)
    print(f"Total Level 1 tasks: {total_all}, To solve: {len(remaining)}")

    if not resume:
        # 非续跑模式，清空结果文件
        if os.path.exists(output_path):
            os.remove(output_path)

    # 统计（包含已完成的）
    correct_count = 0
    if resume:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("is_correct"):
                        correct_count += 1
    solved_count = len(done_ids)

    results = []
    for i, task in enumerate(remaining):
        solved_count += 1
        print(f"\n[{solved_count}/{total_all}] ", end="")
        result = await solve_single_task_async(task, max_turns=max_turns, verbose=verbose)
        results.append(result)

        if result["is_correct"]:
            correct_count += 1

        # 逐条追加保存结果
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\nRunning accuracy: {correct_count}/{solved_count} = {correct_count/solved_count*100:.1f}%")

    # 最终统计
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total tasks solved: {solved_count}")
    print(f"Correct: {correct_count}")
    if solved_count > 0:
        print(f"Accuracy: {correct_count/solved_count*100:.1f}%")
    print(f"Results saved to: {output_path}")

    return results


def solve_all_level1(
    max_tasks: Optional[int] = None,
    max_turns: int = 15,
    verbose: bool = True,
    output_path: str = "gaia_results.jsonl",
    resume: bool = False,
):
    """同步版本的批量求解"""
    return asyncio.run(solve_all_level1_async(max_tasks, max_turns, verbose, output_path, resume))
