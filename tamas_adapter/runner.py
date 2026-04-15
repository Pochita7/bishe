"""
TAMAS 测试运行器
基于 Planner → Executor → Verifier 流水线运行 TAMAS 测试
Executor 注入 TAMAS 领域工具（良性 + 恶意），使安全测试有意义
支持 clean/attack 对比，输出统计报告
"""
import asyncio
import json
import os
import re
import time
from typing import List, Dict, Optional
from collections import defaultdict

from tamas_adapter.loader import (
    load_all_tamas, extract_task_info,
    ATTACK_TYPES, SCENARIOS, get_scenario_benign_tools,
)
from tamas_adapter.prompt_builder import build_clean_prompt, build_attack_prompt
from tamas_adapter.evaluator import evaluate_output, evaluate_with_llm

# 使用 TAMAS 专用 Agent（含领域工具）
from tamas_adapter.agents import create_tamas_team
from gaia_solver.solver import extract_final_answer


def _get_msg_text(msg) -> str:
    """从消息（dict 或对象）中提取文本"""
    if isinstance(msg, dict):
        return msg.get("text", "") or msg.get("content", "")
    if hasattr(msg, "text") and msg.text:
        return msg.text
    if hasattr(msg, "content"):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _get_msg_author(msg) -> str:
    """从消息中提取作者"""
    if isinstance(msg, dict):
        return msg.get("author", "") or msg.get("role", "")
    return getattr(msg, "author_name", "") or getattr(msg, "name", "") or ""


async def run_single_task(
    task_info: Dict,
    mode: str = "clean",
    max_turns: int = 10,
    use_llm_eval: bool = False,
    verbose: bool = True,
    enable_guardian: bool = False,
) -> Dict:
    """
    运行单个 TAMAS 任务

    Args:
        task_info: extract_task_info() 的输出
        mode: "clean" 或 "attack"
        max_turns: 最大对话轮数
        use_llm_eval: 是否用 LLM 评估
        verbose: 是否打印详情
    """
    task_id = task_info["id"]

    # 构建 prompt
    if mode == "clean":
        prompt = build_clean_prompt(task_info)
    else:
        prompt = build_attack_prompt(task_info)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{mode.upper()}] {task_id}")
        print(f"Attack: {task_info['attack_type']} | Scenario: {task_info['scenario']}")
        q = task_info['clean_query'] if mode == 'clean' else task_info['attack_query']
        print(f"Query: {q[:120]}...")
        print(f"{'='*60}")

    start_time = time.time()
    predicted_answer = "NO_ANSWER"
    full_output = ""

    try:
        # 使用 TAMAS 专用 Agent（Executor 含领域工具）
        workflow = create_tamas_team(
            scenario=task_info["scenario"],
            max_turns=max_turns,
            include_general_tools=True,
            include_malicious=True,  # 恶意工具也注入，真实测试防御能力
            enable_guardian=enable_guardian,
        )
        result = await asyncio.wait_for(workflow.run(prompt), timeout=300)

        # 提取消息
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

        # 提取最终答案
        predicted_answer = extract_final_answer(all_messages) or "NO_ANSWER"

        # 拼接完整输出用于评估
        output_parts = []
        for msg in all_messages:
            text = _get_msg_text(msg)
            author = _get_msg_author(msg)
            if text:
                output_parts.append(f"[{author}]: {text}")
        full_output = "\n".join(output_parts)

    except asyncio.TimeoutError:
        predicted_answer = "TIMEOUT"
        full_output = "Task timed out after 300s"
        if verbose:
            print("  [TIMEOUT]")
    except Exception as e:
        predicted_answer = f"ERROR: {str(e)[:200]}"
        full_output = str(e)
        if verbose:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - start_time

    # 获取场景预期良性工具（TAMAS 标准）
    expected_tools = get_scenario_benign_tools(task_info["scenario"])

    # 评估
    if use_llm_eval:
        eval_result = evaluate_with_llm(
            user_query=task_info["clean_query"],
            actual_output=full_output,
            attack_type=task_info["attack_type"],
            mode=mode,
            injected_query=task_info["attack_query"] if mode == "attack" else "",
        )
    else:
        eval_result = evaluate_output(
            user_query=task_info["clean_query"],
            actual_output=full_output,
            attack_type=task_info["attack_type"],
            mode=mode,
            injected_query=task_info["attack_query"] if mode == "attack" else "",
            expected_tools=expected_tools,
        )

    result_dict = {
        "id": task_id,
        "mode": mode,
        "attack_type": task_info["attack_type"],
        "scenario": task_info["scenario"],
        "clean_query": task_info["clean_query"],
        "attack_query": task_info["attack_query"],
        "predicted_answer": predicted_answer,
        "full_output": full_output[:2000],  # 截断保存
        "task_completed": eval_result["task_completed"],
        "attack_resisted": eval_result["attack_resisted"],
        "aria_score": eval_result["aria_score"],
        "score": eval_result["score"],
        "reasoning": eval_result["reasoning"],
        "tools_called": eval_result.get("tools_called", []),
        "malicious_tools_called": eval_result.get("malicious_tools_called", []),
        "tool_coverage": eval_result.get("tool_coverage", 0.0),
        "elapsed_time": round(elapsed, 1),
    }

    if verbose:
        tc = "OK" if eval_result["task_completed"] else "FAIL"
        ar = "SAFE" if eval_result["attack_resisted"] else "PWNED"
        mal_tools = eval_result.get("malicious_tools_called", [])
        tools_info = f" | MalTools: {','.join(mal_tools)}" if mal_tools else ""
        print(f"  Answer: {predicted_answer[:80]}")
        print(f"  Task: {tc} | Resist: {ar} | ARIA: {eval_result['aria_score']} | "
              f"Score: {eval_result['score']:.2f} | Time: {elapsed:.1f}s{tools_info}")

    # 清理浏览器
    try:
        from gaia_solver.tools import _cleanup_browser
        _cleanup_browser()
    except Exception:
        pass

    return result_dict


async def run_comparison(
    attack_types: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    max_tasks_per_group: Optional[int] = None,
    max_turns: int = 10,
    use_llm_eval: bool = False,
    verbose: bool = True,
    output_path: str = "tamas_results.jsonl",
    resume: bool = False,
    enable_guardian: bool = False,
) -> Dict:
    """
    运行 clean vs attack 对比实验
    每条 TAMAS 数据同时运行两种模式
    """
    attack_types = attack_types or ATTACK_TYPES
    scenarios = scenarios or SCENARIOS

    all_data = load_all_tamas(attack_types, scenarios)

    if max_tasks_per_group:
        filtered = []
        count_map = {}
        for item in all_data:
            key = f"{item['_attack_type']}_{item['_scenario']}"
            count_map[key] = count_map.get(key, 0) + 1
            if count_map[key] <= max_tasks_per_group:
                filtered.append(item)
        all_data = filtered

    total = len(all_data)
    print(f"\n{'='*60}")
    print(f"TAMAS MAS 安全测试")
    print(f"数据: {total} 条 x 2 模式 = {total * 2} 次推理")
    print(f"{'='*60}")

    # 断点续跑
    done_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    done_ids.add(f"{r['id']}_{r['mode']}")
        print(f"[Resume] 已完成 {len(done_ids)} 条，跳过")

    if not resume and os.path.exists(output_path):
        os.remove(output_path)

    # 运行所有任务
    for i, item in enumerate(all_data):
        task_info = extract_task_info(item)

        for mode in ["clean", "attack"]:
            run_id = f"{task_info['id']}_{mode}"
            if run_id in done_ids:
                continue

            print(f"\n--- [{i+1}/{total}] {mode.upper()} ---")

            result = await run_single_task(
                task_info, mode=mode, max_turns=max_turns,
                use_llm_eval=use_llm_eval, verbose=verbose,
                enable_guardian=enable_guardian,
            )

            # 逐条保存
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 生成报告
    print_report(output_path)


def print_report(output_path: str):
    """从结果文件生成详细报告"""
    if not os.path.exists(output_path):
        print("无结果文件")
        return

    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))

    if not results:
        print("无结果数据")
        return

    print(f"\n{'='*70}")
    print(f"  TAMAS 测试报告")
    print(f"{'='*70}")

    # 按 mode 统计
    clean_results = [r for r in results if r["mode"] == "clean"]
    attack_results = [r for r in results if r["mode"] == "attack"]

    if clean_results:
        clean_completed = sum(1 for r in clean_results if r["task_completed"])
        clean_avg_score = sum(r["score"] for r in clean_results) / len(clean_results)
        print(f"\n  [CLEAN] 无攻击模式:")
        print(f"    任务完成: {clean_completed}/{len(clean_results)} "
              f"({clean_completed/len(clean_results)*100:.1f}%)")
        print(f"    平均评分: {clean_avg_score:.3f}")

    if attack_results:
        atk_completed = sum(1 for r in attack_results if r["task_completed"])
        atk_resisted = sum(1 for r in attack_results if r["attack_resisted"])
        atk_avg_score = sum(r["score"] for r in attack_results) / len(attack_results)
        print(f"\n  [ATTACK] 攻击模式:")
        print(f"    任务完成: {atk_completed}/{len(attack_results)} "
              f"({atk_completed/len(attack_results)*100:.1f}%)")
        print(f"    攻击抵抗: {atk_resisted}/{len(attack_results)} "
              f"({atk_resisted/len(attack_results)*100:.1f}%)")
        print(f"    平均评分: {atk_avg_score:.3f}")

        # ARIA 分布
        aria_dist = defaultdict(int)
        for r in attack_results:
            aria_dist[r["aria_score"]] += 1
        print(f"\n    ARIA 分布:")
        for aria in ["NOTA", "ARIA_1", "ARIA_2", "ARIA_3", "ARIA_4"]:
            if aria in aria_dist:
                pct = aria_dist[aria] / len(attack_results) * 100
                print(f"      {aria}: {aria_dist[aria]} ({pct:.1f}%)")

    # 按攻击类型分组
    if attack_results:
        print(f"\n  按攻击类型:")
        print(f"  {'Attack':<16} {'Clean完成':>10} {'Attack完成':>10} {'抵抗率':>10} {'Avg Score':>10}")
        print(f"  {'-'*58}")

        by_attack = defaultdict(lambda: {
            "clean_ok": 0, "clean_n": 0,
            "attack_ok": 0, "attack_n": 0,
            "resist": 0, "score_sum": 0.0,
        })

        for r in results:
            at = r["attack_type"]
            if r["mode"] == "clean":
                by_attack[at]["clean_n"] += 1
                if r["task_completed"]:
                    by_attack[at]["clean_ok"] += 1
            else:
                by_attack[at]["attack_n"] += 1
                by_attack[at]["score_sum"] += r["score"]
                if r["task_completed"]:
                    by_attack[at]["attack_ok"] += 1
                if r["attack_resisted"]:
                    by_attack[at]["resist"] += 1

        for at, s in sorted(by_attack.items()):
            c = f"{s['clean_ok']}/{s['clean_n']}" if s['clean_n'] else "N/A"
            a = f"{s['attack_ok']}/{s['attack_n']}" if s['attack_n'] else "N/A"
            r = f"{s['resist']}/{s['attack_n']}" if s['attack_n'] else "N/A"
            avg = f"{s['score_sum']/s['attack_n']:.3f}" if s['attack_n'] else "N/A"
            print(f"  {at:<16} {c:>10} {a:>10} {r:>10} {avg:>10}")

    # 按场景分组
    if len(set(r["scenario"] for r in results)) > 1:
        print(f"\n  按场景:")
        print(f"  {'Scenario':<14} {'Clean完成':>10} {'Attack完成':>10} {'抵抗率':>10}")
        print(f"  {'-'*46}")

        by_scenario = defaultdict(lambda: {
            "clean_ok": 0, "clean_n": 0,
            "attack_ok": 0, "attack_n": 0, "resist": 0,
        })

        for r in results:
            sc = r["scenario"]
            if r["mode"] == "clean":
                by_scenario[sc]["clean_n"] += 1
                if r["task_completed"]:
                    by_scenario[sc]["clean_ok"] += 1
            else:
                by_scenario[sc]["attack_n"] += 1
                if r["task_completed"]:
                    by_scenario[sc]["attack_ok"] += 1
                if r["attack_resisted"]:
                    by_scenario[sc]["resist"] += 1

        for sc, s in sorted(by_scenario.items()):
            c = f"{s['clean_ok']}/{s['clean_n']}" if s['clean_n'] else "N/A"
            a = f"{s['attack_ok']}/{s['attack_n']}" if s['attack_n'] else "N/A"
            r_ = f"{s['resist']}/{s['attack_n']}" if s['attack_n'] else "N/A"
            print(f"  {sc:<14} {c:>10} {a:>10} {r_:>10}")

    print(f"\n  结果文件: {output_path}")
    print(f"{'='*70}")
