"""
MASTeam — 多智能体团队编排引擎（手动编排版）

确定性流转，不依赖 GroupChat 框架的消息广播:
  1. Planner 接收任务 → 生成 PLAN（指定每步由哪个 Worker 执行）
  2. 解析 PLAN → 按序调度 Worker.run()（各自调用工具 → 返回 RESULT）
  3. Worker 间可通过 HANDOFF 链式协作（如 WebSearcher → WebBrowser），无需回 Planner
  4. 收集所有 Worker 结果 → 构建 review prompt → Planner 审查
  5. Planner 输出 FINAL_ANSWER → 结束；或输出新 PLAN → 重复步骤 2-5
  6. max_rounds 兜底: 强制 Planner 给出最终回答

关键设计:
  - 每次 agent.run() 独立调用（无跨轮 session 状态），上下文通过 prompt 传递
  - 解析 PLAN 中的 "→ Assign: WorkerName" 来决定调度哪个 Worker
  - Worker 输出 HANDOFF: TargetWorker - task → 自动链式转发（最多 3 次/步）
  - 解析失败时 → LLM Selector 备选
  - Planner system_message 中 {agent_info} 自动替换为实际 Worker 能力信息
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import List, Optional, Dict, Any, Tuple

from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

from .metrics import MetricsCollector, MetricsLogger, TaskMetrics


# ============================================================
# MASTeam
# ============================================================

class MASTeam:
    """
    多智能体团队编排引擎（手动编排版）。

    Args:
        planner:          Planner Agent 配置 (dict: name, description, system_message)
        workers:          Worker Agent 配置列表 (list[dict]: name, description, system_message, tools)
        client:           OpenAIChatClient LLM 客户端
        max_rounds:       Planner 最大规划轮数（每轮: Planner → Workers → Planner review）
        verbose:          是否打印日志
        **kwargs:         兼容旧参数 (selector_prompt 等)
    """

    def __init__(
        self,
        planner: Dict[str, Any],
        workers: List[Dict[str, Any]],
        client: OpenAIChatClient,
        max_rounds: int = 5,
        verbose: bool = True,
        metrics_logger: Optional[MetricsLogger] = None,
        worker_timeout: int = 120,
        max_tool_calls_per_worker: int = 10,
        **kwargs,
    ):
        self.client = client
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.metrics_logger = metrics_logger
        self.worker_timeout = worker_timeout
        self.max_tool_calls_per_worker = max_tool_calls_per_worker

        # ---- 构建 Agent 信息摘要 ----
        self._agent_info = self._build_agent_info(workers)
        self._planner_name = planner["name"]
        self._worker_names: List[str] = [w["name"] for w in workers]

        # ---- 创建 agent_framework.Agent 实例 ----
        self._agents: Dict[str, Agent] = {}
        self._agent_list: List[Agent] = []
        self._build_agents(planner, workers)

        # ---- LLM Selector（PLAN 解析失败时的备选） ----
        selector_prompt = (
            "You select which worker agent should handle a task.\n\n"
            f"Available workers:\n{self._agent_info}\n\n"
            "Given a task description, reply with ONLY the worker name.\n"
            f"Valid names: {self._worker_names}"
        )
        self._selector_agent = Agent(
            client=client,
            name="_Selector",
            instructions=selector_prompt,
            default_options={"temperature": 0},
        )

    # --------------------------------------------------------
    # 构建
    # --------------------------------------------------------

    @staticmethod
    def _build_agent_info(workers: List[Dict[str, Any]]) -> str:
        """构建所有 Worker 的能力描述"""
        lines = []
        for w in workers:
            tool_names = [t.name for t in w.get("tools", []) if hasattr(t, "name")]
            tools_str = ", ".join(tool_names) if tool_names else "(no tools)"
            lines.append(f"- **{w['name']}**: {w['description']}\n    Tools: [{tools_str}]")
        return "\n".join(lines)

    def _build_agents(self, planner: Dict[str, Any], workers: List[Dict[str, Any]]):
        """将配置 dict 转换为 agent_framework.Agent 实例"""
        # Planner: 注入 {agent_info}
        planner_sys = planner["system_message"].replace("{agent_info}", self._agent_info)
        planner_agent = Agent(
            client=self.client,
            instructions=planner_sys,
            name=planner["name"],
            description=planner.get("description", "Planner"),
            default_options={"temperature": 0},
        )
        self._agents[planner["name"]] = planner_agent
        self._agent_list.append(planner_agent)

        # Workers
        for w in workers:
            tools = w.get("tools", [])
            agent = Agent(
                client=self.client,
                instructions=w["system_message"],
                name=w["name"],
                description=w.get("description", w["name"]),
                tools=tools if tools else None,
                default_options={"temperature": 0},
            )
            self._agents[w["name"]] = agent
            self._agent_list.append(agent)

    # --------------------------------------------------------
    # PLAN 解析
    # --------------------------------------------------------

    def _parse_plan(self, plan_text: str) -> List[Tuple[str, str]]:
        """
        从 Planner 输出中解析步骤。

        支持:
            1. Do X → Assign: WebSearcher
            1. Do X → WebSearcher
            1. Do X -> Assign: CodeExecutor

        Returns: [(step_description, worker_name), ...]
        """
        steps = []
        valid_names = set(self._worker_names)

        for line in plan_text.split("\n"):
            m = re.match(r"^\s*\d+[\.\)]\s*(.+)", line)
            if not m:
                continue
            step_text = m.group(1)

            # 尝试提取 Worker 名称
            assign_match = re.search(
                r"(?:→|->)\s*(?:[Aa]ssign:\s*)?(\w+)\s*$", step_text
            )
            if not assign_match:
                assign_match = re.search(r"[Aa]ssign:\s*(\w+)", step_text)

            if assign_match:
                worker_raw = assign_match.group(1)
                worker = self._match_worker_name(worker_raw, valid_names)
                if worker:
                    desc = step_text[:assign_match.start()].rstrip(" →->:,").strip()
                    steps.append((desc, worker))

        return steps

    def _match_worker_name(self, raw_name: str, valid_names: set) -> Optional[str]:
        """模糊匹配 worker 名称"""
        if raw_name in valid_names:
            return raw_name
        for name in valid_names:
            if name.lower() == raw_name.lower():
                return name
        for name in valid_names:
            if raw_name.lower() in name.lower() or name.lower() in raw_name.lower():
                return name
        return None

    # --------------------------------------------------------
    # HANDOFF 解析
    # --------------------------------------------------------

    def _parse_handoff(self, worker_text: str) -> Optional[Tuple[str, str]]:
        """
        从 Worker 输出中检测 HANDOFF 请求。

        格式: HANDOFF: <WorkerName> - <task description>

        Returns: (target_worker_name, task_description) 或 None
        """
        match = re.search(
            r"HANDOFF:\s*(\w+)\s*[-–—]\s*(.+?)(?:\n|$)", worker_text, re.IGNORECASE
        )
        if not match:
            return None

        raw_name = match.group(1).strip()
        task_desc = match.group(2).strip()
        valid_names = set(self._worker_names)
        target = self._match_worker_name(raw_name, valid_names)

        if target and task_desc:
            return (target, task_desc)
        return None

    # --------------------------------------------------------
    # LLM Selector (备选)
    # --------------------------------------------------------

    async def _select_worker_llm(self, step_desc: str) -> Optional[str]:
        """PLAN 解析失败时，使用 LLM 选择 Worker"""
        prompt = f"Task: {step_desc}\nWhich worker should handle this?"
        try:
            result = await asyncio.wait_for(
                self._selector_agent.run(prompt), timeout=30
            )
            text = result.text if hasattr(result, "text") else str(result)
            for name in self._worker_names:
                if name.lower() in text.lower():
                    return name
        except Exception:
            pass
        return None

    # --------------------------------------------------------
    # Agent 调用
    # --------------------------------------------------------

    async def _invoke_agent(self, agent_name: str, prompt: str, timeout: int = 120) -> Tuple[str, Any]:
        """调用一个 Agent，返回 (文本输出, AgentResponse)."""
        agent = self._agents.get(agent_name)
        if not agent:
            return f"[Error: unknown agent '{agent_name}']", None
        try:
            result = await asyncio.wait_for(agent.run(prompt), timeout=timeout)
            text = result.text if hasattr(result, "text") else str(result)

            # ---- 工具调用上限检查 ----
            # 统计本次 agent.run() 中实际的工具调用次数
            tool_count = 0
            messages = getattr(result, "messages", [])
            for msg in messages:
                contents = getattr(msg, "contents", [])
                for content in contents:
                    if getattr(content, "type", None) == "function_call":
                        tool_count += 1

            if tool_count > self.max_tool_calls_per_worker and self.verbose:
                print(f"    [WARN] {agent_name} made {tool_count} tool calls (limit: {self.max_tool_calls_per_worker})")

            return text, result
        except asyncio.TimeoutError:
            return "[Timeout]", None
        except Exception as e:
            return f"[Error: {e}]", None

    # --------------------------------------------------------
    # 并行分组
    # --------------------------------------------------------

    @staticmethod
    def _group_steps_for_parallel(
        steps: List[Tuple[str, str]],
    ) -> List[List[Tuple[str, str]]]:
        """
        将独立步骤分组用于并行执行。

        规则: 同一个 Worker 的步骤必须串行（可能有数据依赖），
        不同 Worker 的步骤放入同一组并行执行。

        Returns: [[group1_steps], [group2_steps], ...]
        """
        if len(steps) <= 1:
            return [steps] if steps else []

        # 贪心: 扫描步骤，遇到相同 Worker 就断开新组
        groups: List[List[Tuple[str, str]]] = []
        current_group: List[Tuple[str, str]] = []
        workers_in_group: set = set()

        for desc, worker in steps:
            if worker in workers_in_group:
                # 同一 Worker 出现两次 → 当前组结束，开新组
                groups.append(current_group)
                current_group = [(desc, worker)]
                workers_in_group = {worker}
            else:
                current_group.append((desc, worker))
                workers_in_group.add(worker)

        if current_group:
            groups.append(current_group)

        return groups

    # --------------------------------------------------------
    # HANDOFF 处理
    # --------------------------------------------------------

    async def _process_handoffs(
        self,
        task: str,
        worker_name: str,
        worker_text: str,
        worker_results: List[Tuple[str, str]],
        history: List[Dict[str, str]],
        collector: 'MetricsCollector',
        max_handoffs: int,
        timeout: int,
    ):
        """处理 Worker 输出中的 HANDOFF 链式协作"""
        handoff_count = 0
        while handoff_count < max_handoffs:
            handoff = self._parse_handoff(worker_text)
            if not handoff:
                break

            target_name, handoff_task = handoff
            if target_name == worker_name:
                break

            handoff_count += 1
            collector.record_handoff(worker_name, target_name, handoff_task)
            if self.verbose:
                print(f"    ↳ HANDOFF → {target_name}: {handoff_task[:120]}")

            handoff_prompt = self._build_worker_prompt(task, handoff_task, worker_results)
            t0 = time.time()
            worker_text, worker_resp = await self._invoke_agent(
                target_name, handoff_prompt, timeout=timeout
            )
            collector.record_agent_call(
                target_name, worker_resp, time.time() - t0, is_planner=False
            )

            history.append({"source": target_name, "content": worker_text})
            worker_results.append((target_name, worker_text))

            if self.verbose:
                preview = worker_text[:200].replace("\n", " ")
                print(f"    [{target_name}]: {preview}")

            worker_name = target_name

    # --------------------------------------------------------
    # Prompt 构建
    # --------------------------------------------------------

    def _build_planner_plan_prompt(self, task: str) -> str:
        """Planner 首轮 prompt"""
        return (
            f"Task: {task}\n\n"
            "Create a PLAN to accomplish this task. "
            "Use as FEW steps as possible (1-2 preferred). "
            "Steps with DIFFERENT workers can run in parallel."
        )

    def _build_planner_review_prompt(
        self,
        task: str,
        plan_text: str,
        worker_results: List[Tuple[str, str]],
    ) -> str:
        """Planner 审查 prompt（收到 Worker 结果后）"""
        parts = [f"Task: {task}\n"]
        parts.append("Worker execution results:")
        for worker_name, result_text in worker_results:
            truncated = (result_text[:2000] + "...(truncated)") if len(result_text) > 2000 else result_text
            parts.append(f"\n[{worker_name}]: {truncated}")
        parts.append(
            "\n\nIMPORTANT: You MUST output FINAL_ANSWER if ANY worker found relevant data.\n"
            "Even partial or approximate data is enough — extract the best answer NOW.\n"
            "Output FINAL_ANSWER: <answer>\n"
            "ONLY create a NEW PLAN (max 2 steps) if workers found ZERO relevant information."
        )
        return "\n".join(parts)

    def _build_worker_prompt(
        self,
        task: str,
        step_desc: str,
        prior_results: List[Tuple[str, str]],
    ) -> str:
        """Worker 执行 prompt"""
        parts = [f"Original task: {task}\n"]
        parts.append(f"Your assignment: {step_desc}\n")
        if prior_results:
            # 只包含最后 2 个结果，截断更激进
            parts.append("Previous results:")
            for name, text in prior_results[-2:]:
                truncated = text[:500] + "..." if len(text) > 500 else text
                parts.append(f"  [{name}]: {truncated}")
        parts.append(
            "\nExecute your assignment using your tools. "
            "Be efficient — use minimum tool calls needed. "
            "When done, summarize the KEY DATA you found and say RESULT: <your findings>.\n"
            "If the task asks 'how many', COUNT the items and include the number in RESULT."
        )
        return "\n".join(parts)

    # --------------------------------------------------------
    # 主流程
    # --------------------------------------------------------

    async def run(
        self,
        task: str,
        task_id: str = "",
        expected_answer: str = "",
    ) -> Dict[str, Any]:
        """
        运行团队完成任务。

        流程: Planner(plan) → Workers(execute) → Planner(review) → loop

        Args:
            task:             任务描述
            task_id:          任务 ID（用于 metrics 记录）
            expected_answer:  期望答案（用于正确率统计）

        Returns:
            {
                "answer":   str,          # FINAL_ANSWER 提取结果
                "messages": List[dict],   # [{"source": ..., "content": ...}, ...]
                "turns":    int,          # 总对话轮数
                "rounds":   int,          # Planner 规划轮数
                "elapsed":  float,        # 总耗时(秒)
                "metrics":  TaskMetrics,  # 详细指标
            }
        """
        start = time.time()
        history: List[Dict[str, str]] = []
        final_answer = ""
        planner_rounds = 0

        # ---- 指标收集器 ----
        collector = MetricsCollector(task_id=task_id, task=task)

        if self.verbose:
            print(f"\n[MASTeam] Starting task...")
            print(f"  Agents: {[a.name for a in self._agent_list]}")

        last_plan_text = ""
        worker_results: List[Tuple[str, str]] = []

        for round_num in range(self.max_rounds):
            # ========== 1. Planner 规划/审查 ==========
            if round_num == 0:
                planner_prompt = self._build_planner_plan_prompt(task)
            else:
                planner_prompt = self._build_planner_review_prompt(
                    task, last_plan_text, worker_results
                )

            t0 = time.time()
            planner_text, planner_resp = await self._invoke_agent(
                self._planner_name, planner_prompt, timeout=60
            )
            collector.record_agent_call(
                self._planner_name, planner_resp, time.time() - t0, is_planner=True
            )
            history.append({"source": self._planner_name, "content": planner_text})
            planner_rounds += 1

            if self.verbose:
                preview = planner_text[:300].replace("\n", " ")
                print(f"\n  [Round {round_num + 1}] [{self._planner_name}]: {preview}")

            # 检查 FINAL_ANSWER（首轮跳过: 强制先走 Worker）
            fa = self._extract_final_answer(planner_text)
            if fa and round_num > 0:
                final_answer = fa
                break

            last_plan_text = planner_text

            # ========== 2. 解析 PLAN → Worker 步骤列表 ==========
            steps = self._parse_plan(planner_text)

            if not steps:
                if self.verbose:
                    print("  [WARN] Could not parse PLAN steps, trying LLM selector...")
                worker_name = await self._select_worker_llm(task)
                if worker_name:
                    steps = [(task[:200], worker_name)]

            if not steps:
                if self.verbose:
                    print("  [WARN] No steps found, will force Planner answer.")
                break

            # ========== 3. 按组调度 Workers（独立步骤并行，支持 HANDOFF）==========
            worker_results = []
            max_handoffs = 2  # 每步最大链式次数，防止无限循环
            wk_timeout = self.worker_timeout

            # ---- 分组: 同一 Worker 的步骤串行，不同 Worker 的步骤并行 ----
            parallel_groups = self._group_steps_for_parallel(steps)

            for group in parallel_groups:
                if len(group) == 1:
                    # 单步顺序执行
                    step_desc, worker_name = group[0]
                    if self.verbose:
                        print(f"  [Step] → {worker_name}: {step_desc[:120]}")

                    worker_prompt = self._build_worker_prompt(task, step_desc, worker_results)
                    t0 = time.time()
                    worker_text, worker_resp = await self._invoke_agent(worker_name, worker_prompt, timeout=wk_timeout)
                    collector.record_agent_call(
                        worker_name, worker_resp, time.time() - t0, is_planner=False
                    )
                    history.append({"source": worker_name, "content": worker_text})
                    worker_results.append((worker_name, worker_text))

                    if self.verbose:
                        preview = worker_text[:200].replace("\n", " ")
                        print(f"    [{worker_name}]: {preview}")

                    # HANDOFF 链式协作
                    await self._process_handoffs(
                        task, worker_name, worker_text, worker_results,
                        history, collector, max_handoffs, wk_timeout
                    )
                else:
                    # 多步并行执行
                    if self.verbose:
                        names = [wn for _, wn in group]
                        print(f"  [Parallel] → {names}")

                    async def _run_step(desc, wname):
                        prompt = self._build_worker_prompt(task, desc, worker_results)
                        t0 = time.time()
                        text, resp = await self._invoke_agent(wname, prompt, timeout=wk_timeout)
                        elapsed_s = time.time() - t0
                        return wname, text, resp, elapsed_s

                    tasks_list = [_run_step(d, w) for d, w in group]
                    results_list = await asyncio.gather(*tasks_list, return_exceptions=True)

                    for item in results_list:
                        if isinstance(item, Exception):
                            if self.verbose:
                                print(f"    [ERROR] Parallel step failed: {item}")
                            continue
                        wname, wtext, wresp, welapsed = item
                        collector.record_agent_call(wname, wresp, welapsed, is_planner=False)
                        history.append({"source": wname, "content": wtext})
                        worker_results.append((wname, wtext))
                        if self.verbose:
                            preview = wtext[:200].replace("\n", " ")
                            print(f"    [{wname}]: {preview}")

                        # HANDOFF
                        await self._process_handoffs(
                            task, wname, wtext, worker_results,
                            history, collector, max_handoffs, wk_timeout
                        )

            # ========== 4. Planner 审查结果 ==========
            review_prompt = self._build_planner_review_prompt(task, last_plan_text, worker_results)
            t0 = time.time()
            planner_review, review_resp = await self._invoke_agent(self._planner_name, review_prompt, timeout=60)
            collector.record_agent_call(
                self._planner_name, review_resp, time.time() - t0, is_planner=True
            )
            history.append({"source": self._planner_name, "content": planner_review})
            planner_rounds += 1

            if self.verbose:
                preview = planner_review[:300].replace("\n", " ")
                print(f"  [{self._planner_name}] review: {preview}")

            # 检查 FINAL_ANSWER
            fa = self._extract_final_answer(planner_review)
            if fa:
                final_answer = fa
                break

            # 没有 FINAL_ANSWER → 把 review 当作新 plan，进入下一轮
            last_plan_text = planner_review

        # ========== 5. 兜底: 强制要求 FINAL_ANSWER ==========
        if not final_answer:
            recent_context = "\n".join(
                f"[{m['source']}]: {m['content'][:400]}" for m in history[-6:]
            )
            force_prompt = (
                f"Task: {task}\n\n"
                f"Gathered information:\n{recent_context}\n\n"
                "You MUST output FINAL_ANSWER: <answer> NOW based on everything above."
            )
            t0 = time.time()
            forced, forced_resp = await self._invoke_agent(self._planner_name, force_prompt, timeout=60)
            collector.record_agent_call(
                self._planner_name, forced_resp, time.time() - t0, is_planner=True
            )
            history.append({"source": self._planner_name, "content": forced})
            planner_rounds += 1
            final_answer = self._extract_final_answer(forced)

            if self.verbose:
                preview = forced[:300].replace("\n", " ")
                print(f"  [{self._planner_name}] forced: {preview}")

        elapsed = time.time() - start

        # ---- 生成 TaskMetrics ----
        metrics = collector.finalize(
            answer=final_answer or "",
            expected_answer=expected_answer,
            rounds=planner_rounds,
        )

        # ---- 自动持久化 ----
        if self.metrics_logger:
            self.metrics_logger.log(metrics)

        if self.verbose:
            turns = len(history)
            print(f"\n  Done: {planner_rounds} planning rounds, {turns} turns, {elapsed:.1f}s")
            print(f"  Answer: {final_answer[:200] if final_answer else '(no answer)'}")
            print(f"  Tokens: in={metrics.input_tokens}, out={metrics.output_tokens}, total={metrics.total_tokens}")
            print(f"  Tool calls: {metrics.total_tool_calls} ({metrics.tool_calls})")
            print(f"  Handoffs: {metrics.handoff_count}")

        return {
            "answer": final_answer or "",
            "messages": history,
            "turns": len(history),
            "rounds": planner_rounds,
            "elapsed": round(elapsed, 1),
            "metrics": metrics,
        }

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """从文本中提取 FINAL_ANSWER，并清理多余描述文字"""
        match = re.search(r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        raw = ""
        if match:
            raw = match.group(1).strip()
        else:
            idx = text.upper().find("FINAL_ANSWER")
            if idx >= 0:
                after = text[idx + len("FINAL_ANSWER"):].strip().lstrip(":").strip()
                if after:
                    raw = after.split("\n")[0].strip()
        if not raw:
            return ""

        # 清理常见的包裹格式
        # 去除末尾句号
        raw = raw.rstrip(".")
        # 去除 "The answer is X" / "X is Y" 等前缀描述
        # 模式: "The ... is/are <answer>" → 提取 <answer>
        cleanup = re.match(
            r'^(?:the\s+)?(?:answer|result|value|number|total|count|name|code|grant\s*(?:number)?|award\s*(?:number)?|NASA\s+award\s*(?:number)?)'
            r'\s+(?:is|are|was|=)\s+(.+)$',
            raw, re.IGNORECASE
        )
        if cleanup:
            raw = cleanup.group(1).strip().rstrip(".")
        # 模式: "<description> is <answer>" — 如果包含 " is " 且最后部分像是答案
        if ' is ' in raw.lower() and not raw[0].isdigit():
            parts = re.split(r'\s+is\s+', raw, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2 and len(parts[1]) < len(parts[0]):
                candidate = parts[1].strip().rstrip(".")
                if candidate:
                    raw = candidate
        # 模式: "... for R. G. Arendt is 80GSFC21M0002" → 取最后的实际值
        for_is = re.search(r'(?:for|of)\s+.+?\s+is\s+(.+?)$', raw, re.IGNORECASE)
        if for_is:
            raw = for_is.group(1).strip().rstrip(".")

        return raw
