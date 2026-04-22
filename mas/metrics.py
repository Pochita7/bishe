"""
MAS 指标追踪与持久化

追踪每次任务执行的:
  - token 消耗 (input_tokens, output_tokens, total_tokens)
  - 耗时 (总时间, Planner 时间, Worker 时间)
  - 工具调用 (哪些工具被调用, 调用次数)
  - Agent 调用 (每个 Agent 调用几次)
  - HANDOFF 次数
  - 轮次和消息数
  - 正确率 (与 expected_answer 对比)

数据以 JSONL 格式追加写入本地文件。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


# ============================================================
# TaskMetrics — 单次任务的指标数据结构
# ============================================================

@dataclass
class TaskMetrics:
    """单次任务执行的完整指标"""

    # 任务标识
    task_id: str = ""
    task: str = ""
    timestamp: str = ""

    # 结果
    answer: str = ""
    expected_answer: str = ""
    is_correct: Optional[bool] = None

    # Token 消耗
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    external_input_tokens: int = 0
    external_output_tokens: int = 0
    external_total_tokens: int = 0

    # 耗时 (秒)
    elapsed: float = 0.0
    planner_time: float = 0.0
    worker_time: float = 0.0

    # Agent 调用统计
    agent_calls: Dict[str, int] = field(default_factory=dict)
    total_agent_calls: int = 0

    # 工具调用统计
    tool_calls: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0
    external_llm_calls: Dict[str, int] = field(default_factory=dict)
    external_llm_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_external_llm_calls: int = 0
    external_calls_without_usage: int = 0

    # HANDOFF 统计
    handoff_count: int = 0
    handoff_details: List[Dict[str, str]] = field(default_factory=list)

    # 轮次
    rounds: int = 0
    turns: int = 0

    # 消息日志 (可选, 用于调试)
    messages: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的 dict（不含 messages 以节省空间）"""
        d = asdict(self)
        # 默认不保存完整消息日志
        d.pop("messages", None)
        return d

    def to_dict_full(self) -> Dict[str, Any]:
        """转为完整 dict（含 messages）"""
        return asdict(self)


# ============================================================
# MetricsCollector — 运行过程中收集指标
# ============================================================

class MetricsCollector:
    """
    在 MASTeam.run() 期间收集指标。

    用法:
        collector = MetricsCollector(task_id="xxx", task="...")
        # 每次 agent.run() 后:
        collector.record_agent_call("Planner", response, elapsed)
        # HANDOFF 时:
        collector.record_handoff("WebSearcher", "CodeExecutor", "compute sum")
        # 完成后:
        metrics = collector.finalize(answer="42", expected="42")
    """

    def __init__(self, task_id: str = "", task: str = ""):
        self.task_id = task_id
        self.task = task
        self.start_time = time.time()
        self.timestamp = datetime.now().isoformat()

        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._external_input_tokens = 0
        self._external_output_tokens = 0
        self._external_total_tokens = 0
        self._planner_time = 0.0
        self._worker_time = 0.0
        self._agent_calls: Dict[str, int] = {}
        self._tool_calls: Dict[str, int] = {}
        self._external_llm_calls: Dict[str, int] = {}
        self._external_llm_usage: Dict[str, Dict[str, Any]] = {}
        self._external_calls_without_usage = 0
        self._handoff_count = 0
        self._handoff_details: List[Dict[str, str]] = []
        self._messages: List[Dict[str, str]] = []

    def record_agent_call(
        self,
        agent_name: str,
        response: Any,
        elapsed: float,
        is_planner: bool = False,
    ):
        """记录一次 Agent 调用的指标。

        Args:
            agent_name:  被调用的 Agent 名称
            response:    agent_framework.AgentResponse 对象
            elapsed:     本次调用耗时 (秒)
            is_planner:  是否为 Planner 调用
        """
        # Agent 调用计数
        self._agent_calls[agent_name] = self._agent_calls.get(agent_name, 0) + 1

        # 耗时
        if is_planner:
            self._planner_time += elapsed
        else:
            self._worker_time += elapsed

        # Token
        usage = getattr(response, "usage_details", None)
        if usage:
            inp = usage.get("input_token_count", 0) or 0
            out = usage.get("output_token_count", 0) or 0
            tot = usage.get("total_token_count", 0) or 0
            self._input_tokens += inp
            self._output_tokens += out
            self._total_tokens += tot if tot else (inp + out)

        # 工具调用 (从 response.messages 中提取 function_call Content)
        messages = getattr(response, "messages", [])
        for msg in messages:
            contents = getattr(msg, "contents", [])
            for content in contents:
                ctype = getattr(content, "type", None)
                if ctype == "function_call":
                    tool_name = getattr(content, "name", None) or "unknown"
                    self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1

        # 消息文本
        text = response.text if hasattr(response, "text") else str(response)
        self._messages.append({"source": agent_name, "content": text})

    def record_handoff(self, from_agent: str, to_agent: str, task_desc: str):
        """记录一次 HANDOFF"""
        self._handoff_count += 1
        self._handoff_details.append({
            "from": from_agent,
            "to": to_agent,
            "task": task_desc[:200],
        })

    def sync_external_usage(self, snapshot: Optional[Dict[str, Any]]):
        """Sync direct-model token usage captured outside agent.run()."""
        if not snapshot:
            self._external_input_tokens = 0
            self._external_output_tokens = 0
            self._external_total_tokens = 0
            self._external_llm_calls = {}
            self._external_llm_usage = {}
            self._external_calls_without_usage = 0
            return

        self._external_input_tokens = int(snapshot.get("input_tokens", 0) or 0)
        self._external_output_tokens = int(snapshot.get("output_tokens", 0) or 0)
        self._external_total_tokens = int(snapshot.get("total_tokens", 0) or 0)
        self._external_calls_without_usage = int(snapshot.get("calls_without_usage", 0) or 0)

        sources = snapshot.get("sources", {}) or {}
        self._external_llm_calls = {
            str(name): int((detail or {}).get("calls", 0) or 0)
            for name, detail in sources.items()
        }
        self._external_llm_usage = {
            str(name): {
                "calls": int((detail or {}).get("calls", 0) or 0),
                "input_tokens": int((detail or {}).get("input_tokens", 0) or 0),
                "output_tokens": int((detail or {}).get("output_tokens", 0) or 0),
                "total_tokens": int((detail or {}).get("total_tokens", 0) or 0),
                "models": dict((detail or {}).get("models", {}) or {}),
                "operations": dict((detail or {}).get("operations", {}) or {}),
            }
            for name, detail in sources.items()
        }

    def finalize(
        self,
        answer: str = "",
        expected_answer: str = "",
        rounds: int = 0,
    ) -> TaskMetrics:
        """
        完成收集，生成 TaskMetrics。

        Args:
            answer:          系统给出的答案
            expected_answer: 期望答案（用于计算正确率）
            rounds:          Planner 规划轮数
        """
        elapsed = time.time() - self.start_time

        # 正确性判断
        is_correct = None
        if expected_answer:
            is_correct = _check_answer(answer, expected_answer)

        combined_input_tokens = self._input_tokens + self._external_input_tokens
        combined_output_tokens = self._output_tokens + self._external_output_tokens
        combined_total_tokens = self._total_tokens + self._external_total_tokens

        return TaskMetrics(
            task_id=self.task_id,
            task=self.task[:500],
            timestamp=self.timestamp,
            answer=answer,
            expected_answer=expected_answer,
            is_correct=is_correct,
            input_tokens=combined_input_tokens,
            output_tokens=combined_output_tokens,
            total_tokens=combined_total_tokens,
            external_input_tokens=self._external_input_tokens,
            external_output_tokens=self._external_output_tokens,
            external_total_tokens=self._external_total_tokens,
            elapsed=round(elapsed, 1),
            planner_time=round(self._planner_time, 1),
            worker_time=round(self._worker_time, 1),
            agent_calls=dict(self._agent_calls),
            total_agent_calls=sum(self._agent_calls.values()),
            tool_calls=dict(self._tool_calls),
            total_tool_calls=sum(self._tool_calls.values()),
            external_llm_calls=dict(self._external_llm_calls),
            external_llm_usage=dict(self._external_llm_usage),
            total_external_llm_calls=sum(self._external_llm_calls.values()),
            external_calls_without_usage=self._external_calls_without_usage,
            handoff_count=self._handoff_count,
            handoff_details=list(self._handoff_details),
            rounds=rounds,
            turns=len(self._messages),
            messages=list(self._messages),
        )


# ============================================================
# MetricsLogger — 持久化到 JSONL 文件
# ============================================================

class MetricsLogger:
    """
    将 TaskMetrics 写入本地 JSONL 文件。

    用法:
        logger = MetricsLogger("mas_metrics.jsonl")
        logger.log(metrics)
        summary = logger.load_summary()
    """

    def __init__(self, filepath: str = "mas_metrics.jsonl"):
        self.filepath = filepath

    def log(self, metrics: TaskMetrics, include_messages: bool = False):
        """追加一条记录到 JSONL 文件。

        Args:
            metrics:          TaskMetrics 数据
            include_messages: 是否保存完整消息日志
        """
        data = metrics.to_dict_full() if include_messages else metrics.to_dict()
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_all(self) -> List[TaskMetrics]:
        """读取所有记录"""
        records = []
        if not os.path.exists(self.filepath):
            return records
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    records.append(TaskMetrics(**{
                        k: v for k, v in d.items()
                        if k in TaskMetrics.__dataclass_fields__
                    }))
        return records

    def load_summary(self) -> Dict[str, Any]:
        """加载并汇总所有记录的统计信息"""
        records = self.load_all()
        if not records:
            return {"count": 0}

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_time = sum(r.elapsed for r in records)
        total_tool = sum(r.total_tool_calls for r in records)
        total_handoffs = sum(r.handoff_count for r in records)
        correct = sum(1 for r in records if r.is_correct is True)
        evaluated = sum(1 for r in records if r.is_correct is not None)

        # 每个工具的总调用次数
        tool_totals: Dict[str, int] = {}
        for r in records:
            for tool, count in r.tool_calls.items():
                tool_totals[tool] = tool_totals.get(tool, 0) + count

        # 每个 Agent 的总调用次数
        agent_totals: Dict[str, int] = {}
        for r in records:
            for agent, count in r.agent_calls.items():
                agent_totals[agent] = agent_totals.get(agent, 0) + count

        return {
            "count": len(records),
            "accuracy": round(correct / evaluated, 4) if evaluated > 0 else None,
            "correct": correct,
            "evaluated": evaluated,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "avg_tokens_per_task": round(total_tokens / len(records)),
            "total_time": round(total_time, 1),
            "avg_time_per_task": round(total_time / len(records), 1),
            "total_tool_calls": total_tool,
            "avg_tools_per_task": round(total_tool / len(records), 1),
            "total_handoffs": total_handoffs,
            "tool_breakdown": dict(sorted(tool_totals.items(), key=lambda x: -x[1])),
            "agent_breakdown": dict(sorted(agent_totals.items(), key=lambda x: -x[1])),
        }

    def print_summary(self):
        """打印汇总统计"""
        s = self.load_summary()
        if s["count"] == 0:
            print("[MetricsLogger] No records found.")
            return

        print(f"\n{'='*50}")
        print(f"  MAS Metrics Summary ({self.filepath})")
        print(f"{'='*50}")
        print(f"  Tasks:       {s['count']}")
        if s["accuracy"] is not None:
            print(f"  Accuracy:    {s['correct']}/{s['evaluated']} = {s['accuracy']:.1%}")
        print(f"  Tokens:      {s['total_tokens']:,} total "
              f"(in: {s['total_input_tokens']:,}, out: {s['total_output_tokens']:,})")
        print(f"  Avg tokens:  {s['avg_tokens_per_task']:,} / task")
        print(f"  Total time:  {s['total_time']:.1f}s")
        print(f"  Avg time:    {s['avg_time_per_task']:.1f}s / task")
        print(f"  Tool calls:  {s['total_tool_calls']} total, "
              f"{s['avg_tools_per_task']:.1f} avg/task")
        print(f"  Handoffs:    {s['total_handoffs']} total")
        if s["tool_breakdown"]:
            print(f"  Tools:       {s['tool_breakdown']}")
        if s["agent_breakdown"]:
            print(f"  Agents:      {s['agent_breakdown']}")
        print(f"{'='*50}")


# ============================================================
# 工具函数
# ============================================================

def _check_answer(predicted: str, expected: str) -> bool:
    """简单正确性判断 — expected 出现在 predicted 中即算正确"""
    if not predicted or not expected:
        return False
    predicted = predicted.strip().lower()
    expected = expected.strip().lower()
    # 精确匹配
    if predicted == expected:
        return True
    # 包含匹配
    if expected in predicted:
        return True
    # 数字匹配 (去除逗号/空格)
    p_clean = predicted.replace(",", "").replace(" ", "")
    e_clean = expected.replace(",", "").replace(" ", "")
    if e_clean in p_clean:
        return True
    return False
