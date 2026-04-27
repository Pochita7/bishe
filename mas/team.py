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
import json
import re
import time
from types import SimpleNamespace
from typing import List, Optional, Dict, Any, Tuple

from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

from .metrics import MetricsCollector, MetricsLogger, TaskMetrics
from .response_cache import ResponseCache, stable_digest
from .sentinel import SecurityControlPlane, SecurityEventBus
from token_accounting import push_tracker, reset_tracker, snapshot_tracker


_SUSPENDED_BY_SENTINEL = "Execution suspended by Sentinel"
_SUSPENDED_BY_GUARDIAN = "Execution suspended by Guardian"
_CACHE_SECURITY_VERSION = 2


class _PartialResponse:
    """超时/异常时的 pseudo-response，仅携带 partial usage_details 和空 messages。"""
    def __init__(self, usage_details=None):
        self.usage_details = usage_details
        self.messages = []
        self.text = ""


class _CachedResponse:
    """Pseudo-response for cached agent outputs.

    Usage is zero because no API call was made. Function-call names are replayed
    into a lightweight message shape so existing metrics still see the original
    tool-call count.
    """

    def __init__(self, text: str, tool_names: Optional[List[str]] = None, cache_key: str = ""):
        self.usage_details = {
            "input_token_count": 0,
            "output_token_count": 0,
            "total_token_count": 0,
        }
        self.text = text
        self.cached_response = True
        self.cache_key = cache_key
        contents = [
            SimpleNamespace(type="function_call", name=str(name or "unknown"))
            for name in (tool_names or [])
        ]
        self.messages = [SimpleNamespace(contents=contents)] if contents else []


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
        security_event_bus: Optional[SecurityEventBus] = None,
        sentinel_control_plane: Optional[SecurityControlPlane] = None,
        **kwargs,
    ):
        self.client = client
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.metrics_logger = metrics_logger
        self.worker_timeout = worker_timeout
        self.max_tool_calls_per_worker = max_tool_calls_per_worker
        self.security_event_bus = security_event_bus
        self.sentinel_control_plane = sentinel_control_plane
        self._current_task_id = ""
        self._active_collector: Optional[MetricsCollector] = None
        self._active_external_usage_tracker: Optional[Dict[str, Any]] = None
        self._active_expected_answer = ""
        self._active_planner_rounds = 0
        self.response_cache = ResponseCache.from_env()

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
            default_options=self._agent_default_options(),
        )
        self._agents["_Selector"] = self._selector_agent

    @staticmethod
    def _trim_text(text: Any, limit: int = 300) -> str:
        if text is None:
            return ""
        value = str(text).replace("\n", " ").strip()
        if len(value) <= limit:
            return value
        return value[:limit] + "...(truncated)"

    def _publish_security_event(
        self,
        *,
        source_agent: str,
        target: str,
        behavior_type: str,
        params: Optional[Dict[str, Any]] = None,
        gate: str = "TeamRuntime",
        decision: str = "allow",
        reason: str = "",
        matched_rules: Optional[List[str]] = None,
    ) -> None:
        if self.security_event_bus is None:
            return
        event_params = dict(params or {})
        event_task_id = str(event_params.get("task_id") or self._current_task_id or "")
        if event_task_id and not event_params.get("task_id"):
            event_params["task_id"] = event_task_id
        self.security_event_bus.publish(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_agent": source_agent,
                "target": target,
                "behavior_type": behavior_type,
                "params": event_params,
                "gate": gate,
                "decision": decision,
                "reason": reason,
                "matched_rules": list(matched_rules or []),
                "task_id": event_task_id,
            }
        )

    def _monitor_behavior(
        self,
        *,
        source_agent: str,
        target: str,
        behavior_type: str,
        params: Optional[Dict[str, Any]] = None,
        gate: str = "TeamRuntime",
        enforce: bool = True,
    ) -> Optional[Dict[str, Any]]:
        safe_params = dict(params or {})
        if self._current_task_id and "task_id" not in safe_params:
            safe_params["task_id"] = self._current_task_id
        self._publish_security_event(
            source_agent=source_agent,
            target=target,
            behavior_type=behavior_type,
            params=safe_params,
            gate=gate,
            decision="allow",
        )

        if not enforce or self.sentinel_control_plane is None:
            return None

        directive = self.sentinel_control_plane.get_runtime_directive(
            source_agent=source_agent,
            target=target,
            behavior_type=behavior_type,
            params=safe_params,
        )
        if directive:
            self._publish_security_event(
                source_agent=source_agent,
                target=target,
                behavior_type=behavior_type,
                params=safe_params,
                gate="SentinelEnforcement",
                decision="block",
                reason=str(directive.get("reason", "Blocked by Sentinel")),
                matched_rules=list(directive.get("matched_rules", [])),
            )
        return directive

    @staticmethod
    def _guardian_action(decision: Any) -> str:
        action = getattr(decision, "action", "")
        return str(getattr(action, "value", action) or "").lower()

    @staticmethod
    def _guardian_rules(decision: Any) -> List[str]:
        return list(getattr(decision, "matched_rules", None) or [])

    @staticmethod
    def _guardian_reason(decision: Any) -> str:
        return str(getattr(decision, "reason", "") or "")

    def _guardian_gate(
        self,
        gate: str,
        text: str,
        *,
        source_agent: str,
        target: str,
    ) -> Tuple[str, str, str, List[str]]:
        guardian = getattr(self, "guardian", None)
        if guardian is None:
            return text, "", "", []

        if gate == "input":
            decision = guardian.check_input(text)
        elif gate == "plan":
            decision = guardian.check_plan(text)
        elif gate == "output":
            try:
                decision = guardian.check_output(
                    text,
                    source_agent=source_agent,
                    target=target,
                )
            except TypeError:
                decision = guardian.check_output(text)
        else:
            return text, "", "", []

        action = self._guardian_action(decision)
        reason = self._guardian_reason(decision)
        rules = self._guardian_rules(decision)
        if action == "sanitize":
            effective = (
                getattr(decision, "sanitized_full", "")
                or getattr(decision, "modified_content", "")
                or text
            )
            return str(effective), action, reason, rules
        return text, action, reason, rules

    def _guardian_notice(self, action: str, phase: str, reason: str) -> str:
        reason_text = reason or "security policy"
        if action == "block":
            return f"[Guardian blocked {phase}: {reason_text}]"
        if action == "sanitize":
            return f"[Guardian sanitized {phase}: {reason_text}]"
        return ""

    def _sanitize_context(
        self,
        items: List[Any],
        *,
        label: str,
    ) -> List[Any]:
        if self.sentinel_control_plane is None or not hasattr(
            self.sentinel_control_plane,
            "sanitize_runtime_context",
        ):
            return items
        sanitized, actions = self.sentinel_control_plane.sanitize_runtime_context(
            items,
            self._current_task_id,
        )
        if actions:
            self._publish_security_event(
                source_agent="Sentinel",
                target="MASTeam",
                behavior_type="output_review",
                params={
                    "context_label": label,
                    "sanitized_count": len(actions),
                    "cleanup_actions": actions[:5],
                },
                gate="ContextSanitizer",
                decision="sanitize",
                reason=f"removed {len(actions)} quarantined prompt/output fragment(s)",
                matched_rules=["SENTINEL_CONTEXT_SANITIZED"],
            )
        return sanitized

    def _guardian_review_output(
        self,
        source_agent: str,
        target: str,
        text: str,
        *,
        phase: str,
    ) -> Tuple[str, str, str]:
        reviewed_text, action, reason, _rules = self._guardian_gate(
            "output",
            text,
            source_agent=source_agent,
            target=target,
        )
        if action == "block":
            return self._guardian_notice("block", phase, reason), action, reason
        if action == "sanitize":
            return reviewed_text, action, reason
        return text, action or "allow", reason

    def _agent_cache_fingerprint(self, agent_name: str) -> Dict[str, Any]:
        agent = self._agents.get(agent_name)
        tool_specs: List[Dict[str, str]] = []
        for tool in getattr(agent, "tools", None) or []:
            tool_specs.append(
                {
                    "name": str(getattr(tool, "name", "") or ""),
                    "description_hash": stable_digest(
                        str(getattr(tool, "description", "") or ""),
                        digest_size=8,
                    ),
                }
            )

        model_name = ""
        for attr in ("model", "model_name", "model_id", "deployment_name"):
            value = getattr(self.client, attr, None)
            if value:
                model_name = str(value)
                break
        if not model_name:
            try:
                from gaia_solver.config import MODEL_NAME  # type: ignore

                model_name = str(MODEL_NAME)
            except Exception:
                model_name = ""

        instructions = str(
            getattr(agent, "instructions", "")
            or getattr(agent, "_instructions", "")
            or ""
        )
        options = getattr(agent, "default_options", None) or getattr(agent, "_default_options", None) or {}
        try:
            options_text = json.dumps(options, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            options_text = repr(options)
        security_policy = self._cache_security_policy_fingerprint()
        return {
            "cache_security_version": _CACHE_SECURITY_VERSION,
            "agent": agent_name,
            "model": model_name,
            "instructions_hash": stable_digest(instructions, digest_size=12),
            "options_hash": stable_digest(options_text, digest_size=8),
            "tools": sorted(tool_specs, key=lambda item: item["name"]),
            "guardian": bool(getattr(self, "guardian", None)),
            "sentinel": bool(self.sentinel_control_plane),
            "security_policy": security_policy,
            "max_tool_calls_per_worker": self.max_tool_calls_per_worker,
        }

    def _cache_security_policy_fingerprint(self) -> Dict[str, Any]:
        if self.sentinel_control_plane is None:
            return {"policy_hash": "", "task_id": self._current_task_id or ""}
        try:
            snapshot = self.sentinel_control_plane.snapshot()
        except Exception:
            return {"policy_hash": "snapshot_error", "task_id": self._current_task_id or ""}

        task_id = self._current_task_id or ""
        active = []
        for item in snapshot.get("active_mitigations", []) or []:
            scope = str(item.get("scope", "") or "")
            if scope and task_id and scope != task_id:
                continue
            active.append(
                {
                    "kind": str(item.get("kind", "") or ""),
                    "target": str(item.get("target", "") or ""),
                    "scope": scope,
                    "persistent": bool(item.get("persistent", False)),
                }
            )
        policy = {
            "active_mitigations": sorted(
                active,
                key=lambda item: (item["kind"], item["target"], item["scope"]),
            ),
            "blocked_domains": sorted(snapshot.get("blocked_domains", []) or []),
            "blocked_sources": sorted(snapshot.get("blocked_sources", []) or []),
            "task_scoped_blocked_domains": sorted(
                (snapshot.get("task_scoped_blocked_domains", {}) or {}).get(task_id, []) or []
            ),
            "task_scoped_blocked_sources": sorted(
                (snapshot.get("task_scoped_blocked_sources", {}) or {}).get(task_id, []) or []
            ),
        }
        return {
            "task_id": task_id,
            "policy_hash": stable_digest(policy, digest_size=12),
        }

    @staticmethod
    def _replay_cached_tool_log(tool_names: List[str]) -> None:
        if not tool_names:
            return
        try:
            from tamas_adapter.tools import replay_tool_call_log  # type: ignore

            replay_tool_call_log(tool_names)
        except Exception:
            return

    def _validate_cached_tool_replay(
        self,
        agent_name: str,
        tool_names: List[str],
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        if not tool_names:
            return None

        guardian = getattr(self, "guardian", None)
        for tool_name_raw in tool_names:
            tool_name = str(tool_name_raw or "unknown")
            params = {
                "tool_name": tool_name,
                "source_agent": agent_name,
                "cached_replay": True,
                "cache_key": cache_key,
            }
            if self._current_task_id:
                params["task_id"] = self._current_task_id

            if guardian is not None and hasattr(guardian, "check_tool_call"):
                decision = guardian.check_tool_call(
                    tool_name,
                    dict(params),
                    source_agent=agent_name,
                )
                action = self._guardian_action(decision)
                if action == "block":
                    return {
                        "action": "block",
                        "source": "Guardian",
                        "reason": self._guardian_reason(decision) or "cached tool replay blocked",
                        "matched_rules": self._guardian_rules(decision),
                    }
                if self.sentinel_control_plane is not None:
                    directive = self.sentinel_control_plane.get_runtime_directive(
                        source_agent=agent_name,
                        target=tool_name,
                        behavior_type="tool_call",
                        params=params,
                    )
                    if directive:
                        return directive
                continue

            directive = self._monitor_behavior(
                source_agent=agent_name,
                target=tool_name,
                behavior_type="tool_call",
                params=params,
                gate="CacheReplay",
                enforce=True,
            )
            if directive:
                return directive
        return None

    @staticmethod
    def _agent_default_options() -> Dict[str, Any]:
        try:
            from gaia_solver.config import get_chat_default_options

            return get_chat_default_options(temperature=0)
        except Exception:
            return {"temperature": 0}

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
            default_options=self._agent_default_options(),
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
                default_options=self._agent_default_options(),
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

    async def _select_worker_llm(
        self,
        step_desc: str,
        collector: Optional[MetricsCollector] = None,
    ) -> Optional[str]:
        """PLAN 解析失败时，使用 LLM 选择 Worker"""
        prompt = f"Task: {step_desc}\nWhich worker should handle this?"
        try:
            t0 = time.time()
            text, result = await self._invoke_agent("_Selector", prompt, timeout=30)
            if collector is not None:
                collector.record_agent_call(
                    "_Selector",
                    result,
                    time.time() - t0,
                    is_planner=True,
                )
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
        """调用一个 Agent，返回 (文本输出, AgentResponse).
        
        即使 Agent 超时，也会通过 _token_usage_tracker contextvar
        收集已消耗的 token 数据，避免超时导致 token 统计丢失。
        """
        agent = self._agents.get(agent_name)
        if not agent:
            return f"[Error: unknown agent '{agent_name}']", None

        cache_key = ""
        cache_fingerprint: Dict[str, Any] = {}
        if self.response_cache.enabled:
            cache_fingerprint = self._agent_cache_fingerprint(agent_name)
            cache_key = self.response_cache.make_key(
                agent_name=agent_name,
                prompt=prompt,
                fingerprint=cache_fingerprint,
            )
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                tool_names = list(cached.get("tool_names") or [])
                cache_directive = self._validate_cached_tool_replay(
                    agent_name,
                    tool_names,
                    cache_key,
                )
                if cache_directive:
                    source = str(cache_directive.get("source", "Sentinel") or "Sentinel")
                    reason = str(cache_directive.get("reason", "cached tool replay blocked") or "cached tool replay blocked")
                    blocked_text = f"[{source} blocked cached response for {agent_name}: {reason}]"
                    if self.verbose:
                        print(f"    [CACHE BLOCKED] {agent_name}: {reason}")
                    return (
                        blocked_text,
                        _CachedResponse(
                            blocked_text,
                            tool_names=[],
                            cache_key=cache_key,
                        ),
                    )
                self._replay_cached_tool_log(tool_names)
                if self.verbose:
                    print(f"    [CACHE HIT] {agent_name}")
                cached_text = str(cached.get("text", ""))
                return (
                    cached_text,
                    _CachedResponse(
                        cached_text,
                        tool_names=tool_names,
                        cache_key=cache_key,
                    ),
                )

        # 设置 token 追踪器 — FunctionInvocationLayer 会在每次 API 调用后更新它
        from agent_framework._tools import _token_usage_tracker
        usage_tracker: dict = {'usage': None}
        token = _token_usage_tracker.set(usage_tracker)
        invoke_start = time.time()

        try:
            result = await asyncio.wait_for(agent.run(prompt), timeout=timeout)
            text = result.text if hasattr(result, "text") else str(result)

            # ---- 工具调用上限检查 ----
            tool_names: List[str] = []
            messages = getattr(result, "messages", [])
            for msg in messages:
                contents = getattr(msg, "contents", [])
                for content in contents:
                    if getattr(content, "type", None) == "function_call":
                        tool_names.append(str(getattr(content, "name", None) or "unknown"))
            tool_count = len(tool_names)

            if tool_count > self.max_tool_calls_per_worker and self.verbose:
                print(f"    [WARN] {agent_name} made {tool_count} tool calls (limit: {self.max_tool_calls_per_worker})")

            if cache_key:
                self.response_cache.put(
                    cache_key,
                    agent_name=agent_name,
                    prompt=prompt,
                    fingerprint=cache_fingerprint,
                    text=text,
                    tool_names=tool_names,
                )

            return text, result
        except asyncio.TimeoutError:
            # 超时时，从 tracker 中获取已消耗的 partial token 数据
            partial_usage = usage_tracker.get('usage')
            if partial_usage and self.verbose:
                inp = partial_usage.get('input_token_count', 0) or 0
                out = partial_usage.get('output_token_count', 0) or 0
                print(f"    [{agent_name}] timeout, partial tokens: in={inp}, out={out}")
            # 构造一个 pseudo-response 以便 record_agent_call 能记录 partial token
            pseudo_resp = _PartialResponse(usage_details=partial_usage)
            return "[Timeout]", pseudo_resp
        except asyncio.CancelledError:
            partial_usage = usage_tracker.get('usage')
            if partial_usage and self._active_collector is not None:
                self._active_collector.record_agent_call(
                    agent_name,
                    _PartialResponse(usage_details=partial_usage),
                    time.time() - invoke_start,
                    is_planner=(agent_name == self._planner_name or agent_name == "_Selector"),
                )
            raise
        except Exception as e:
            partial_usage = usage_tracker.get('usage')
            pseudo_resp = _PartialResponse(usage_details=partial_usage) if partial_usage else None
            return f"[Error: {e}]", pseudo_resp
        finally:
            _token_usage_tracker.reset(token)

    def get_partial_metrics(
        self,
        answer: str = "",
        expected_answer: Optional[str] = None,
    ) -> Optional[TaskMetrics]:
        """Return metrics collected so far when an outer benchmark timeout cancels run()."""
        if self._active_collector is None:
            return None
        self._active_collector.sync_external_usage(
            snapshot_tracker(self._active_external_usage_tracker)
        )
        return self._active_collector.finalize(
            answer=answer,
            expected_answer=self._active_expected_answer if expected_answer is None else expected_answer,
            rounds=self._active_planner_rounds,
        )

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
            self._publish_security_event(
                source_agent=worker_name,
                target=target_name,
                behavior_type="handoff",
                params={
                    "task_preview": self._trim_text(handoff_task, 240),
                    "original_task": self._trim_text(task, 240),
                },
                gate="HandoffMonitor",
                decision="allow",
            )
            if self.verbose:
                print(f"    ↳ HANDOFF → {target_name}: {handoff_task[:120]}")

            directive = self._monitor_behavior(
                source_agent=target_name,
                target=worker_name,
                behavior_type="agent_message",
                params={
                    "assignment_preview": self._trim_text(handoff_task, 240),
                    "origin": "handoff",
                },
                gate="WorkerDispatch",
            )
            if directive:
                blocked_text = f"[Sentinel blocked {target_name}: {directive.get('reason', 'execution suspended')}]"
                history.append({"source": "Sentinel", "content": blocked_text})
                worker_results.append((target_name, blocked_text))
                if self.verbose:
                    print(f"    [Sentinel]: {blocked_text}")
                break

            worker_results[:] = self._sanitize_context(
                worker_results,
                label="handoff_worker_results",
            )
            history[:] = self._sanitize_context(history, label="handoff_history")
            handoff_prompt = self._build_worker_prompt(task, handoff_task, worker_results)
            t0 = time.time()
            worker_text, worker_resp = await self._invoke_agent(
                target_name, handoff_prompt, timeout=timeout
            )
            collector.record_agent_call(
                target_name, worker_resp, time.time() - t0, is_planner=False
            )
            worker_text, output_action, output_reason = self._guardian_review_output(
                target_name,
                self._planner_name,
                worker_text,
                phase=f"handoff result from {target_name}",
            )
            if output_action in {"sanitize", "block"}:
                notice = self._guardian_notice(
                    output_action,
                    f"handoff result from {target_name}",
                    output_reason,
                )
                history.append({"source": "Guardian", "content": notice})
                if self.verbose:
                    print(f"    [Guardian]: {notice}")
                if output_action == "block":
                    worker_results.append((target_name, worker_text))
                    break
            result_directive = self._monitor_behavior(
                source_agent=target_name,
                target=self._planner_name,
                behavior_type="worker_result",
                params={
                    "result_preview": self._trim_text(worker_text, 320),
                    "handoff_from": worker_name,
                },
                gate="ResultMonitor",
            )
            if result_directive:
                blocked_text = (
                    f"[Sentinel blocked result from {target_name}: "
                    f"{result_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": blocked_text})
                worker_results.append((target_name, blocked_text))
                if self.verbose:
                    print(f"    [Sentinel]: {blocked_text}")
                break

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
        self._current_task_id = task_id or f"session-{int(start * 1000)}"
        guardian = getattr(self, "guardian", None)
        if guardian is not None and hasattr(guardian, "set_current_task_id"):
            guardian.set_current_task_id(self._current_task_id)
        sentinel = getattr(self, "sentinel", None)
        if sentinel is not None and hasattr(sentinel, "set_current_task_id"):
            sentinel.set_current_task_id(self._current_task_id)
        history: List[Dict[str, str]] = []
        final_answer = ""
        planner_rounds = 0

        # ---- 指标收集器 ----
        collector = MetricsCollector(task_id=task_id, task=task)
        self._active_collector = collector
        external_usage_token, external_usage_tracker = push_tracker()
        self._active_external_usage_tracker = external_usage_tracker
        self._active_expected_answer = expected_answer
        self._active_planner_rounds = 0

        if self.verbose:
            print(f"\n[MASTeam] Starting task...")
            print(f"  Agents: {[a.name for a in self._agent_list]}")

        task_input_decision = "allow"
        task_input_reason = ""
        task_input_rules: List[str] = []
        task_input_params: Dict[str, Any] = {
            "task_id": task_id,
            "task_preview": self._trim_text(task, 320),
        }
        task, guardian_action, guardian_reason, guardian_rules = self._guardian_gate(
            "input",
            task,
            source_agent="User",
            target=self._planner_name,
        )
        if guardian_action == "sanitize":
            collector.task = task
            task_input_decision = "sanitize"
            task_input_reason = guardian_reason or "Guardian sanitized task input"
            task_input_rules = list(guardian_rules)
            task_input_params = {
                "task_id": task_id,
                "task_preview": self._trim_text(task, 320),
                "sanitized": True,
                "sanitization_reason": task_input_reason,
                "guardian_gate": "InputGate",
            }
            notice = self._guardian_notice("sanitize", "task input", task_input_reason)
            history.append({"source": "Guardian", "content": notice})
            if self.verbose:
                print(f"  [Guardian]: {notice}")
        elif guardian_action == "block":
            task_input_decision = "block"
            task_input_reason = guardian_reason or "Guardian blocked task input"
            task_input_rules = list(guardian_rules)
            task_input_params = {
                "task_id": task_id,
                "task_preview": "[withheld_by_guardian]",
                "guardian_gate": "InputGate",
            }
            notice = self._guardian_notice("block", "task input", task_input_reason)
            history.append({"source": "Guardian", "content": notice})
            final_answer = _SUSPENDED_BY_GUARDIAN
            if self.verbose:
                print(f"  [Guardian]: {notice}")

        if not final_answer and sentinel is not None and hasattr(sentinel, "review_task_input"):
            task_review = sentinel.review_task_input(
                task,
                task_id=self._current_task_id,
                source_agent="User",
                target=self._planner_name,
            )
            if task_review.action == "sanitize":
                task = task_review.effective_task
                collector.task = task
                task_input_decision = "sanitize"
                task_input_reason = "; ".join(task_review.reasons[:2]) or "sanitized task input"
                task_input_rules = list(task_review.matched_rules)
                task_input_params = {
                    "task_id": task_id,
                    "task_preview": self._trim_text(task, 320),
                    "sanitized": True,
                    "detected_markers": list(task_review.detected_markers),
                    "sanitization_reason": task_input_reason,
                }
                sanitized_text = f"[Sentinel sanitized task input: {task_input_reason}]"
                history.append({"source": "Sentinel", "content": sanitized_text})
                if self.verbose:
                    print(f"  [Sentinel]: {sanitized_text}")
            elif task_review.action == "hitl":
                task_input_decision = "block"
                task_input_reason = "; ".join(task_review.reasons[:2]) or "task input requires human review"
                task_input_rules = list(task_review.matched_rules)
                task_input_params = {
                    "task_id": task_id,
                    "task_preview": "[withheld_for_hitl]",
                    "detected_markers": list(task_review.detected_markers),
                    "sanitization_reason": task_input_reason,
                }
                blocked_text = f"[Sentinel paused task input: {task_input_reason}]"
                history.append({"source": "Sentinel", "content": blocked_text})
                final_answer = _SUSPENDED_BY_SENTINEL
                if self.verbose:
                    print(f"  [Sentinel]: {blocked_text}")

        self._publish_security_event(
            source_agent="User",
            target=self._planner_name,
            behavior_type="task_input",
            params=task_input_params,
            gate="TaskIngress",
            decision=task_input_decision,
            reason=task_input_reason,
            matched_rules=task_input_rules,
        )

        last_plan_text = ""
        worker_results: List[Tuple[str, str]] = []

        for round_num in range(0 if final_answer else self.max_rounds):
            # ========== 1. Planner 规划/审查 ==========
            if round_num == 0:
                planner_prompt = self._build_planner_plan_prompt(task)
            else:
                history = self._sanitize_context(history, label="planner_history")
                worker_results = self._sanitize_context(
                    worker_results,
                    label="planner_review_worker_results",
                )
                planner_prompt = self._build_planner_review_prompt(
                    task, last_plan_text, worker_results
                )

            planner_directive = self._monitor_behavior(
                source_agent=self._planner_name,
                target="MASTeam",
                behavior_type="agent_message",
                params={
                    "round": round_num + 1,
                    "prompt_preview": self._trim_text(planner_prompt, 320),
                    "task_id": task_id,
                },
                gate="PlannerDispatch",
            )
            if planner_directive:
                final_answer = _SUSPENDED_BY_SENTINEL
                blocked_text = (
                    f"[Sentinel blocked {self._planner_name}: "
                    f"{planner_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": blocked_text})
                if self.verbose:
                    print(f"\n  [Sentinel]: {blocked_text}")
                break

            t0 = time.time()
            planner_text, planner_resp = await self._invoke_agent(
                self._planner_name, planner_prompt, timeout=60
            )
            collector.record_agent_call(
                self._planner_name, planner_resp, time.time() - t0, is_planner=True
            )
            planner_rounds += 1
            self._active_planner_rounds = planner_rounds

            planner_text, plan_guardian_action, plan_guardian_reason, _plan_guardian_rules = self._guardian_gate(
                "plan",
                planner_text,
                source_agent=self._planner_name,
                target="MASTeam",
            )
            if plan_guardian_action in {"sanitize", "block"}:
                notice = self._guardian_notice(
                    plan_guardian_action,
                    "planner output",
                    plan_guardian_reason,
                )
                history.append({"source": "Guardian", "content": notice})
                if self.verbose:
                    print(f"  [Guardian]: {notice}")
                if plan_guardian_action == "block":
                    final_answer = _SUSPENDED_BY_GUARDIAN
                    break

            fa = self._extract_final_answer(planner_text)
            if fa and round_num > 0:
                planner_text, final_guardian_action, final_guardian_reason = self._guardian_review_output(
                    self._planner_name,
                    "User",
                    planner_text,
                    phase="planner final answer",
                )
                if final_guardian_action in {"sanitize", "block"}:
                    notice = self._guardian_notice(
                        final_guardian_action,
                        "planner final answer",
                        final_guardian_reason,
                    )
                    history.append({"source": "Guardian", "content": notice})
                    if self.verbose:
                        print(f"  [Guardian]: {notice}")
                    if final_guardian_action == "block":
                        final_answer = _SUSPENDED_BY_GUARDIAN
                        break
                fa = self._extract_final_answer(planner_text)
            planner_result_directive = self._monitor_behavior(
                source_agent=self._planner_name,
                target="MASTeam",
                behavior_type="final_answer" if fa and round_num > 0 else "plan_review",
                params={
                    "round": round_num + 1,
                    "content_preview": self._trim_text(planner_text, 320),
                    "parsed_steps": len(self._parse_plan(planner_text)),
                },
                gate="PlannerResult",
            )
            if planner_result_directive:
                final_answer = _SUSPENDED_BY_SENTINEL
                blocked_text = (
                    f"[Sentinel blocked {self._planner_name}: "
                    f"{planner_result_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": blocked_text})
                if self.verbose:
                    print(f"\n  [Sentinel]: {blocked_text}")
                break

            history.append({"source": self._planner_name, "content": planner_text})

            if self.verbose:
                preview = planner_text[:300].replace("\n", " ")
                print(f"\n  [Round {round_num + 1}] [{self._planner_name}]: {preview}")

            # 检查 FINAL_ANSWER（首轮跳过: 强制先走 Worker）
            if fa and round_num > 0:
                final_answer = fa
                break

            last_plan_text = planner_text

            # ========== 2. 解析 PLAN → Worker 步骤列表 ==========
            steps = self._parse_plan(planner_text)

            if not steps:
                if self.verbose:
                    print("  [WARN] Could not parse PLAN steps, trying LLM selector...")
                worker_name = await self._select_worker_llm(task, collector)
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

                    assignment_directive = self._monitor_behavior(
                        source_agent=self._planner_name,
                        target=worker_name,
                        behavior_type="worker_assignment",
                        params={
                            "round": round_num + 1,
                            "step_preview": self._trim_text(step_desc, 240),
                            "parallel_group_size": 1,
                        },
                        gate="PlanDispatch",
                    )
                    if assignment_directive:
                        blocked_text = (
                            f"[Sentinel blocked assignment to {worker_name}: "
                            f"{assignment_directive.get('reason', 'execution suspended')}]"
                        )
                        history.append({"source": "Sentinel", "content": blocked_text})
                        worker_results.append((worker_name, blocked_text))
                        if self.verbose:
                            print(f"    [Sentinel]: {blocked_text}")
                        continue
                    worker_directive = self._monitor_behavior(
                        source_agent=worker_name,
                        target=self._planner_name,
                        behavior_type="agent_message",
                        params={
                            "assignment_preview": self._trim_text(step_desc, 240),
                            "round": round_num + 1,
                        },
                        gate="WorkerDispatch",
                    )
                    if worker_directive:
                        blocked_text = (
                            f"[Sentinel blocked {worker_name}: "
                            f"{worker_directive.get('reason', 'execution suspended')}]"
                        )
                        history.append({"source": "Sentinel", "content": blocked_text})
                        worker_results.append((worker_name, blocked_text))
                        if self.verbose:
                            print(f"    [Sentinel]: {blocked_text}")
                        continue

                    worker_results = self._sanitize_context(
                        worker_results,
                        label="worker_prompt_results",
                    )
                    history = self._sanitize_context(history, label="worker_prompt_history")
                    worker_prompt = self._build_worker_prompt(task, step_desc, worker_results)
                    t0 = time.time()
                    worker_text, worker_resp = await self._invoke_agent(worker_name, worker_prompt, timeout=wk_timeout)
                    collector.record_agent_call(
                        worker_name, worker_resp, time.time() - t0, is_planner=False
                    )
                    worker_text, output_action, output_reason = self._guardian_review_output(
                        worker_name,
                        self._planner_name,
                        worker_text,
                        phase=f"worker result from {worker_name}",
                    )
                    if output_action in {"sanitize", "block"}:
                        notice = self._guardian_notice(
                            output_action,
                            f"worker result from {worker_name}",
                            output_reason,
                        )
                        history.append({"source": "Guardian", "content": notice})
                        if self.verbose:
                            print(f"    [Guardian]: {notice}")
                        if output_action == "block":
                            worker_results.append((worker_name, worker_text))
                            continue
                    result_directive = self._monitor_behavior(
                        source_agent=worker_name,
                        target=self._planner_name,
                        behavior_type="worker_result",
                        params={
                            "result_preview": self._trim_text(worker_text, 320),
                            "round": round_num + 1,
                        },
                        gate="ResultMonitor",
                    )
                    if result_directive:
                        blocked_text = (
                            f"[Sentinel blocked result from {worker_name}: "
                            f"{result_directive.get('reason', 'execution suspended')}]"
                        )
                        history.append({"source": "Sentinel", "content": blocked_text})
                        worker_results.append((worker_name, blocked_text))
                        if self.verbose:
                            print(f"    [Sentinel]: {blocked_text}")
                        continue

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
                        assignment_directive = self._monitor_behavior(
                            source_agent=self._planner_name,
                            target=wname,
                            behavior_type="worker_assignment",
                            params={
                                "round": round_num + 1,
                                "step_preview": self._trim_text(desc, 240),
                                "parallel_group_size": len(group),
                            },
                            gate="PlanDispatch",
                        )
                        if assignment_directive:
                            text = (
                                f"[Sentinel blocked assignment to {wname}: "
                                f"{assignment_directive.get('reason', 'execution suspended')}]"
                            )
                            return wname, text, None, 0.0
                        directive = self._monitor_behavior(
                            source_agent=wname,
                            target=self._planner_name,
                            behavior_type="agent_message",
                            params={
                                "assignment_preview": self._trim_text(desc, 240),
                                "round": round_num + 1,
                            },
                            gate="WorkerDispatch",
                        )
                        if directive:
                            text = (
                                f"[Sentinel blocked {wname}: "
                                f"{directive.get('reason', 'execution suspended')}]"
                            )
                            return wname, text, None, 0.0
                        prompt_results = self._sanitize_context(
                            list(worker_results),
                            label="parallel_worker_prompt_results",
                        )
                        prompt = self._build_worker_prompt(task, desc, prompt_results)
                        t0 = time.time()
                        text, resp = await self._invoke_agent(wname, prompt, timeout=wk_timeout)
                        elapsed_s = time.time() - t0
                        if resp is not None:
                            collector.record_agent_call(wname, resp, elapsed_s, is_planner=False)
                        text, _output_action, _output_reason = self._guardian_review_output(
                            wname,
                            self._planner_name,
                            text,
                            phase=f"worker result from {wname}",
                        )
                        return wname, text, None, elapsed_s

                    tasks_list = [_run_step(d, w) for d, w in group]
                    results_list = await asyncio.gather(*tasks_list, return_exceptions=True)

                    for item in results_list:
                        if isinstance(item, Exception):
                            if self.verbose:
                                print(f"    [ERROR] Parallel step failed: {item}")
                            continue
                        wname, wtext, wresp, welapsed = item
                        if wresp is not None:
                            collector.record_agent_call(wname, wresp, welapsed, is_planner=False)
                        result_directive = self._monitor_behavior(
                            source_agent=wname,
                            target=self._planner_name,
                            behavior_type="worker_result",
                            params={
                                "result_preview": self._trim_text(wtext, 320),
                                "round": round_num + 1,
                            },
                            gate="ResultMonitor",
                        )
                        if result_directive:
                            blocked_text = (
                                f"[Sentinel blocked result from {wname}: "
                                f"{result_directive.get('reason', 'execution suspended')}]"
                            )
                            history.append({"source": "Sentinel", "content": blocked_text})
                            worker_results.append((wname, blocked_text))
                            if self.verbose:
                                print(f"    [Sentinel]: {blocked_text}")
                            continue

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
            history = self._sanitize_context(history, label="planner_review_history")
            worker_results = self._sanitize_context(
                worker_results,
                label="planner_review_worker_results",
            )
            review_prompt = self._build_planner_review_prompt(task, last_plan_text, worker_results)
            planner_directive = self._monitor_behavior(
                source_agent=self._planner_name,
                target="MASTeam",
                behavior_type="agent_message",
                params={
                    "round": round_num + 1,
                    "prompt_preview": self._trim_text(review_prompt, 320),
                    "phase": "review",
                    "task_id": task_id,
                },
                gate="PlannerDispatch",
            )
            if planner_directive:
                final_answer = _SUSPENDED_BY_SENTINEL
                blocked_text = (
                    f"[Sentinel blocked {self._planner_name}: "
                    f"{planner_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": blocked_text})
                if self.verbose:
                    print(f"  [Sentinel]: {blocked_text}")
                break
            t0 = time.time()
            planner_review, review_resp = await self._invoke_agent(self._planner_name, review_prompt, timeout=60)
            collector.record_agent_call(
                self._planner_name, review_resp, time.time() - t0, is_planner=True
            )
            planner_rounds += 1
            self._active_planner_rounds = planner_rounds
            planner_review, review_guardian_action, review_guardian_reason, _review_guardian_rules = self._guardian_gate(
                "output",
                planner_review,
                source_agent=self._planner_name,
                target="User",
            )
            if review_guardian_action in {"sanitize", "block"}:
                notice = self._guardian_notice(
                    review_guardian_action,
                    "planner review output",
                    review_guardian_reason,
                )
                history.append({"source": "Guardian", "content": notice})
                if self.verbose:
                    print(f"  [Guardian]: {notice}")
                if review_guardian_action == "block":
                    final_answer = _SUSPENDED_BY_GUARDIAN
                    break
            review_answer = self._extract_final_answer(planner_review)
            review_result_directive = self._monitor_behavior(
                source_agent=self._planner_name,
                target="MASTeam",
                behavior_type="final_answer" if review_answer else "plan_review",
                params={
                    "round": round_num + 1,
                    "content_preview": self._trim_text(planner_review, 320),
                    "phase": "review",
                },
                gate="PlannerResult",
            )
            if review_result_directive:
                final_answer = _SUSPENDED_BY_SENTINEL
                blocked_text = (
                    f"[Sentinel blocked {self._planner_name}: "
                    f"{review_result_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": blocked_text})
                if self.verbose:
                    print(f"  [Sentinel]: {blocked_text}")
                break

            history.append({"source": self._planner_name, "content": planner_review})

            if self.verbose:
                preview = planner_review[:300].replace("\n", " ")
                print(f"  [{self._planner_name}] review: {preview}")

            # 检查 FINAL_ANSWER
            fa = review_answer
            if fa:
                final_answer = fa
                break

            # 没有 FINAL_ANSWER → 把 review 当作新 plan，进入下一轮
            last_plan_text = planner_review

        # ========== 5. 兜底: 强制要求 FINAL_ANSWER ==========
        if not final_answer:
            history = self._sanitize_context(history, label="forced_answer_history")
            recent_context = "\n".join(
                f"[{m['source']}]: {m['content'][:400]}" for m in history[-6:]
            )
            force_prompt = (
                f"Task: {task}\n\n"
                f"Gathered information:\n{recent_context}\n\n"
                "You MUST output FINAL_ANSWER: <answer> NOW based on everything above."
            )
            planner_directive = self._monitor_behavior(
                source_agent=self._planner_name,
                target="User",
                behavior_type="agent_message",
                params={
                    "prompt_preview": self._trim_text(force_prompt, 320),
                    "phase": "forced_answer",
                    "task_id": task_id,
                },
                gate="PlannerDispatch",
            )
            if planner_directive:
                forced = (
                    f"[Sentinel blocked {self._planner_name}: "
                    f"{planner_directive.get('reason', 'execution suspended')}]"
                )
                history.append({"source": "Sentinel", "content": forced})
                final_answer = _SUSPENDED_BY_SENTINEL
            else:
                t0 = time.time()
                forced, forced_resp = await self._invoke_agent(self._planner_name, force_prompt, timeout=60)
                collector.record_agent_call(
                    self._planner_name, forced_resp, time.time() - t0, is_planner=True
                )
                planner_rounds += 1
                self._active_planner_rounds = planner_rounds
                forced, forced_guardian_action, forced_guardian_reason = self._guardian_review_output(
                    self._planner_name,
                    "User",
                    forced,
                    phase="forced final answer",
                )
                if forced_guardian_action in {"sanitize", "block"}:
                    notice = self._guardian_notice(
                        forced_guardian_action,
                        "forced final answer",
                        forced_guardian_reason,
                    )
                    history.append({"source": "Guardian", "content": notice})
                    if self.verbose:
                        print(f"  [Guardian]: {notice}")
                    if forced_guardian_action == "block":
                        final_answer = _SUSPENDED_BY_GUARDIAN
                        forced = notice
                if not final_answer:
                    forced_answer = self._extract_final_answer(forced)
                    forced_result_directive = self._monitor_behavior(
                        source_agent=self._planner_name,
                        target="User",
                        behavior_type="final_answer",
                        params={
                            "content_preview": self._trim_text(forced, 320),
                            "phase": "forced_answer",
                        },
                        gate="PlannerResult",
                    )
                    if forced_result_directive:
                        forced = (
                            f"[Sentinel blocked {self._planner_name}: "
                            f"{forced_result_directive.get('reason', 'execution suspended')}]"
                        )
                        history.append({"source": "Sentinel", "content": forced})
                        final_answer = _SUSPENDED_BY_SENTINEL
                    else:
                        history.append({"source": self._planner_name, "content": forced})
                        final_answer = forced_answer

            if self.verbose:
                preview = forced[:300].replace("\n", " ")
                print(f"  [{self._planner_name}] forced: {preview}")

        elapsed = time.time() - start

        # ---- 生成 TaskMetrics ----
        collector.sync_external_usage(snapshot_tracker(external_usage_tracker))
        metrics = collector.finalize(
            answer=final_answer or "",
            expected_answer=expected_answer,
            rounds=planner_rounds,
        )
        reset_tracker(external_usage_token)
        self._active_external_usage_tracker = None
        self._active_collector = None

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
            if hasattr(self, "sentinel") and getattr(self, "sentinel", None) is not None:
                print(f"  Sentinel: {getattr(self, 'sentinel').status}")

        return {
            "answer": final_answer or "",
            "messages": history,
            "turns": len(history),
            "rounds": planner_rounds,
            "elapsed": round(elapsed, 1),
            "metrics": metrics,
            "sentinel_status": getattr(getattr(self, "sentinel", None), "status", None),
            "response_cache": self.response_cache.stats(),
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
