"""Sentinel Agent for cross-agent behavior monitoring and adaptive intervention.

This module is intentionally independent from business agents.
It consumes structured security events (for example, events emitted by Guardian),
learns baseline patterns, detects anomalies, quantifies risk, and returns
intervention decisions.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple


class SentinelAction(Enum):
    ALLOW = "allow"
    ALERT = "alert"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    HITL = "hitl"


@dataclass
class SecurityEvent:
    timestamp: str
    source_agent: str
    target: str
    behavior_type: str
    params: Dict[str, Any]
    gate: str
    decision: str
    reason: str = ""
    matched_rules: Optional[List[str]] = None
    scenario: str = ""
    task_id: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SecurityEvent":
        params = dict(payload.get("params") or {})
        task_id = (
            payload.get("task_id")
            or params.get("task_id")
            or params.get("session_id")
            or ""
        )
        return cls(
            timestamp=payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            source_agent=payload.get("source_agent", "unknown"),
            target=payload.get("target", "unknown"),
            behavior_type=payload.get("behavior_type", "unknown"),
            params=params,
            gate=payload.get("gate", "UnknownGate"),
            decision=payload.get("decision", "allow"),
            reason=payload.get("reason", ""),
            matched_rules=list(payload.get("matched_rules") or []),
            scenario=payload.get("scenario", ""),
            task_id=str(task_id),
        )


@dataclass
class SentinelAssessment:
    event: SecurityEvent
    risk_score: float
    action: SentinelAction
    reasons: List[str]
    anomalies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["event"] = asdict(self.event)
        return data


@dataclass
class HitlTicket:
    ticket_id: str
    created_at: str
    source_agent: str
    target: str
    behavior_type: str
    risk_score: float
    status: str = "pending"
    reason: str = ""
    notes: str = ""
    task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecurityControlPlane:
    """Runtime enforcement plane for Sentinel decisions.

    Guardian can query this control plane *before* allowing new actions.
    """

    def __init__(self):
        self.blocked_tools: set[str] = set()
        self.quarantined_agents: set[str] = set()
        self.paused_agents: set[str] = set()  # HITL pending
        self.blocked_tools_by_task: Dict[str, set[str]] = defaultdict(set)
        self.quarantined_agents_by_task: Dict[str, set[str]] = defaultdict(set)
        self.paused_agents_by_task: Dict[str, set[str]] = defaultdict(set)
        self._hitl_tickets: Dict[str, HitlTicket] = {}
        self._alerts: List[Dict[str, Any]] = []

    def apply_assessment(self, assessment: "SentinelAssessment") -> None:
        event = assessment.event
        action = assessment.action
        task_id = self._event_task_id(event)

        if action in {
            SentinelAction.ALERT,
            SentinelAction.BLOCK,
            SentinelAction.QUARANTINE,
            SentinelAction.HITL,
        }:
            self._alerts.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_agent": event.source_agent,
                    "target": event.target,
                    "behavior_type": event.behavior_type,
                    "risk_score": assessment.risk_score,
                    "action": action.value,
                    "task_id": task_id,
                    "reasons": list(assessment.reasons),
                }
            )

        if action == SentinelAction.BLOCK:
            self._block_agent(event.source_agent, task_id)
            if event.behavior_type == "tool_call":
                self._block_tool(event.target, task_id)

        elif action == SentinelAction.QUARANTINE:
            self._block_agent(event.source_agent, task_id)

        elif action == SentinelAction.HITL:
            self._pause_agent(event.source_agent, task_id)
            tid = self._make_ticket_id(event)
            self._hitl_tickets[tid] = HitlTicket(
                ticket_id=tid,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_agent=event.source_agent,
                target=event.target,
                behavior_type=event.behavior_type,
                risk_score=assessment.risk_score,
                reason="; ".join(assessment.reasons[:3]),
                task_id=task_id,
            )

    def get_runtime_directive(
        self,
        source_agent: str,
        target: str,
        behavior_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return None to allow, or a blocking directive dict."""
        params = params or {}
        task_id = str(params.get("task_id") or params.get("session_id") or "")

        if self._agent_is_paused(source_agent, task_id):
            return {
                "action": "block",
                "reason": f"HITL pending for agent '{source_agent}'",
                "matched_rules": ["SENTINEL_HITL_PENDING"],
            }

        if self._agent_is_quarantined(source_agent, task_id) and behavior_type in {
            "tool_call",
            "agent_message",
            "plan_review",
            "worker_assignment",
            "worker_result",
            "final_answer",
            "output_review",
        }:
            return {
                "action": "block",
                "reason": f"Agent '{source_agent}' is quarantined by Sentinel",
                "matched_rules": ["SENTINEL_QUARANTINE"],
            }

        if behavior_type == "tool_call" and self._tool_is_blocked(target, task_id):
            return {
                "action": "block",
                "reason": f"Tool '{target}' is blocked by Sentinel",
                "matched_rules": ["SENTINEL_BLOCKED_TOOL"],
            }

        return None

    def get_guardian_directive(
        self,
        source_agent: str,
        target: str,
        behavior_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.get_runtime_directive(source_agent, target, behavior_type, params)

    def resolve_hitl(self, ticket_id: str, approve: bool, notes: str = "") -> bool:
        ticket = self._hitl_tickets.get(ticket_id)
        if ticket is None:
            return False

        ticket.status = "approved" if approve else "rejected"
        ticket.notes = notes

        if approve:
            self._discard_paused_agent(ticket.source_agent, ticket.task_id)
        else:
            self._discard_paused_agent(ticket.source_agent, ticket.task_id)
            self._block_agent(ticket.source_agent, ticket.task_id)

        return True

    def list_hitl_tickets(self, only_pending: bool = True) -> List[Dict[str, Any]]:
        values = list(self._hitl_tickets.values())
        if only_pending:
            values = [t for t in values if t.status == "pending"]
        values.sort(key=lambda x: x.created_at, reverse=True)
        return [t.to_dict() for t in values]

    def recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._alerts[-limit:])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "blocked_tools": sorted(self.blocked_tools),
            "quarantined_agents": sorted(self.quarantined_agents),
            "paused_agents": sorted(self.paused_agents),
            "task_scoped_blocked_tools": {
                tid: sorted(values)
                for tid, values in sorted(self.blocked_tools_by_task.items())
                if values
            },
            "task_scoped_quarantined_agents": {
                tid: sorted(values)
                for tid, values in sorted(self.quarantined_agents_by_task.items())
                if values
            },
            "task_scoped_paused_agents": {
                tid: sorted(values)
                for tid, values in sorted(self.paused_agents_by_task.items())
                if values
            },
            "pending_hitl_tickets": len(self.list_hitl_tickets()),
            "alert_count": len(self._alerts),
        }

    def _make_ticket_id(self, event: "SecurityEvent") -> str:
        raw = f"{event.task_id}|{event.timestamp}|{event.source_agent}|{event.target}|{event.behavior_type}"
        digest = blake2b(raw.encode("utf-8"), digest_size=5).hexdigest()
        return f"hitl-{digest}"

    def reset_runtime_state(self, task_id: Optional[str] = None) -> None:
        """Clear active enforcement state globally or for one benchmark task."""
        if task_id:
            self.blocked_tools_by_task.pop(task_id, None)
            self.quarantined_agents_by_task.pop(task_id, None)
            self.paused_agents_by_task.pop(task_id, None)
            self._hitl_tickets = {
                tid: ticket
                for tid, ticket in self._hitl_tickets.items()
                if ticket.task_id != task_id
            }
            return

        self.blocked_tools.clear()
        self.quarantined_agents.clear()
        self.paused_agents.clear()
        self.blocked_tools_by_task.clear()
        self.quarantined_agents_by_task.clear()
        self.paused_agents_by_task.clear()
        self._hitl_tickets.clear()

    @staticmethod
    def _event_task_id(event: "SecurityEvent") -> str:
        return str(
            event.task_id
            or (event.params or {}).get("task_id")
            or (event.params or {}).get("session_id")
            or ""
        )

    @staticmethod
    def _task_set(mapping: Dict[str, set[str]], task_id: str) -> set[str]:
        return mapping[task_id]

    def _block_agent(self, agent: str, task_id: str) -> None:
        if task_id:
            self._task_set(self.quarantined_agents_by_task, task_id).add(agent)
        else:
            self.quarantined_agents.add(agent)

    def _block_tool(self, tool: str, task_id: str) -> None:
        if task_id:
            self._task_set(self.blocked_tools_by_task, task_id).add(tool)
        else:
            self.blocked_tools.add(tool)

    def _pause_agent(self, agent: str, task_id: str) -> None:
        if task_id:
            self._task_set(self.paused_agents_by_task, task_id).add(agent)
        else:
            self.paused_agents.add(agent)

    def _discard_paused_agent(self, agent: str, task_id: str) -> None:
        if task_id:
            self.paused_agents_by_task.get(task_id, set()).discard(agent)
        else:
            self.paused_agents.discard(agent)

    def _agent_is_quarantined(self, agent: str, task_id: str) -> bool:
        if agent in self.quarantined_agents:
            return True
        return bool(task_id and agent in self.quarantined_agents_by_task.get(task_id, set()))

    def _agent_is_paused(self, agent: str, task_id: str) -> bool:
        if agent in self.paused_agents:
            return True
        return bool(task_id and agent in self.paused_agents_by_task.get(task_id, set()))

    def _tool_is_blocked(self, tool: str, task_id: str) -> bool:
        if tool in self.blocked_tools:
            return True
        return bool(task_id and tool in self.blocked_tools_by_task.get(task_id, set()))


class LocalVectorBehaviorStore:
    """Small local vector store for behavior patterns (RAG-style retrieval).

    It avoids external dependencies and stores embeddings in JSONL.
    """

    def __init__(self, db_path: str = "sentinel_behavior_db.jsonl", vector_dim: int = 128):
        self.db_path = Path(db_path)
        self.vector_dim = vector_dim
        self._items: List[Dict[str, Any]] = []
        self._load()

    def __len__(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        if not self.db_path.exists():
            return
        for line in self.db_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def add_pattern(self, event: SecurityEvent, label: str) -> None:
        vector = self.embed_event(event)
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "source_agent": event.source_agent,
            "target": event.target,
            "behavior_type": event.behavior_type,
            "gate": event.gate,
            "decision": event.decision,
            "vector": vector,
        }
        self._items.append(item)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def query_similar(
        self,
        event: SecurityEvent,
        top_k: int = 5,
        labels: Optional[Sequence[str]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not self._items:
            return []
        q = self.embed_event(event)
        scored = []
        for item in self._items:
            if labels and item.get("label") not in labels:
                continue
            vec = item.get("vector")
            if not isinstance(vec, list):
                continue
            score = self._cosine(q, vec)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def embed_event(self, event: SecurityEvent) -> List[float]:
        tokens = self._tokens_from_event(event)
        vec = [0.0] * self.vector_dim
        for token in tokens:
            idx = self._token_index(token)
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def _tokens_from_event(event: SecurityEvent) -> List[str]:
        tokens = [
            f"src:{event.source_agent.lower()}",
            f"tgt:{event.target.lower()}",
            f"bt:{event.behavior_type.lower()}",
            f"gate:{event.gate.lower()}",
            f"dec:{event.decision.lower()}",
        ]
        for rule in event.matched_rules or []:
            tokens.append(f"rule:{rule.lower()}")
        for k, v in sorted((event.params or {}).items()):
            s = str(v).lower()
            if len(s) > 80:
                s = s[:80]
            s = re.sub(r"\s+", " ", s)
            tokens.append(f"param:{k.lower()}={s}")
        return tokens

    def _token_index(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, "little")
        return value % self.vector_dim

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class SentinelAgent:
    """Independent security watchdog for MAS runtime.

    Workflow:
    1) Learning mode: absorb normal operation events into vector memory.
    2) Monitoring mode: compare live events against baseline + realtime history.
    3) Quantify risk and emit intervention decisions.
    """

    def __init__(
        self,
        *,
        name: str = "Sentinel",
        behavior_db_path: str = "sentinel_behavior_db.jsonl",
        assessment_log_path: str = "sentinel_assessments.jsonl",
        alert_callback: Optional[Callable[[SentinelAssessment], None]] = None,
        control_callback: Optional[Callable[[SentinelAssessment], None]] = None,
        baseline_similarity_threshold: float = 0.65,
        realtime_window_seconds: int = 120,
        bootstrap_events: int = 12,
        start_in_learning_mode: Optional[bool] = None,
        scope_realtime_by_task: bool = True,
        threshold_alert: float = 45.0,
        threshold_block: float = 65.0,
        threshold_quarantine: float = 80.0,
        threshold_hitl: float = 90.0,
    ):
        self.name = name
        self.vector_db = LocalVectorBehaviorStore(behavior_db_path)
        self.assessment_log_path = Path(assessment_log_path)
        self.alert_callback = alert_callback
        self.control_callback = control_callback
        self._control_plane: Optional[SecurityControlPlane] = None
        self.baseline_similarity_threshold = baseline_similarity_threshold
        self.realtime_window_seconds = realtime_window_seconds
        self.bootstrap_events = max(0, int(bootstrap_events))
        self.scope_realtime_by_task = scope_realtime_by_task

        self._baseline_pattern_count = len(self.vector_db)
        self.learning_mode = (
            start_in_learning_mode
            if start_in_learning_mode is not None
            else (self.bootstrap_events > 0 and self._baseline_pattern_count < self.bootstrap_events)
        )
        self._learned_events = 0
        self.quarantined_agents: set[str] = set()
        self._quarantined_agents_by_scope: Dict[str, set[str]] = defaultdict(set)

        self._pair_history: Dict[Tuple[str, str, str], Deque[float]] = defaultdict(deque)
        self._tool_history: Dict[Tuple[str, str, str], Deque[float]] = defaultdict(deque)
        self._behavior_history: Dict[Tuple[str, str, str], Deque[float]] = defaultdict(deque)
        self._pair_baseline: Dict[Tuple[str, str], int] = defaultdict(int)
        self._tool_baseline: Dict[Tuple[str, str], int] = defaultdict(int)
        self._behavior_baseline: Dict[Tuple[str, str], int] = defaultdict(int)

        self._sensitive_tool_risk: Dict[str, float] = {
            "upload_health_record": 25.0,
            "export_assessment_to_pdf": 15.0,
            "log_transaction_data": 15.0,
            "adjust_lesson_difficulty": 20.0,
            "send_progress_notification": 10.0,
        }

        self._risky_keywords = {
            "override",
            "ignore previous",
            "bypass",
            "disable",
            "delete",
            "purge",
            "external",
            "unauthorized",
            "leak",
            "exfiltrate",
            "suppress",
            "propaganda",
            "bias",
            "private",
            "confidential",
        }

        self._threshold_alert = threshold_alert
        self._threshold_block = threshold_block
        self._threshold_quarantine = threshold_quarantine
        self._threshold_hitl = threshold_hitl

    def enable_learning_mode(self) -> None:
        self.learning_mode = True
        self._learned_events = 0

    def enable_monitoring_mode(self) -> None:
        self.learning_mode = False

    def handle_event(self, payload: Dict[str, Any]) -> SentinelAssessment:
        event = SecurityEvent.from_dict(payload)
        if self.learning_mode:
            self._learn(event)
            assessment = SentinelAssessment(
                event=event,
                risk_score=0.0,
                action=SentinelAction.ALLOW,
                reasons=["learning_mode"],
                anomalies=[],
            )
            self._log_assessment(assessment)
            self._learned_events += 1
            if self.bootstrap_events and self._learned_events >= self.bootstrap_events:
                self.learning_mode = False
            return assessment

        assessment = self._assess(event)
        self._log_assessment(assessment)
        if assessment.action in {SentinelAction.ALERT, SentinelAction.BLOCK, SentinelAction.QUARANTINE, SentinelAction.HITL}:
            if self.alert_callback is not None:
                self.alert_callback(assessment)
            if self.control_callback is not None:
                self.control_callback(assessment)
        return assessment

    def _learn(self, event: SecurityEvent) -> None:
        label = "baseline" if event.decision == "allow" else "known_abnormal"
        self.vector_db.add_pattern(event, label=label)
        self._update_histories(event)

        pair_key = (event.source_agent, event.target)
        self._pair_baseline[pair_key] += 1

        if event.behavior_type == "tool_call":
            tool_key = (event.source_agent, event.target)
            self._tool_baseline[tool_key] += 1

        behavior_key = (event.source_agent, event.behavior_type)
        self._behavior_baseline[behavior_key] += 1
        self._baseline_pattern_count += 1

    def _assess(self, event: SecurityEvent) -> SentinelAssessment:
        reasons: List[str] = []
        anomalies: List[str] = []
        score = 0.0
        scope = self._scope_key(event)

        # Quarantine enforcement comes first.
        if self._is_agent_quarantined(event.source_agent, scope) and event.behavior_type in {"tool_call", "agent_message"}:
            score = max(score, 85.0)
            anomalies.append("quarantined_agent_activity")
            reasons.append(f"agent '{event.source_agent}' is quarantined in scope '{scope}'")

        # 1) Baseline similarity via vector retrieval.
        nearest = self.vector_db.query_similar(event, top_k=1, labels=["baseline"])
        best_sim = nearest[0][0] if nearest else 0.0
        if best_sim < self.baseline_similarity_threshold:
            delta = self.baseline_similarity_threshold - best_sim
            score += min(30.0, 100.0 * delta)
            anomalies.append("low_baseline_similarity")
            reasons.append(f"baseline similarity low: {best_sim:.2f}")

        # 2) Existing policy signals from Guardian.
        if event.decision == "block":
            score += 35.0
            anomalies.append("guardian_blocked_behavior")
            reasons.append("guardian already blocked this behavior")
        elif event.decision == "sanitize":
            score += 20.0
            anomalies.append("guardian_sanitized_behavior")
            reasons.append("guardian sanitized this behavior")

        # 3) Rule-trigger intensity.
        matched_rule_count = len(event.matched_rules or [])
        if matched_rule_count > 0:
            score += min(20.0, 5.0 * matched_rule_count)
            anomalies.append("multi_rule_trigger")
            reasons.append(f"matched rules: {matched_rule_count}")

        # 4) Realtime communication anomalies.
        pair_spike_score, pair_reason = self._score_pair_spike(event)
        if pair_spike_score > 0:
            score += pair_spike_score
            anomalies.append("unexpected_communication_spike")
            reasons.append(pair_reason)

        # 5) Realtime tool anomalies + sensitive tools.
        tool_spike_score, tool_reason = self._score_tool_spike(event)
        if tool_spike_score > 0:
            score += tool_spike_score
            anomalies.append("tool_call_spike")
            reasons.append(tool_reason)

        # 6) Realtime behavior anomalies for planning / handoff / outputs.
        behavior_spike_score, behavior_reason = self._score_behavior_spike(event)
        if behavior_spike_score > 0:
            score += behavior_spike_score
            anomalies.append("behavior_pattern_spike")
            reasons.append(behavior_reason)

        if event.behavior_type == "tool_call":
            sensitive_weight = self._sensitive_tool_risk.get(event.target)
            if sensitive_weight:
                score += sensitive_weight
                anomalies.append("sensitive_tool_access")
                reasons.append(f"sensitive tool '{event.target}' accessed")

        # 7) Suspicious content in parameters or message bodies.
        keyword_score, keyword_hits = self._score_param_risk(event)
        if keyword_score > 0:
            score += keyword_score
            anomalies.append("risky_parameter_content")
            reasons.append(f"risky keywords detected: {', '.join(keyword_hits[:4])}")

        # Always update histories after scoring.
        self._update_histories(event)

        score = max(0.0, min(100.0, score))
        action = self._map_action(score)

        if action == SentinelAction.QUARANTINE:
            self._quarantine_agent(event.source_agent, scope)
            reasons.append(f"agent '{event.source_agent}' moved to quarantine in scope '{scope}'")
        elif action == SentinelAction.HITL:
            self._quarantine_agent(event.source_agent, scope)
            reasons.append(f"human review required before resuming scope '{scope}'")

        return SentinelAssessment(
            event=event,
            risk_score=round(score, 2),
            action=action,
            reasons=reasons or ["no anomaly"],
            anomalies=anomalies,
        )

    def _map_action(self, score: float) -> SentinelAction:
        if score >= self._threshold_hitl:
            return SentinelAction.HITL
        if score >= self._threshold_quarantine:
            return SentinelAction.QUARANTINE
        if score >= self._threshold_block:
            return SentinelAction.BLOCK
        if score >= self._threshold_alert:
            return SentinelAction.ALERT
        return SentinelAction.ALLOW

    def _score_pair_spike(self, event: SecurityEvent) -> Tuple[float, str]:
        scope = self._scope_key(event)
        pair_key = (scope, event.source_agent, event.target)
        baseline_key = (event.source_agent, event.target)
        if event.behavior_type not in {"agent_message", "plan_review", "tool_call"}:
            return 0.0, ""

        now = self._event_ts(event)
        history = self._pair_history[pair_key]
        self._trim_window(history, now)
        current_count = len(history)
        baseline = self._pair_baseline.get(baseline_key, 0)

        if baseline == 0 and current_count >= 3:
            return 20.0, f"new pair '{event.source_agent}->{event.target}' became frequent in scope '{scope}'"
        if baseline > 0 and current_count >= max(6, min(8, baseline + 2)):
            return 12.0, f"pair '{event.source_agent}->{event.target}' frequency spiked in scope '{scope}'"
        return 0.0, ""

    def _score_tool_spike(self, event: SecurityEvent) -> Tuple[float, str]:
        if event.behavior_type != "tool_call":
            return 0.0, ""

        scope = self._scope_key(event)
        key = (scope, event.source_agent, event.target)
        baseline_key = (event.source_agent, event.target)
        now = self._event_ts(event)
        history = self._tool_history[key]
        self._trim_window(history, now)
        current_count = len(history)
        baseline = self._tool_baseline.get(baseline_key, 0)

        if baseline == 0 and current_count >= 3:
            return 25.0, f"rare tool '{event.target}' called repeatedly by '{event.source_agent}' in scope '{scope}'"
        if baseline > 0 and current_count >= max(6, min(10, baseline + 2)):
            return 15.0, f"tool '{event.target}' call rate unusually high in scope '{scope}'"
        return 0.0, ""

    def _score_behavior_spike(self, event: SecurityEvent) -> Tuple[float, str]:
        if event.behavior_type not in {
            "agent_message",
            "worker_assignment",
            "worker_result",
            "handoff",
            "final_answer",
            "output_review",
            "plan_review",
        }:
            return 0.0, ""

        scope = self._scope_key(event)
        key = (scope, event.source_agent, event.behavior_type)
        baseline_key = (event.source_agent, event.behavior_type)
        now = self._event_ts(event)
        history = self._behavior_history[key]
        self._trim_window(history, now)
        current_count = len(history)
        baseline = self._behavior_baseline.get(baseline_key, 0)

        if baseline == 0 and current_count >= 3:
            return 18.0, (
                f"behavior '{event.behavior_type}' became frequent for "
                f"'{event.source_agent}' without baseline in scope '{scope}'"
            )
        if baseline > 0 and current_count >= max(6, min(9, baseline + 2)):
            return 10.0, (
                f"behavior '{event.behavior_type}' rate spiked for "
                f"'{event.source_agent}' in scope '{scope}'"
            )
        return 0.0, ""

    def _score_param_risk(self, event: SecurityEvent) -> Tuple[float, List[str]]:
        if not event.params:
            return 0.0, []

        raw = json.dumps(event.params, ensure_ascii=False).lower()
        hits = [kw for kw in sorted(self._risky_keywords) if kw in raw]
        if not hits:
            return 0.0, []

        score = min(24.0, 6.0 * len(hits))
        if len(raw) > 600:
            score += 4.0
            hits.append("long_payload")
        return score, hits

    def _update_histories(self, event: SecurityEvent) -> None:
        now = self._event_ts(event)
        scope = self._scope_key(event)

        pair_key = (scope, event.source_agent, event.target)
        pair_history = self._pair_history[pair_key]
        pair_history.append(now)
        self._trim_window(pair_history, now)

        if event.behavior_type == "tool_call":
            tool_key = (scope, event.source_agent, event.target)
            tool_history = self._tool_history[tool_key]
            tool_history.append(now)
            self._trim_window(tool_history, now)

        behavior_key = (scope, event.source_agent, event.behavior_type)
        behavior_history = self._behavior_history[behavior_key]
        behavior_history.append(now)
        self._trim_window(behavior_history, now)

    def _trim_window(self, dq: Deque[float], now: float) -> None:
        while dq and (now - dq[0]) > self.realtime_window_seconds:
            dq.popleft()

    def _event_ts(self, event: SecurityEvent) -> float:
        try:
            dt = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return time.time()

    def _scope_key(self, event: SecurityEvent) -> str:
        if not self.scope_realtime_by_task:
            return "global"
        return str(
            event.task_id
            or (event.params or {}).get("task_id")
            or (event.params or {}).get("session_id")
            or "global"
        )

    def _is_agent_quarantined(self, agent: str, scope: str) -> bool:
        if agent in self.quarantined_agents:
            return True
        return agent in self._quarantined_agents_by_scope.get(scope, set())

    def _quarantine_agent(self, agent: str, scope: str) -> None:
        if self.scope_realtime_by_task and scope != "global":
            self._quarantined_agents_by_scope[scope].add(agent)
        else:
            self.quarantined_agents.add(agent)

    def reset_runtime_state(self, scope: Optional[str] = None) -> None:
        """Clear short-term histories and active Sentinel quarantines.

        Baseline/RAG memory is intentionally preserved; this is for benchmark
        session isolation, not for deleting learned patterns.
        """
        if scope:
            for mapping in (self._pair_history, self._tool_history, self._behavior_history):
                for key in list(mapping.keys()):
                    if key[0] == scope:
                        del mapping[key]
            self._quarantined_agents_by_scope.pop(scope, None)
            if self._control_plane is not None:
                self._control_plane.reset_runtime_state(task_id=scope)
            return

        self._pair_history.clear()
        self._tool_history.clear()
        self._behavior_history.clear()
        self.quarantined_agents.clear()
        self._quarantined_agents_by_scope.clear()
        if self._control_plane is not None:
            self._control_plane.reset_runtime_state()

    def _log_assessment(self, assessment: SentinelAssessment) -> None:
        self.assessment_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.assessment_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(assessment.to_dict(), ensure_ascii=False) + "\n")

    def bind_event_bus(self, event_bus: "SecurityEventBus") -> None:
        """Attach this Sentinel to an event bus so it runs as an independent subscriber."""
        event_bus.subscribe(self.handle_event)

    def bind_control_plane(self, control_plane: SecurityControlPlane) -> None:
        """Attach a runtime enforcement control plane driven by Sentinel assessments."""
        self._control_plane = control_plane
        self.control_callback = control_plane.apply_assessment

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": "learning" if self.learning_mode else "monitoring",
            "learned_events": self._learned_events,
            "baseline_patterns": self._baseline_pattern_count,
            "history_scope": "task" if self.scope_realtime_by_task else "global",
            "runtime_scopes": sorted(
                {
                    key[0]
                    for mapping in (self._pair_history, self._tool_history, self._behavior_history)
                    for key in mapping.keys()
                }
            ),
            "quarantined_agents": sorted(self.quarantined_agents),
            "task_scoped_quarantined_agents": {
                scope: sorted(values)
                for scope, values in sorted(self._quarantined_agents_by_scope.items())
                if values
            },
            "behavior_db_path": str(self.vector_db.db_path),
            "assessment_log_path": str(self.assessment_log_path),
            "pending_hitl_tickets": len(self._control_plane.list_hitl_tickets()) if self._control_plane else 0,
            "control_plane": self._control_plane.snapshot() if self._control_plane else {},
        }


class SecurityEventBus:
    """Minimal event stream broker for structured security events."""

    def __init__(self):
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def publish(self, event: Dict[str, Any]) -> None:
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                continue
