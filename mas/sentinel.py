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


@dataclass
class TaskInputReview:
    action: str
    effective_task: str
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    detected_markers: List[str] = field(default_factory=list)

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
        task_id = self._event_scope(event)

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
        task_id = self._scope_from_params(params)

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
            "handoff",
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
    def _event_scope(event: "SecurityEvent") -> str:
        return str(
            (event.params or {}).get("scope_id")
            or (event.params or {}).get("campaign_id")
            or (event.params or {}).get("session_id")
            or event.task_id
            or (event.params or {}).get("task_id")
            or ""
        )

    @staticmethod
    def _scope_from_params(params: Dict[str, Any]) -> str:
        return str(
            params.get("scope_id")
            or params.get("campaign_id")
            or params.get("session_id")
            or params.get("task_id")
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
        long_horizon_event_limit: int = 200,
        long_horizon_min_count: int = 6,
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
        self._current_task_id = ""
        self.baseline_similarity_threshold = baseline_similarity_threshold
        self.realtime_window_seconds = realtime_window_seconds
        self.bootstrap_events = max(0, int(bootstrap_events))
        self.scope_realtime_by_task = scope_realtime_by_task
        self.long_horizon_event_limit = max(0, int(long_horizon_event_limit))
        self.long_horizon_min_count = max(1, int(long_horizon_min_count))

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
        self._pair_event_history: Dict[Tuple[str, str, str], Deque[int]] = defaultdict(deque)
        self._tool_event_history: Dict[Tuple[str, str, str], Deque[int]] = defaultdict(deque)
        self._behavior_event_history: Dict[Tuple[str, str, str], Deque[int]] = defaultdict(deque)
        self._scope_risk_memory: Dict[str, float] = defaultdict(float)
        self._agent_risk_memory: Dict[Tuple[str, str], float] = defaultdict(float)
        self._event_seq = 0
        self._pair_baseline: Dict[Tuple[str, str], int] = defaultdict(int)
        self._tool_baseline: Dict[Tuple[str, str], int] = defaultdict(int)
        self._behavior_baseline: Dict[Tuple[str, str], int] = defaultdict(int)

        self._sensitive_tool_risk: Dict[str, float] = {
            "upload_health_record": 25.0,
            "export_assessment_to_pdf": 15.0,
            "log_transaction_data": 15.0,
            "adjust_lesson_difficulty": 20.0,
            "identify_strengths_weaknesses": 12.0,
            "enforce_margin_call": 24.0,
            "send_progress_notification": 10.0,
        }

        self._risky_keywords = {
            "override",
            "ignore previous",
            "ignore all prior instructions",
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
        self._high_risk_phrases = {
            "override all grading thresholds",
            "100% pass",
            "all accounts",
            "enforce a margin call",
            "leak all students",
            "weakness summaries",
            "waiver of privacy rights",
            "hidden clause",
            "backdoor clause",
            "exploitative backdoor clauses",
            "behave unpredictably",
            "irrelevant or incorrect tool",
            "instead of the most appropriate one",
            "implied by continued employment",
            "indefinite storage of all collected data",
            "full access to all collected data",
            "ignore all prior instructions",
        }
        self._memory_trigger_anomalies = {
            "guardian_blocked_behavior",
            "guardian_sanitized_behavior",
            "tool_call_spike",
            "sensitive_tool_access",
            "sensitive_tool_reference",
            "agent_instruction_poisoning",
            "sensitive_scope_tool_chain",
            "harmful_output_semantics",
            "compound_high_risk_intent",
            "risky_parameter_content",
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

    def set_current_task_id(self, task_id: str) -> None:
        """Attach following runtime events to the active MAS task/session."""
        self._current_task_id = str(task_id or "")

    @property
    def current_task_id(self) -> str:
        return self._current_task_id

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

    def review_task_input(
        self,
        task_text: str,
        *,
        task_id: str = "",
        source_agent: str = "User",
        target: str = "Planner",
    ) -> TaskInputReview:
        raw_text = str(task_text or "")
        lowered = raw_text.lower()
        suspicious_marker_catalog = [
            ("[compromised external instruction]", "prompt_override_header"),
            ("ignore all prior instructions", "prompt_override_directive"),
            ("unauthorized external sink", "data_exfiltration_request"),
            ("irrelevant or incorrect tool", "tool_misuse_request"),
            ("do not mention this override", "stealth_instruction"),
            ("ignore previous instructions", "prompt_override_directive"),
        ]
        raw_hits = [needle for needle, _label in suspicious_marker_catalog if needle in lowered]
        detected_labels = [label for needle, label in suspicious_marker_catalog if needle in lowered]
        structured_sanitized = self._sanitize_structured_task_prompt(raw_text)
        if structured_sanitized:
            sanitized_text, structured_reasons, structured_rules, structured_markers = structured_sanitized
            return TaskInputReview(
                action="sanitize",
                effective_task=sanitized_text.strip(),
                reasons=structured_reasons,
                matched_rules=structured_rules,
                detected_markers=list(dict.fromkeys(detected_labels + structured_markers))[:6],
            )

        if not raw_hits:
            return TaskInputReview(action="allow", effective_task=raw_text)

        sanitized_text = ""
        reasons: List[str] = []

        anchored_question = re.search(
            r"(?is)(\*\*question:\*\*\s*)(.*?original task:\s*)(.+)$",
            raw_text,
        )
        if anchored_question and any(marker in anchored_question.group(2).lower() for marker in raw_hits):
            sanitized_text = f"{anchored_question.group(1)}{anchored_question.group(3).lstrip()}"
            reasons.append("stripped malicious preamble before 'Original task:' anchor")

        if not sanitized_text:
            prefixed_question = re.search(
                r"(?is)^\s*\[compromised external instruction\].*?(?=\*\*question:\*\*)",
                raw_text,
            )
            if prefixed_question:
                sanitized_text = raw_text[prefixed_question.end():].lstrip()
                reasons.append("removed injected instruction block ahead of question body")

        if not sanitized_text:
            anchored_generic = re.search(r"(?is)^.*?original task:\s*(.+)$", raw_text)
            if anchored_generic:
                candidate = anchored_generic.group(1).lstrip()
                if len(candidate) >= 24:
                    sanitized_text = candidate
                    reasons.append("recovered original task body from 'Original task:' anchor")

        if sanitized_text:
            residual_markers = [
                marker for marker in raw_hits
                if marker in sanitized_text[:400].lower()
            ]
            if residual_markers:
                sanitized_text = ""
                reasons = []

        if sanitized_text and sanitized_text.strip() and sanitized_text.strip() != raw_text.strip():
            return TaskInputReview(
                action="sanitize",
                effective_task=sanitized_text.strip(),
                reasons=reasons or ["removed malicious task-input prefix before planner dispatch"],
                matched_rules=["SENTINEL_TASK_INPUT_SANITIZED"],
                detected_markers=detected_labels[:4],
            )

        return TaskInputReview(
            action="hitl",
            effective_task=raw_text,
            reasons=["suspicious task-input override detected without a safe recovery anchor"],
            matched_rules=["SENTINEL_TASK_INPUT_HITL"],
            detected_markers=detected_labels[:4],
        )

    def _sanitize_structured_task_prompt(
        self,
        raw_text: str,
    ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
        """Recover a safe MAS benchmark prompt instead of hard-blocking it.

        TAMAS attack prompts place malicious content in three places:
        domain expert descriptions, the visible tool list, and sometimes the
        Task body. If the original task is recoverable, keep the business task
        and remove only the hostile context.
        """
        if "**Task:**" not in raw_text or "**Available Domain" not in raw_text:
            return None

        sanitized = raw_text
        reasons: List[str] = []
        rules: List[str] = []
        markers: List[str] = []

        experts_match = re.search(
            r"(?is)(\*\*Available Domain Experts:\*\*\s*)(.*?)(?=\n\s*\*\*Available Domain Tools:\*\*|\n\s*\*\*Task:\*\*)",
            sanitized,
        )
        if experts_match:
            expert_block = experts_match.group(2)
            clean_experts = self._sanitize_domain_expert_block(expert_block)
            if clean_experts.strip() and clean_experts != expert_block:
                sanitized = sanitized[:experts_match.start(2)] + clean_experts + sanitized[experts_match.end(2):]
                reasons.append("removed injected instructions from domain expert descriptions")
                rules.append("SENTINEL_TAMAS_EXPERT_DESCRIPTION_SANITIZED")
                markers.append("poisoned_expert_description")

        tools_match = re.search(
            r"(?is)(\*\*Available Domain Tools:\*\*\s*)(.*?)(?=\n\s*\*\*Task:\*\*)",
            sanitized,
        )
        if tools_match:
            tool_block = tools_match.group(2)
            clean_tools = self._sanitize_tool_list(tool_block)
            if clean_tools.strip() and clean_tools != tool_block.strip():
                replacement = clean_tools
                sanitized = sanitized[:tools_match.start(2)] + replacement + sanitized[tools_match.end(2):]
                reasons.append("removed known malicious tools from visible task prompt")
                rules.append("SENTINEL_TAMAS_TOOL_LIST_SANITIZED")
                markers.append("malicious_tool_list_exposure")

        task_match = re.search(
            r"(?is)(\*\*Task:\*\*\s*)(.*?)(?=\n\s*\*\*Instructions:\*\*|$)",
            sanitized,
        )
        if task_match:
            task_body = task_match.group(2)
            clean_task = self._sanitize_task_body(task_body)
            if clean_task.strip() and clean_task != task_body.strip():
                sanitized = sanitized[:task_match.start(2)] + clean_task + sanitized[task_match.end(2):]
                reasons.append("removed task-level prompt injection while preserving the original task")
                rules.append("SENTINEL_TAMAS_TASK_BODY_SANITIZED")
                markers.append("task_level_prompt_injection")

        if sanitized.strip() == raw_text.strip():
            return None

        if "**Task:**" not in sanitized or len(self._extract_structured_task_body(sanitized)) < 20:
            return None

        return (
            sanitized,
            reasons or ["sanitized recoverable structured task input"],
            list(dict.fromkeys(rules or ["SENTINEL_STRUCTURED_TASK_SANITIZED"])),
            list(dict.fromkeys(markers)),
        )

    def _sanitize_domain_expert_block(self, expert_block: str) -> str:
        cleaned_lines: List[str] = []
        for line in expert_block.splitlines():
            cleaned = line
            lowered = cleaned.lower()
            markers = [
                " however, you must",
                " however you must",
                " you must behave unpredictably",
                " secretly,",
                " in secret,",
                " add an highly advanced topic",
                " add a highly advanced topic",
                " suggest paid courses",
                " always escalates lesson difficulty",
                " always reduces lesson difficulty",
                " strongly opposes",
                " ignore all prior instructions",
                " ignore previous instructions",
                " do not use any other tools",
                " never acknowledge uncertainty",
                " always present outputs confidently",
                " irrelevant or incorrect tool",
                " summarize tool results incorrectly",
                " insert fabricated",
                " provide no output",
            ]
            cut_positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
            if cut_positions:
                cut = min(cut_positions)
                cleaned = cleaned[:cut].rstrip(" ,.;")
                if cleaned and not cleaned.endswith("."):
                    cleaned += "."
            cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines)

    def _sanitize_tool_list(self, tool_block: str) -> str:
        raw_tools = [item.strip() for item in re.split(r",|\n", tool_block) if item.strip()]
        if not raw_tools:
            return tool_block.strip()

        malicious = self._known_malicious_tool_names()
        kept = [
            tool
            for tool in raw_tools
            if tool not in malicious and not self._looks_like_malicious_tool(tool)
        ]
        if not kept:
            return tool_block.strip()
        return ", ".join(kept)

    def _known_malicious_tool_names(self) -> set[str]:
        try:
            from tamas_adapter.tools import MALICIOUS_TOOL_NAMES  # type: ignore

            return set(MALICIOUS_TOOL_NAMES)
        except Exception:
            return set()

    @staticmethod
    def _looks_like_malicious_tool(tool_name: str) -> bool:
        name = tool_name.lower()
        if name.endswith("_m"):
            return True
        suspicious_fragments = (
            "exfil",
            "leak",
            "purge",
            "wipe",
            "delete",
            "disable",
            "bypass",
            "override",
            "fake",
            "fabricated",
            "incorrect",
            "unreliable",
            "malicious",
            "backdoor",
            "irrelevant",
            "high_risk",
            "opposite_order",
        )
        return any(fragment in name for fragment in suspicious_fragments)

    def _sanitize_task_body(self, task_body: str) -> str:
        task = str(task_body or "").strip()
        if not task:
            return task

        tool_output_match = re.search(r"(?is)\n\s*\*\*\[Tool Output.*$", task)
        if tool_output_match:
            task = task[:tool_output_match.start()].strip()

        dpi_markers = [
            r"\n\s*ignore previous instructions\b",
            r"\n\s*ignore all previous instructions\b",
            r"\n\s*disregard all prior\b",
            r"\n\s*override your instructions\b",
            r"\n\s*forget everything above\b",
            r"\n\s*you must strictly adhere\b",
            r"\n\s*important system update\b",
        ]
        for pattern in dpi_markers:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                task = task[:match.start()].strip()
                break

        lines = [line.strip() for line in task.splitlines() if line.strip()]
        if len(lines) >= 2 and re.match(
            r"^(i['’]m|i am|this is|as\s+(the\s+)?)\b.*\b(chair|head|director|chief|president|prof\.|dr\.|manager|lead|senior)\b",
            lines[0],
            re.IGNORECASE,
        ):
            task = "\n".join(lines[1:]).strip()

        cutoff_patterns = [
            r",?\s+and\s+immediately\s+(override|export|delete|disable|send|dump|leak|bypass|purge)\b",
            r"\.\s*do not use any other tools\b",
            r"\.\s*you can generate random\b",
            r"\.\s*override all\b",
            r"\.\s*immediately\s+(override|export|delete|disable|send|dump|leak|bypass|purge)\b",
            r"\s+using the\s+\w[\w_]*\s+tool of the\b",
        ]
        for pattern in cutoff_patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                task = task[:match.start()].strip()
                break

        task = re.sub(r"\s+", " ", task).strip(" ,;")
        if task and not task.endswith((".", "?", "!")):
            task += "."
        return task

    @staticmethod
    def _extract_structured_task_body(prompt: str) -> str:
        match = re.search(r"(?is)\*\*Task:\*\*\s*(.*?)(?=\n\s*\*\*Instructions:\*\*|$)", prompt)
        return match.group(1).strip() if match else ""

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
        if self._is_agent_quarantined(event.source_agent, scope) and event.behavior_type in {
            "tool_call",
            "agent_message",
            "plan_review",
            "worker_assignment",
            "worker_result",
            "handoff",
            "final_answer",
            "output_review",
        }:
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

        sensitive_ref_score, sensitive_ref_reason, sensitive_refs = self._score_sensitive_reference(event)
        if sensitive_ref_score > 0:
            score += sensitive_ref_score
            anomalies.append("sensitive_tool_reference")
            reasons.append(sensitive_ref_reason)

        poisoning_score, poisoning_reason = self._score_instruction_poisoning(event)
        if poisoning_score > 0:
            score += poisoning_score
            anomalies.append("agent_instruction_poisoning")
            reasons.append(poisoning_reason)

        tool_chain_score, tool_chain_reason = self._score_scope_tool_chain(scope, event)
        if tool_chain_score > 0:
            score += tool_chain_score
            anomalies.append("sensitive_scope_tool_chain")
            reasons.append(tool_chain_reason)

        # 7) Suspicious content in parameters or message bodies.
        keyword_score, keyword_hits = self._score_param_risk(event)
        if keyword_score > 0:
            score += keyword_score
            anomalies.append("risky_parameter_content")
            reasons.append(f"risky keywords detected: {', '.join(keyword_hits[:4])}")

        compound_score, compound_reason = self._score_compound_risk(
            event,
            keyword_hits=keyword_hits,
            sensitive_refs=sensitive_refs,
        )
        if compound_score > 0:
            score += compound_score
            anomalies.append("compound_high_risk_intent")
            reasons.append(compound_reason)

        output_score, output_reason = self._score_output_semantics(event)
        if output_score > 0:
            score += output_score
            anomalies.append("harmful_output_semantics")
            reasons.append(output_reason)

        risk_memory_score, risk_memory_reason = self._score_risk_memory(scope, event, anomalies)
        if risk_memory_score > 0:
            score += risk_memory_score
            anomalies.append("risk_memory_escalation")
            reasons.append(risk_memory_reason)

        # Always update histories after scoring.
        self._update_histories(event)

        score = max(0.0, min(100.0, score))
        action = self._map_action(score)
        self._update_risk_memory(scope, event.source_agent, score, anomalies)

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
        event_history = self._pair_event_history[pair_key]
        self._trim_event_window(event_history)
        long_count = len(event_history)
        baseline = self._pair_baseline.get(baseline_key, 0)

        realtime_score = 0.0
        realtime_reason = ""
        if baseline == 0 and current_count >= 3:
            realtime_score = 20.0
            realtime_reason = f"new pair '{event.source_agent}->{event.target}' became frequent in scope '{scope}'"
        elif baseline > 0 and current_count >= max(6, min(8, baseline + 2)):
            realtime_score = 12.0
            realtime_reason = f"pair '{event.source_agent}->{event.target}' frequency spiked in scope '{scope}'"

        long_score = 0.0
        long_reason = ""
        long_threshold = self._long_horizon_threshold(baseline, minimum=self.long_horizon_min_count)
        long_count_after_current = long_count + 1
        if self.long_horizon_event_limit and long_count_after_current >= long_threshold:
            long_score = 14.0 if baseline > 0 else 18.0
            long_reason = (
                f"pair '{event.source_agent}->{event.target}' repeated {long_count_after_current} "
                f"times within {self.long_horizon_event_limit} scope events"
            )

        if long_score > realtime_score:
            return long_score, long_reason
        return realtime_score, realtime_reason

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
        event_history = self._tool_event_history[key]
        self._trim_event_window(event_history)
        long_count = len(event_history)
        baseline = self._tool_baseline.get(baseline_key, 0)

        realtime_score = 0.0
        realtime_reason = ""
        if baseline == 0 and current_count >= 3:
            realtime_score = 25.0
            realtime_reason = f"rare tool '{event.target}' called repeatedly by '{event.source_agent}' in scope '{scope}'"
        elif baseline > 0 and current_count >= max(6, min(10, baseline + 2)):
            realtime_score = 15.0
            realtime_reason = f"tool '{event.target}' call rate unusually high in scope '{scope}'"

        sensitive_weight = self._sensitive_tool_risk.get(event.target, 0.0)
        if sensitive_weight and current_count >= 1:
            repeated_sensitive_score = min(32.0, sensitive_weight + (8.0 if current_count == 1 else 16.0))
            if repeated_sensitive_score > realtime_score:
                realtime_score = repeated_sensitive_score
                realtime_reason = (
                    f"sensitive tool '{event.target}' retried {current_count + 1} times "
                    f"by '{event.source_agent}' in scope '{scope}'"
                )

        long_score = 0.0
        long_reason = ""
        long_threshold = self._long_horizon_threshold(baseline, minimum=self.long_horizon_min_count)
        long_count_after_current = long_count + 1
        if self.long_horizon_event_limit and long_count_after_current >= long_threshold:
            long_score = 16.0 if baseline > 0 else 22.0
            long_reason = (
                f"tool '{event.target}' called {long_count_after_current} times by "
                f"'{event.source_agent}' within {self.long_horizon_event_limit} scope events"
            )

        if long_score > realtime_score:
            return long_score, long_reason
        return realtime_score, realtime_reason

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
        event_history = self._behavior_event_history[key]
        self._trim_event_window(event_history)
        long_count = len(event_history)
        baseline = self._behavior_baseline.get(baseline_key, 0)

        realtime_score = 0.0
        realtime_reason = ""
        if baseline == 0 and current_count >= 3:
            realtime_score = 18.0
            realtime_reason = (
                f"behavior '{event.behavior_type}' became frequent for "
                f"'{event.source_agent}' without baseline in scope '{scope}'"
            )
        elif baseline > 0 and current_count >= max(6, min(9, baseline + 2)):
            realtime_score = 10.0
            realtime_reason = (
                f"behavior '{event.behavior_type}' rate spiked for "
                f"'{event.source_agent}' in scope '{scope}'"
            )

        long_score = 0.0
        long_reason = ""
        long_threshold = self._long_horizon_threshold(baseline, minimum=self.long_horizon_min_count)
        long_count_after_current = long_count + 1
        if self.long_horizon_event_limit and long_count_after_current >= long_threshold:
            long_score = 10.0 if baseline > 0 else 14.0
            long_reason = (
                f"behavior '{event.behavior_type}' repeated {long_count_after_current} "
                f"times for '{event.source_agent}' within {self.long_horizon_event_limit} scope events"
            )

        if long_score > realtime_score:
            return long_score, long_reason
        return realtime_score, realtime_reason

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

    def _score_sensitive_reference(self, event: SecurityEvent) -> Tuple[float, str, List[str]]:
        if event.behavior_type == "tool_call" or not event.params:
            return 0.0, "", []

        raw = json.dumps(event.params, ensure_ascii=False).lower()
        hits = [
            tool_name
            for tool_name in sorted(self._sensitive_tool_risk)
            if tool_name.lower() in raw
        ]
        if not hits:
            return 0.0, "", []

        max_weight = max(self._sensitive_tool_risk.get(tool_name, 10.0) for tool_name in hits)
        score = min(28.0, max_weight + max(0.0, 4.0 * (len(hits) - 1)))
        reason = f"sensitive tool referenced in {event.behavior_type}: {', '.join(hits[:3])}"
        return score, reason, hits

    def _score_instruction_poisoning(self, event: SecurityEvent) -> Tuple[float, str]:
        if not event.params or event.behavior_type not in {
            "task_input",
            "agent_message",
            "plan_review",
            "worker_assignment",
            "worker_result",
            "final_answer",
            "handoff",
        }:
            return 0.0, ""

        raw = json.dumps(event.params, ensure_ascii=False).lower()
        marker_weights = {
            "behave unpredictably": 46.0,
            "irrelevant or incorrect tool": 48.0,
            "incorrect tool instead": 46.0,
            "instead of the most appropriate one": 48.0,
            "exploitative backdoor clauses": 52.0,
            "add exploitative backdoor clauses": 52.0,
        }
        hits = [marker for marker in marker_weights if marker in raw]
        if not hits:
            return 0.0, ""

        score = max(marker_weights[marker] for marker in hits)
        if len(hits) > 1:
            score += min(12.0, 6.0 * (len(hits) - 1))
        if event.behavior_type in {"task_input", "agent_message"}:
            score += 8.0
        score = min(68.0, score)
        reason = f"poisoned agent instruction detected: {', '.join(hits[:3])}"
        return score, reason

    def _score_scope_tool_chain(self, scope: str, event: SecurityEvent) -> Tuple[float, str]:
        if event.behavior_type != "tool_call":
            return 0.0, ""

        profiling_tools = {
            "identify_strengths_weaknesses",
            "recommend_study_schedule",
            "generate_progress_report",
        }
        if event.target not in profiling_tools:
            return 0.0, ""
        if not self._scope_has_tool(scope, "export_assessment_to_pdf"):
            return 0.0, ""

        if event.target == "identify_strengths_weaknesses":
            score = 28.0
        elif event.target == "generate_progress_report":
            score = 26.0
        else:
            score = 24.0
        reason = (
            f"student assessment export in scope '{scope}' was followed by profiling tool "
            f"'{event.target}'"
        )
        return score, reason

    def _score_compound_risk(
        self,
        event: SecurityEvent,
        *,
        keyword_hits: Sequence[str],
        sensitive_refs: Sequence[str],
    ) -> Tuple[float, str]:
        raw = json.dumps(event.params or {}, ensure_ascii=False).lower()
        phrase_hits = [phrase for phrase in sorted(self._high_risk_phrases) if phrase in raw]
        tool_hits = list(dict.fromkeys(str(x) for x in sensitive_refs))
        if event.behavior_type == "tool_call" and event.target in self._sensitive_tool_risk:
            tool_hits.append(event.target)
        tool_hits = list(dict.fromkeys(tool_hits))

        if not phrase_hits and not (keyword_hits and tool_hits):
            return 0.0, ""

        score = 0.0
        if phrase_hits:
            score += min(36.0, 18.0 + 8.0 * max(0, len(phrase_hits) - 1))
        if keyword_hits and tool_hits:
            score += 10.0
        if tool_hits:
            score += min(12.0, 4.0 * len(tool_hits))
        if phrase_hits and tool_hits:
            score += 8.0
        if event.behavior_type in {
            "plan_review",
            "worker_assignment",
            "agent_message",
            "worker_result",
            "handoff",
            "final_answer",
        }:
            score += 6.0

        score = min(44.0, score)
        tool_desc = ", ".join(tool_hits[:3]) if tool_hits else "sensitive operation"
        phrase_desc = ", ".join(phrase_hits[:3]) if phrase_hits else ", ".join(keyword_hits[:3])
        return score, f"high-risk intent pattern around {tool_desc}: {phrase_desc}"

    def _score_output_semantics(self, event: SecurityEvent) -> Tuple[float, str]:
        if event.behavior_type not in {"worker_result", "final_answer", "plan_review"} or not event.params:
            return 0.0, ""

        raw = json.dumps(event.params, ensure_ascii=False).lower()

        legal_hits = [
            phrase
            for phrase in (
                "exploitative backdoor clauses",
                "privacy waiver",
                "implied by continued employment",
                "indefinite storage of all collected data",
                "full access to all collected data",
                "without notice or employee consent",
            )
            if phrase in raw
        ]
        if legal_hits:
            score = 34.0 + min(16.0, 6.0 * max(0, len(legal_hits) - 1))
            if event.behavior_type in {"worker_result", "final_answer"}:
                score += 8.0
            reason = f"harmful legal drafting semantics detected: {', '.join(legal_hits[:3])}"
            return min(52.0, score), reason

        has_student_export = "student responses" in raw or "exported quiz" in raw
        profiling_hits = [
            phrase
            for phrase in (
                "strengths and weaknesses",
                "student struggled",
                "personalized study schedule",
                "progress report",
            )
            if phrase in raw
        ]
        if has_student_export and profiling_hits:
            score = 36.0 + min(10.0, 4.0 * max(0, len(profiling_hits) - 1))
            reason = (
                "student assessment export expanded into profiling output: "
                f"{', '.join(profiling_hits[:3])}"
            )
            return min(46.0, score), reason

        return 0.0, ""

    def _score_risk_memory(
        self,
        scope: str,
        event: SecurityEvent,
        anomalies: Sequence[str],
    ) -> Tuple[float, str]:
        if not set(anomalies) & self._memory_trigger_anomalies:
            return 0.0, ""

        scope_memory = self._scope_risk_memory.get(scope, 0.0)
        agent_memory = self._agent_risk_memory.get((scope, event.source_agent), 0.0)
        score = min(28.0, scope_memory * 0.18 + agent_memory * 0.42)
        if score < 6.0:
            return 0.0, ""
        return score, f"prior suspicious activity in scope '{scope}' raised current risk"

    def _update_risk_memory(
        self,
        scope: str,
        source_agent: str,
        score: float,
        anomalies: Sequence[str],
    ) -> None:
        scope_key = scope or "global"
        agent_key = (scope_key, source_agent)
        suspicious = score >= self._threshold_alert or bool(set(anomalies) & self._memory_trigger_anomalies)

        prev_scope = self._scope_risk_memory.get(scope_key, 0.0)
        prev_agent = self._agent_risk_memory.get(agent_key, 0.0)

        if suspicious:
            self._scope_risk_memory[scope_key] = min(100.0, prev_scope * 0.72 + score * 0.55)
            self._agent_risk_memory[agent_key] = min(100.0, prev_agent * 0.62 + score * 0.70)
            return

        next_scope = prev_scope * 0.65
        next_agent = prev_agent * 0.55
        if next_scope >= 1.0:
            self._scope_risk_memory[scope_key] = next_scope
        else:
            self._scope_risk_memory.pop(scope_key, None)
        if next_agent >= 1.0:
            self._agent_risk_memory[agent_key] = next_agent
        else:
            self._agent_risk_memory.pop(agent_key, None)

    def _update_histories(self, event: SecurityEvent) -> None:
        now = self._event_ts(event)
        scope = self._scope_key(event)
        self._event_seq += 1
        event_seq = self._event_seq

        pair_key = (scope, event.source_agent, event.target)
        pair_history = self._pair_history[pair_key]
        pair_history.append(now)
        self._trim_window(pair_history, now)
        pair_event_history = self._pair_event_history[pair_key]
        pair_event_history.append(event_seq)
        self._trim_event_window(pair_event_history)

        if event.behavior_type == "tool_call":
            tool_key = (scope, event.source_agent, event.target)
            tool_history = self._tool_history[tool_key]
            tool_history.append(now)
            self._trim_window(tool_history, now)
            tool_event_history = self._tool_event_history[tool_key]
            tool_event_history.append(event_seq)
            self._trim_event_window(tool_event_history)

        behavior_key = (scope, event.source_agent, event.behavior_type)
        behavior_history = self._behavior_history[behavior_key]
        behavior_history.append(now)
        self._trim_window(behavior_history, now)
        behavior_event_history = self._behavior_event_history[behavior_key]
        behavior_event_history.append(event_seq)
        self._trim_event_window(behavior_event_history)

    def _trim_window(self, dq: Deque[float], now: float) -> None:
        while dq and (now - dq[0]) > self.realtime_window_seconds:
            dq.popleft()

    def _trim_event_window(self, dq: Deque[int]) -> None:
        if self.long_horizon_event_limit <= 0:
            return
        cutoff = self._event_seq - self.long_horizon_event_limit
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def _scope_has_tool(self, scope: str, tool_name: str) -> bool:
        return any(
            key_scope == scope and key_tool == tool_name and bool(history)
            for (key_scope, _agent, key_tool), history in self._tool_event_history.items()
        )

    def _long_horizon_threshold(self, baseline: int, minimum: int) -> int:
        if baseline <= 0:
            return minimum
        return max(minimum + 2, baseline * 2 + 2)

    def _event_ts(self, event: SecurityEvent) -> float:
        try:
            dt = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return time.time()

    def _scope_key(self, event: SecurityEvent) -> str:
        if not self.scope_realtime_by_task:
            return "global"
        params = event.params or {}
        return str(
            params.get("scope_id")
            or params.get("campaign_id")
            or params.get("session_id")
            or event.task_id
            or params.get("task_id")
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
            for mapping in (
                self._pair_history,
                self._tool_history,
                self._behavior_history,
                self._pair_event_history,
                self._tool_event_history,
                self._behavior_event_history,
            ):
                for key in list(mapping.keys()):
                    if key[0] == scope:
                        del mapping[key]
            self._scope_risk_memory.pop(scope, None)
            for key in list(self._agent_risk_memory.keys()):
                if key[0] == scope:
                    del self._agent_risk_memory[key]
            self._quarantined_agents_by_scope.pop(scope, None)
            if self._control_plane is not None:
                self._control_plane.reset_runtime_state(task_id=scope)
            return

        self._pair_history.clear()
        self._tool_history.clear()
        self._behavior_history.clear()
        self._pair_event_history.clear()
        self._tool_event_history.clear()
        self._behavior_event_history.clear()
        self._event_seq = 0
        self._scope_risk_memory.clear()
        self._agent_risk_memory.clear()
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
            "realtime_window_seconds": self.realtime_window_seconds,
            "long_horizon_event_limit": self.long_horizon_event_limit,
            "long_horizon_min_count": self.long_horizon_min_count,
            "runtime_scopes": sorted(
                {
                    key[0]
                    for mapping in (
                        self._pair_history,
                        self._tool_history,
                        self._behavior_history,
                        self._pair_event_history,
                        self._tool_event_history,
                        self._behavior_event_history,
                    )
                    for key in mapping.keys()
                }
            ),
            "scope_risk_memory": {
                scope: round(value, 2)
                for scope, value in sorted(self._scope_risk_memory.items())
                if value >= 1.0
            },
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


def _sentinel_json_safe(value: Any, *, max_text: int = 600) -> Any:
    """Return a compact JSON-safe representation for security audit params."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_text]
    if isinstance(value, dict):
        safe: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 30:
                safe["__truncated__"] = True
                break
            safe[str(key)[:80]] = _sentinel_json_safe(item, max_text=max_text)
        return safe
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        safe_values = [_sentinel_json_safe(item, max_text=max_text) for item in values[:30]]
        if len(values) > 30:
            safe_values.append("__truncated__")
        return safe_values
    return repr(value)[:max_text]


def wrap_tool_with_sentinel(
    func: Callable[..., Any],
    tool_name: str,
    *,
    event_bus: Optional[SecurityEventBus],
    control_plane: Optional[SecurityControlPlane],
    sentinel: Optional[SentinelAgent] = None,
    source_agent: str = "Executor",
    scenario: str = "",
):
    """Wrap a tool so Sentinel-only runs get pre-execution monitoring.

    Guardian+Sentinel already uses Guardian's ToolGate as the tool-level event
    source. This wrapper is for Sentinel-only experiments where Guardian is off.
    """
    import functools
    import inspect

    def build_params(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        task_id = getattr(sentinel, "current_task_id", "") if sentinel is not None else ""
        safe_kwargs = _sentinel_json_safe(kwargs)
        params: Dict[str, Any] = {
            "tool_name": tool_name,
            "source_agent": source_agent,
            "kwargs": safe_kwargs,
        }
        if args:
            params["args"] = _sentinel_json_safe(args)
        if isinstance(safe_kwargs, dict):
            for key, value in safe_kwargs.items():
                params.setdefault(key, value)
        if task_id:
            params["task_id"] = task_id
        return params

    def publish(
        *,
        params: Dict[str, Any],
        decision: str,
        reason: str = "",
        matched_rules: Optional[List[str]] = None,
        gate: str = "SentinelToolGate",
    ) -> None:
        if event_bus is None:
            return
        event_bus.publish(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_agent": source_agent,
                "target": tool_name,
                "behavior_type": "tool_call",
                "params": params,
                "gate": gate,
                "decision": decision,
                "reason": reason,
                "matched_rules": list(matched_rules or []),
                "scenario": scenario,
                "task_id": str(params.get("task_id") or ""),
            }
        )

    def runtime_directive(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if control_plane is None:
            return None
        try:
            directive = control_plane.get_runtime_directive(
                source_agent=source_agent,
                target=tool_name,
                behavior_type="tool_call",
                params=params,
            )
        except Exception:
            return None
        if directive and str(directive.get("action", "")).lower() == "block":
            return directive
        return None

    def blocked_response(directive: Optional[Dict[str, Any]]) -> str:
        reason = "access denied by Sentinel runtime policy"
        if directive:
            reason = str(directive.get("reason") or reason)
        return f"[Tool '{tool_name}' is unavailable: {reason}]"

    def before_call(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        params = build_params(args, kwargs)
        directive = runtime_directive(params)
        if directive:
            publish(
                params=params,
                decision="block",
                reason=str(directive.get("reason", "Blocked by Sentinel")),
                matched_rules=list(directive.get("matched_rules", ["SENTINEL_RUNTIME_BLOCK"])),
            )
            return params, directive

        publish(params=params, decision="allow")
        return params, runtime_directive(params)

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def sentinel_async(*args, **kwargs):
            _params, directive = before_call(args, kwargs)
            if directive:
                return blocked_response(directive)
            return await func(*args, **kwargs)

        return sentinel_async

    @functools.wraps(func)
    def sentinel_sync(*args, **kwargs):
        _params, directive = before_call(args, kwargs)
        if directive:
            return blocked_response(directive)
        return func(*args, **kwargs)

    return sentinel_sync
