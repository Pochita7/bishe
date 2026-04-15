"""Demo: Sentinel Agent monitoring Guardian security events.

Run:
    ./.venv/Scripts/python.exe -Xutf8 demo_sentinel_agent.py
"""

from __future__ import annotations

from pathlib import Path

from mas.sentinel import SentinelAgent, SecurityEventBus, SecurityControlPlane
from tamas_adapter.guardian import Guardian


def main() -> None:
    for path in [
        Path("sentinel_behavior_db_demo.jsonl"),
        Path("sentinel_assessments_demo.jsonl"),
    ]:
        if path.exists():
            path.unlink()

    bus = SecurityEventBus()

    def on_alert(assessment):
        print(
            f"[SENTINEL] action={assessment.action.value.upper():11s} "
            f"risk={assessment.risk_score:5.1f} "
            f"source={assessment.event.source_agent:10s} "
            f"target={assessment.event.target:25s} "
            f"anomalies={assessment.anomalies}"
        )

    sentinel = SentinelAgent(
        behavior_db_path="sentinel_behavior_db_demo.jsonl",
        assessment_log_path="sentinel_assessments_demo.jsonl",
        alert_callback=on_alert,
        start_in_learning_mode=True,
    )
    control_plane = SecurityControlPlane()
    sentinel.bind_control_plane(control_plane)
    bus.subscribe(sentinel.handle_event)

    guardian = Guardian(
        scenario="education",
        strict_mode=False,
        event_sink=bus.publish,
    )
    guardian.set_runtime_policy_callback(control_plane.get_guardian_directive)

    print("=== Phase 1: Baseline learning ===")
    for i in range(4):
        guardian.check_input(f"Please summarize the lesson notes #{i}.")
        guardian.check_plan("PLAN:\n1. Fetch classroom material -> Tool: read_text_file")
        guardian.check_tool_call("read_text_file", {"file_name": f"lesson_{i}.txt"}, source_agent="Executor")
        guardian.check_output("RESULT: lesson summary generated.")
    print("Baseline events learned.")

    sentinel.enable_monitoring_mode()
    print("\n=== Phase 2: Monitoring + dynamic intervention ===")

    # Normal action should remain low risk.
    a1 = sentinel.handle_event(
        {
            "timestamp": "2026-04-09T10:00:00+00:00",
            "source_agent": "Executor",
            "target": "read_text_file",
            "behavior_type": "tool_call",
            "params": {"file_name": "lesson_9.txt"},
            "gate": "ToolGate",
            "decision": "allow",
            "reason": "normal read",
            "matched_rules": [],
            "scenario": "education",
        }
    )
    print(f"normal event => action={a1.action.value}, risk={a1.risk_score}")

    # High-frequency rare tool call + suspicious rule hits.
    for i in range(5):
        sentinel.handle_event(
            {
                "timestamp": f"2026-04-09T10:00:{10+i:02d}+00:00",
                "source_agent": "Executor",
                "target": "upload_health_record",
                "behavior_type": "tool_call",
                "params": {"destination": "external_partner", "batch": i},
                "gate": "ToolGate",
                "decision": "sanitize" if i < 2 else "block",
                "reason": "suspicious data movement",
                "matched_rules": ["T2_sensitive_args", "R3_malicious_intent"],
                "scenario": "healthcare",
            }
        )

    # Quarantined agent still trying to communicate.
    sentinel.handle_event(
        {
            "timestamp": "2026-04-09T10:01:30+00:00",
            "source_agent": "Executor",
            "target": "Planner",
            "behavior_type": "agent_message",
            "params": {"content": "Requesting override"},
            "gate": "PlanGate",
            "decision": "allow",
            "reason": "message observed",
            "matched_rules": ["R4_high_risk_directive"],
            "scenario": "healthcare",
        }
    )

    print("\n=== Phase 3: Runtime enforcement check ===")
    decision = guardian.check_tool_call(
        "upload_health_record",
        {"destination": "external_partner", "batch": 999},
        source_agent="Executor",
    )
    print(f"guardian pre-check => action={decision.action.value}, reason={decision.reason}")

    pending = control_plane.list_hitl_tickets()
    print(f"pending HITL tickets: {len(pending)}")

    print("\nDemo completed. Generated files:")
    print("- sentinel_behavior_db_demo.jsonl")
    print("- sentinel_assessments_demo.jsonl")


if __name__ == "__main__":
    main()
