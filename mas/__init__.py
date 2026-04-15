"""
MAS — Multi-Agent System 多智能体架构

核心组件:
    MASTeam:   多智能体团队编排（Planner + Workers，手动编排确定性流转）
    基于 agent_framework 的 Agent + FunctionTool 实现

工厂函数:
    mas.factory.create_gaia_team()   创建 GAIA 测试团队（5 个专业化 Worker）
    mas.factory.create_tamas_team()  创建 TAMAS 安全测试团队（5+1 Worker + Guardian）

架构:
    User Task
        ↓
    Planner (规划) → PLAN: 1. action → Assign: Worker_X  2. action → Assign: Worker_Y
        ↓
    解析 PLAN → 按序调度
        ↓
    Worker_X.run(step_1) → RESULT
    Worker_Y.run(step_2) → RESULT
        ↓                    ↑
        └── HANDOFF 链式协作 ─┘  (Worker 可直接转交另一个 Worker)
        ↓
    Planner (审查所有结果) → FINAL_ANSWER 或 新 PLAN → 循环
"""

from mas.team import MASTeam
from mas.metrics import MetricsCollector, MetricsLogger, TaskMetrics
from mas.sentinel import SentinelAgent, SecurityEventBus, SentinelAction, SentinelAssessment, SecurityControlPlane

__all__ = [
    "MASTeam",
    "MetricsCollector",
    "MetricsLogger",
    "TaskMetrics",
    "SentinelAgent",
    "SecurityEventBus",
    "SentinelAction",
    "SentinelAssessment",
    "SecurityControlPlane",
]
