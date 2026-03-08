"""
TAMAS 适配器 - 多智能体系统安全测试框架

核心组件:
- tools.py   TAMAS 领域工具加载（良性 + 恶意 FunctionTool）
- agents.py  TAMAS 专用 Agent 创建（Executor 含领域工具）
- loader.py  TAMAS 数据解析（clean/attack query 提取）
- prompt_builder.py  Prompt 构建（clean/attack 模式）
- evaluator.py  安全评估（ARIA 评分 + 工具调用检测）
- runner.py  测试运行器（clean vs attack 对比实验）
- guardian.py  Guardian 防火墙 (5 Gates: ToolDesc + Input + Plan + Tool + Output)
"""

from tamas_adapter.guardian import (
    Guardian,
    Action,
    GuardianDecision,
    GuardianLog,
    ToolDescRisk,
    ToolDescScanResult,
)
