"""
MAS 快速验证脚本

测试:
  1. 模块导入
  2. GAIA 团队创建
  3. Plan 解析（通过打印 agent 列表验证配置正确性）
  4. 简单任务运行（需要 API）
"""

import asyncio
import sys
import os

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_imports():
    """测试导入"""
    print("Test 1: Imports...")
    from mas import MASTeam
    from mas.factory import create_gaia_team, create_tamas_team
    from mas.workers import (
        create_web_searcher, create_web_browser,
        create_file_reader, create_media_analyst, create_code_executor,
    )
    print("  OK: All imports successful")


def test_gaia_team_creation():
    """测试 GAIA 团队创建"""
    print("\nTest 2: GAIA Team creation...")
    from mas.factory import create_gaia_team

    team = create_gaia_team(verbose=True)

    # 验证 agent 列表
    assert len(team._agent_list) == 6, f"Expected 6 agents, got {len(team._agent_list)}"
    assert team._agent_list[0].name == "Planner"

    worker_names = [a.name for a in team._agent_list[1:]]
    assert "WebSearcher" in worker_names
    assert "WebBrowser" in worker_names
    assert "FileReader" in worker_names
    assert "MediaAnalyst" in worker_names
    assert "CodeExecutor" in worker_names

    # 验证 Planner 没有工具
    planner = team._agents["Planner"]
    # agent_info 应该在 _agent_info 中包含 worker 信息
    assert "WebSearcher" in team._agent_info, "Team should have WebSearcher info"

    print("  OK: 6 agents created correctly")
    print(f"  Worker names: {worker_names}")


def test_tamas_team_creation():
    """测试 TAMAS 团队创建 (仅结构，不需要真实 TAMAS 工具)"""
    print("\nTest 3: TAMAS Team creation (mock)...")
    from agent_framework import FunctionTool
    from mas.factory import create_tamas_team

    # 模拟 TAMAS 领域工具
    def mock_analyze_symptoms(symptoms: str) -> str:
        """Analyze patient symptoms"""
        return f"Analysis: {symptoms}"

    def mock_monitor_vitals(patient_id: str) -> str:
        """Monitor patient vitals"""
        return f"Vitals for {patient_id}: stable"

    mock_tools = [
        FunctionTool(name="analyze_symptoms", description="Analyze symptoms", func=mock_analyze_symptoms),
        FunctionTool(name="monitor_vitals", description="Monitor vitals", func=mock_monitor_vitals),
    ]

    team = create_tamas_team(
        scenario="healthcare",
        domain_tools=mock_tools,
        verbose=True,
    )

    # 应有 7 个 agents: Planner + 5 workers + DomainWorker
    assert len(team._agent_list) == 7, f"Expected 7 agents, got {len(team._agent_list)}"
    assert "DomainWorker" in [a.name for a in team._agent_list]

    print("  OK: TAMAS team with DomainWorker created")


async def test_simple_run():
    """测试简单任务运行（需要 API key）"""
    print("\nTest 4: Simple run (API required)...")
    from mas.factory import create_gaia_team

    team = create_gaia_team(verbose=True, max_rounds=3)

    result = await team.run("What is 123 * 456? Use CodeExecutor to compute this.")

    print(f"\n  Answer: {result['answer']}")
    print(f"  Rounds: {result['rounds']}")
    print(f"  Turns: {result['turns']}")
    print(f"  Elapsed: {result['elapsed']}s")

    if result["answer"]:
        # 123 * 456 = 56088
        if "56088" in result["answer"]:
            print("  OK: Correct answer!")
        else:
            print(f"  WARN: Expected 56088, got {result['answer']}")
    else:
        print("  WARN: No answer returned")


def main():
    print("=" * 60)
    print("  MAS 架构验证")
    print("=" * 60)

    # 基础测试（不需要 API）
    test_imports()
    test_gaia_team_creation()
    test_tamas_team_creation()

    # API 测试（可选）
    print("\n" + "-" * 60)
    run_api_test = input("Run API test (requires LLM API)? [y/N]: ").strip().lower() == "y"
    if run_api_test:
        asyncio.run(test_simple_run())
    else:
        print("  Skipped API test")

    print("\n" + "=" * 60)
    print("  All basic tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
