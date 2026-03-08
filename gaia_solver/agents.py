"""
多智能体定义 - 结合 Microsoft Agent Framework + MetaGPT SOP 思想

架构说明 (已从 AutoGen v0.7 迁移到 agent-framework)：
===========
借鉴 MetaGPT 的 "软件公司" SOP 概念，使用手动编排实现
确定性的 Agent 流转：

1. Planner (规划师 - 对应 MetaGPT 产品经理)
   - 分析问题，制定分步求解计划

2. Executor (执行者 - 对应 MetaGPT 工程师)
   - 调用工具函数执行计划中的每一步
   - 拥有搜索、文件读取、代码执行、图片分析、音频转录等工具

3. Verifier (验证者 - 对应 MetaGPT QA 工程师)
   - 检查执行结果，验证答案格式
   - 输出最终的 FINAL_ANSWER

使用手动编排确保确定性 SOP 流转: Planner → Executor(多轮) → Verifier
"""
import re
import asyncio
from agent_framework import Agent, FunctionTool, Message
from agent_framework.openai import OpenAIChatClient

from gaia_solver.config import get_text_client, get_vision_client
from gaia_solver import tools


# ============================================================
# System Prompts（融入 MetaGPT SOP 思想）
# ============================================================

PLANNER_SYSTEM_MSG = """You are the Planner. Create a SHORT plan (2-3 steps max). NEVER answer yourself.

Format:
PLAN:
1. [action] -> Tool: [tool_name]
2. [action] -> Tool: [tool_name]
Expected format: [number/text/list]

Rules:
- Calculations: ALWAYS use execute_python, never mental math.
- Known facts (records, constants, dates): use execute_python directly.
- Obscure facts: search_web, then fetch_webpage if needed.
- Images: analyze_image with SPECIFIC question.
- Audio: transcribe_audio first.
- YouTube: analyze_youtube_video(url, question) to analyze video content directly.
- Specific webpages/changelogs: fetch_webpage for simple pages, browse_webpage for JS-heavy/dynamic sites.
- Interactive pages (forms, tabs, pagination): browse_webpage + browser_click to navigate.
- Text/letter puzzles: use execute_python to process programmatically.
- NEVER say FINAL_ANSWER. Only create a plan for the Executor.
"""

EXECUTOR_SYSTEM_MSG = """You are the Executor. Call tools and find answers.

Tools: search_web(query), search_wikipedia(topic,section), fetch_webpage(url), browse_webpage(url,question), browser_click(selector,wait_after), browser_scroll(direction), analyze_youtube_video(url,question), read_attachment(file_name), read_excel_file(file_name,sheet_name), read_docx_file(file_name), read_pptx_file(file_name), read_text_file(file_name), analyze_image(file_name,question), transcribe_audio(file_name), execute_python(code)

Rules:
1. Math: ALWAYS use execute_python.
2. Search failed? Try different keywords or fetch_webpage with URL from results.
3. YouTube: Use analyze_youtube_video(url, question) with a SPECIFIC question about the video.
4. Dynamic/JS pages: Use browse_webpage(url) instead of fetch_webpage. Can follow up with browser_click/browser_scroll.
5. NEVER repeat same tool call. Vary approach.
6. When done: say RESULT: <answer>
7. Be concise. Only report key findings.
"""

VERIFIER_SYSTEM_MSG = """You format and verify answers. Output FINAL_ANSWER in EVERY response.

Format:
FINAL_ANSWER: <answer>

Rules:
1. "how many thousand" → divide by 1000. E.g. 17055 → 17
2. Numbers → no units unless asked.
3. Names → name only, no extra words.
4. Lists → comma+space: "a, b, c". Only items Executor found.
5. No units/explanation/quotes unless asked.
6. TRUST the Executor's findings. Only fix FORMAT, not content.
7. If no result: best guess from partial info.
8. Species: full name (e.g. "Rockhopper penguin").
9. Always provide FINAL_ANSWER even if unsure.
"""


# ============================================================
# 工具函数列表构建
# ============================================================

def get_all_gaia_tools() -> list:
    """
    构建所有 GAIA 通用工具的 FunctionTool 列表（公共接口）。
    
    包含 15 个工具：
    - 网络搜索: search_web, search_wikipedia, fetch_webpage
    - 浏览器交互: browse_webpage, browser_click, browser_scroll
    - 视频分析: analyze_youtube_video
    - 文件读取: read_attachment, read_excel_file, read_docx_file, read_pptx_file, read_text_file
    - 多模态: analyze_image, transcribe_audio
    - 代码执行: execute_python
    
    可被 gaia_solver 和 tamas_adapter 共同使用。
    """
    return [
        FunctionTool(name="search_web", description="Search DuckDuckGo+Wikipedia. Input: query.", func=tools.search_web),
        FunctionTool(name="search_wikipedia", description="Wikipedia page. Input: topic, optional section.", func=tools.search_wikipedia),
        FunctionTool(name="fetch_webpage", description="Fetch URL text. Input: url.", func=tools.fetch_webpage),
        FunctionTool(name="analyze_youtube_video", description="Analyze YouTube video with vision model. Input: url, question.", func=tools.analyze_youtube_video),
        FunctionTool(name="read_attachment", description="Auto-read any file. Input: file_name.", func=tools.read_attachment),
        FunctionTool(name="read_excel_file", description="Read Excel with colors. Input: file_name, optional sheet_name.", func=tools.read_excel_file),
        FunctionTool(name="read_docx_file", description="Read Word doc. Input: file_name.", func=tools.read_docx_file),
        FunctionTool(name="read_pptx_file", description="Read PowerPoint. Input: file_name.", func=tools.read_pptx_file),
        FunctionTool(name="read_text_file", description="Read text/csv/json/py. Input: file_name.", func=tools.read_text_file),
        FunctionTool(name="analyze_image", description="Vision model for images. Input: file_name, question.", func=tools.analyze_image),
        FunctionTool(name="transcribe_audio", description="Audio to text. Input: file_name.", func=tools.transcribe_audio),
        FunctionTool(name="execute_python", description="Run Python code. ALWAYS use for math. Input: code.", func=tools.execute_python),
        FunctionTool(name="browse_webpage", description="Open URL in real browser (JS support). Input: url, optional question for vision analysis.", func=tools.browse_webpage),
        FunctionTool(name="browser_click", description="Click element on current page. Input: CSS selector, optional wait_after ms.", func=tools.browser_click),
        FunctionTool(name="browser_scroll", description="Scroll current page. Input: direction ('down' or 'up').", func=tools.browser_scroll),
    ]


# ============================================================
# Agent 创建
# ============================================================

def _create_agents(has_image: bool = False):
    """创建 Planner, Executor, Verifier 三个 Agent。
    如果任务包含图片，Executor 使用视觉模型以获得多模态能力。
    """
    text_client = get_text_client()
    executor_client = get_vision_client() if has_image else text_client

    # 限制 Executor 工具调用迭代次数（默认 40 太大，导致 token 浪费和超时）
    executor_client.function_invocation_configuration["max_iterations"] = 10

    planner = Agent(
        client=text_client,
        name="Planner",
        instructions=PLANNER_SYSTEM_MSG,
        description="Analyzes tasks and creates step-by-step plans.",
        default_options={"temperature": 0},
    )

    executor = Agent(
        client=executor_client,
        name="Executor",
        instructions=EXECUTOR_SYSTEM_MSG,
        description="Executes plan steps by calling tools.",
        tools=get_all_gaia_tools(),
        default_options={"temperature": 0},
    )

    verifier = Agent(
        client=text_client,
        name="Verifier",
        instructions=VERIFIER_SYSTEM_MSG,
        description="Verifies results and outputs FINAL_ANSWER.",
        default_options={"temperature": 0},
    )

    return planner, executor, verifier


async def run_sop_workflow(task_prompt: str, max_turns: int = 15,
                           verbose: bool = True, has_image: bool = False) -> list:
    """
    手动编排的 SOP 工作流。
    确定性流转: Planner → Executor(多轮) → Verifier
    
    Token 优化策略:
    - Planner 只收到 task_prompt（不带多余指令）
    - Executor 收到 task_prompt + planner 计划（不重复 system prompt 内容）
    - Verifier 只收到 question + executor 结论（不带中间过程）
    - 工具结果自动截断到 3000 字符
    
    Returns:
        消息列表 (list of dicts with 'role', 'author', 'text')
    """
    planner, executor, verifier = _create_agents(has_image=has_image)
    messages = []
    
    PLANNER_TIMEOUT = 45
    EXECUTOR_TIMEOUT = 300
    VERIFIER_TIMEOUT = 30
    
    def record_msg(author: str, text: str):
        messages.append({"role": "assistant", "author": author, "text": text})
        if verbose:
            display = text[:300] + "..." if len(text) > 300 else text
            try:
                print(f"  [{author}]: {display}")
            except UnicodeEncodeError:
                print(f"  [{author}]: {display.encode('ascii', 'replace').decode('ascii')}")
    
    # ========== Step 1: Planner ==========
    try:
        planner_result = await asyncio.wait_for(planner.run(task_prompt), timeout=PLANNER_TIMEOUT)
        planner_text = planner_result.text if hasattr(planner_result, 'text') else str(planner_result)
        record_msg("Planner", planner_text)
    except asyncio.TimeoutError:
        record_msg("Planner", "Timeout")
        return messages
    except Exception as e:
        record_msg("Planner", f"Error: {e}")
        return messages
    
    # Planner 不应直接回答; 即使意外包含 FINAL_ANSWER 也继续流程
    # 确保每道题都经过 Executor 工具执行 + Verifier 格式化
    
    # ========== Step 2: Executor ==========
    # 只传必需内容: task + plan（不重复完整 system prompt 中已有的工具说明）
    executor_prompt = (
        f"{task_prompt}\n\n"
        f"Plan:\n{planner_text}\n\n"
        f"Execute now. Say RESULT: <answer> when done."
    )
    
    executor_text = ""
    try:
        executor_result = await asyncio.wait_for(
            executor.run(executor_prompt), timeout=EXECUTOR_TIMEOUT
        )
        executor_text = executor_result.text if hasattr(executor_result, 'text') else str(executor_result)
        record_msg("Executor", executor_text)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        record_msg("Executor", "Timeout")
    except Exception as e:
        record_msg("Executor", f"Error: {e}")

    # Executor 补充轮: 只在没有 RESULT 时触发，且只传精简信息
    if executor_text and "RESULT" not in executor_text.upper() and "FINAL_ANSWER" not in executor_text.upper():
        try:
            # 只提取 executor 的关键发现，不重复原始任务
            key_findings = executor_text[-1500:] if len(executor_text) > 1500 else executor_text
            followup = f"Progress so far:\n{key_findings}\n\nContinue. Say RESULT: <answer>"
            executor_result2 = await asyncio.wait_for(
                executor.run(followup), timeout=EXECUTOR_TIMEOUT
            )
            executor_text2 = executor_result2.text if hasattr(executor_result2, 'text') else str(executor_result2)
            record_msg("Executor", executor_text2)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    # ========== Step 3: Verifier ==========
    # 只传原始问题 + executor 最终结论（大幅减少 token）
    # 从 task_prompt 中提取纯问题部分
    q_match = re.search(r'\*\*Question:\*\*\s*(.+?)(?:\n\*\*|$)', task_prompt, re.DOTALL)
    question_only = q_match.group(1).strip() if q_match else task_prompt[:500]
    
    # 只取 executor 最后一条消息的结论部分
    executor_msgs = [m for m in messages if m["author"] == "Executor"]
    last_executor = executor_msgs[-1]["text"] if executor_msgs else "No result"
    # 截取关键结果（最后 800 字符通常包含 RESULT）
    executor_conclusion = last_executor[-800:] if len(last_executor) > 800 else last_executor
    
    verifier_prompt = (
        f"Question: {question_only}\n\n"
        f"Executor found: {executor_conclusion}\n\n"
        f"Output FINAL_ANSWER:"
    )
    
    try:
        verifier_result = await asyncio.wait_for(
            verifier.run(verifier_prompt), timeout=VERIFIER_TIMEOUT
        )
        verifier_text = verifier_result.text if hasattr(verifier_result, 'text') else str(verifier_result)
        record_msg("Verifier", verifier_text)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        record_msg("Verifier", "Timeout")
    except Exception as e:
        record_msg("Verifier", f"Error: {e}")
    
    return messages


def create_team(max_turns: int = 20, has_image: bool = False):
    """
    返回一个兼容旧接口的 workflow 对象。
    内部使用手动编排实现 SOP 流转。
    """
    return SOPWorkflow(max_turns=max_turns, has_image=has_image)


class SOPWorkflow:
    """手动编排的 SOP 工作流，兼容旧的 team.run() 接口"""
    
    def __init__(self, max_turns: int = 20, has_image: bool = False):
        self.max_turns = max_turns
        self.has_image = has_image
    
    async def run(self, task_prompt: str):
        """运行工作流并返回结果"""
        messages = await run_sop_workflow(
            task_prompt, 
            max_turns=self.max_turns, 
            verbose=True,
            has_image=self.has_image,
        )
        return SOPWorkflowResult(messages)


class SOPWorkflowResult:
    """工作流结果，模拟旧 API"""
    
    def __init__(self, messages: list):
        self._messages = messages
    
    def get_outputs(self):
        """返回消息列表（兼容旧接口）"""
        # 转换为简单的 message 对象列表
        return [self._messages]
