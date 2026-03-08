"""
预定义 Worker Agent 配置

每个 Worker 专注一类工具:
  - WebSearcher:  网络搜索 (search_web, search_wikipedia)
  - WebBrowser:   网页浏览/交互 (fetch_webpage, browse_webpage, browser_click, browser_scroll)
  - FileReader:   文件读取 (read_attachment, read_excel_file, read_docx_file, read_pptx_file, read_text_file)
  - MediaAnalyst: 多媒体分析 (analyze_image, transcribe_audio, analyze_youtube_video)
  - CodeExecutor: 代码执行 (execute_python)

协作机制:
  Worker 完成自身工作后，如果发现后续需要另一个 Worker 协助，
  可以在输出末尾添加 HANDOFF: <WorkerName> - <具体任务描述>，
  编排引擎会自动将任务转交给目标 Worker，无需回到 Planner。

每个函数返回 dict 配置（name, description, system_message, tools），
由 MASTeam 负责转换为 agent_framework.Agent。
"""

from typing import List, Dict, Any


# 所有 Worker 共享的 HANDOFF 说明
_HANDOFF_INSTRUCTIONS = """

Collaboration:
  If your task requires another worker's help, you may add at the END of your output:
    HANDOFF: <WorkerName> - <specific task for them>
  Available workers: WebSearcher, WebBrowser, FileReader, MediaAnalyst, CodeExecutor
  Examples:
    HANDOFF: WebBrowser - read the page at https://example.com/data
    HANDOFF: CodeExecutor - compute the sum of [2, 3, 5, 7, 11]
  Only use HANDOFF when genuinely needed. Still include your own RESULT first."""


def create_web_searcher(tools: list) -> Dict[str, Any]:
    """创建网络搜索 Worker 配置"""
    return {
        "name": "WebSearcher",
        "description": (
            "Searches the web and Wikipedia for information. "
            "Use for external knowledge, facts, current events, or encyclopedic data."
        ),
        "system_message": (
            "You are WebSearcher. Find information from the web.\n\n"
            "Rules:\n"
            "1. Use search_web for general queries.\n"
            "2. Use search_wikipedia for encyclopedic / factual lookups.\n"
            "3. Return RELEVANT information concisely — do NOT dump raw HTML.\n"
            "4. If first search fails, try rephrasing the query ONCE. Do NOT keep retrying.\n"
            "5. Say RESULT: <findings> when done.\n"
            "6. If you found URLs that need deeper reading, add:\n"
            "   HANDOFF: WebBrowser - read <url> for <specific info>\n"
            "EFFICIENCY: Use at most 3 tool calls total. Focus on the MOST relevant query first."
            + _HANDOFF_INSTRUCTIONS
        ),
        "tools": tools,
    }


def create_web_browser(tools: list) -> Dict[str, Any]:
    """创建网页浏览 Worker 配置"""
    return {
        "name": "WebBrowser",
        "description": (
            "Fetches and browses web pages. "
            "Use for reading specific URLs, dynamic/JS pages, "
            "clicking elements, scrolling, and extracting page content."
        ),
        "system_message": (
            "You are WebBrowser. Fetch and browse web pages.\n\n"
            "Rules:\n"
            "1. Use fetch_webpage for simple static pages.\n"
            "2. Use browse_webpage for dynamic/JS-heavy pages.\n"
            "3. Use browser_click and browser_scroll for page interaction.\n"
            "4. Extract only the RELEVANT content.\n"
            "5. Say RESULT: <findings> when done.\n"
            "6. If data needs computation/analysis, add:\n"
            "   HANDOFF: CodeExecutor - compute <specific task>\n"
            "EFFICIENCY: Use at most 3 tool calls total. Fetch the most important page first."
            + _HANDOFF_INSTRUCTIONS
        ),
        "tools": tools,
    }


def create_file_reader(tools: list) -> Dict[str, Any]:
    """创建文件读取 Worker 配置"""
    return {
        "name": "FileReader",
        "description": (
            "Reads and parses files: Excel (.xlsx/.csv), Word (.docx), "
            "PowerPoint (.pptx), plain text, and generic attachments."
        ),
        "system_message": (
            "You are FileReader. Read and extract file content.\n\n"
            "Rules:\n"
            "1. read_excel_file for .xlsx / .csv\n"
            "2. read_docx_file for .docx\n"
            "3. read_pptx_file for .pptx\n"
            "4. read_text_file for .txt / .json / .py / .md\n"
            "5. read_attachment as fallback for unknown formats.\n"
            "6. Summarize relevant content.\n"
            "7. Say RESULT: <findings> when done.\n"
            "8. If file data needs further processing, add:\n"
            "   HANDOFF: CodeExecutor - analyze/compute <specific task>\n"
            "EFFICIENCY: Read the file ONCE. Do NOT re-read, just summarize what you found."
            + _HANDOFF_INSTRUCTIONS
        ),
        "tools": tools,
    }


def create_media_analyst(tools: list) -> Dict[str, Any]:
    """创建多媒体分析 Worker 配置"""
    return {
        "name": "MediaAnalyst",
        "description": (
            "Analyzes images, transcribes audio, and analyzes YouTube videos. "
            "Use for visual questions, audio transcription, or video content analysis."
        ),
        "system_message": (
            "You are MediaAnalyst. Analyze multimedia content.\n\n"
            "Rules:\n"
            "1. analyze_image for images — provide a SPECIFIC question about the image.\n"
            "2. transcribe_audio for audio files (.mp3, .wav, .flac).\n"
            "3. analyze_youtube_video for YouTube URLs — provide a SPECIFIC question.\n"
            "4. Return structured, relevant findings.\n"
            "5. Say RESULT: <findings> when done.\n"
            "6. If media contains data that needs search/verification, add:\n"
            "   HANDOFF: WebSearcher - search for <info from media>\n"
            "EFFICIENCY: Call each tool ONCE with a precise query. Do NOT repeat calls."
            + _HANDOFF_INSTRUCTIONS
        ),
        "tools": tools,
    }


def create_code_executor(tools: list) -> Dict[str, Any]:
    """创建代码执行 Worker 配置"""
    return {
        "name": "CodeExecutor",
        "description": (
            "Executes Python code for calculations, data analysis, "
            "string manipulation, date math, and any computation. "
            "Use whenever exact computation is needed."
        ),
        "system_message": (
            "You are CodeExecutor. Write and run Python code.\n\n"
            "Rules:\n"
            "1. ALWAYS use print() to output results.\n"
            "2. ALWAYS compute — never estimate or guess.\n"
            "3. Keep code concise and focused on the specific computation.\n"
            "4. If code fails, debug and retry with a different approach (max 1 retry).\n"
            "5. Import standard libraries as needed (math, datetime, re, json, etc.).\n"
            "6. Say RESULT: <answer> when done.\n"
            "7. If you need external data to compute, add:\n"
            "   HANDOFF: WebSearcher - search for <missing data>\n"
            "EFFICIENCY: Write ONE comprehensive script. Do NOT split into multiple small calls."
            + _HANDOFF_INSTRUCTIONS
        ),
        "tools": tools,
    }
