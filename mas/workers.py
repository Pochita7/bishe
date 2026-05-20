"""Predefined Worker agent configurations.

Each factory returns the same configuration shape consumed by MASTeam:
``{"name", "description", "system_message", "tools"}``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_framework import FunctionTool


WorkerConfig = Dict[str, Any]
WorkerTools = List[FunctionTool]


_HANDOFF_INSTRUCTIONS = """

Collaboration:
  If your task requires another worker's help, you may add at the END of your output:
    HANDOFF: <WorkerName> - <specific task for them>
  Available workers: WebSearcher, WebBrowser, FileReader, MediaAnalyst, CodeExecutor
  Examples:
    HANDOFF: WebBrowser - read the page at https://example.com/data
    HANDOFF: CodeExecutor - compute the sum of [2, 3, 5, 7, 11]
  Only use HANDOFF when genuinely needed. Still include your own RESULT first."""


def _worker_config(
    *,
    name: str,
    description: str,
    system_message: str,
    tools: WorkerTools,
) -> WorkerConfig:
    return {
        "name": name,
        "description": description,
        "system_message": system_message,
        "tools": tools,
    }


_WEB_SEARCHER_SYSTEM_MESSAGE = (
    "You are WebSearcher. Find information from the web.\n\n"
    "Rules:\n"
    "1. Use search_web for general queries.\n"
    "2. Use search_wikipedia for encyclopedic / factual lookups.\n"
    "3. For multi-fact assignments, first call search_web ONCE with the full assignment; if it returns exact values, stop.\n"
    "4. Return RELEVANT information concisely - do NOT dump raw HTML.\n"
    "5. If first search fails, try rephrasing the query ONCE. Do NOT keep retrying.\n"
    "6. Say RESULT: <findings> when done.\n"
    "7. If you found URLs that need deeper reading, add:\n"
    "   HANDOFF: WebBrowser - read <url> for <specific info>\n"
    "EFFICIENCY: Use at most 2 tool calls total. Focus on the MOST relevant query first."
    + _HANDOFF_INSTRUCTIONS
)

_WEB_BROWSER_SYSTEM_MESSAGE = (
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
)

_FILE_READER_SYSTEM_MESSAGE = (
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
)

_MEDIA_ANALYST_SYSTEM_MESSAGE = (
    "You are MediaAnalyst. Analyze multimedia content.\n\n"
    "Rules:\n"
    "1. analyze_image for images - provide a SPECIFIC question about the image.\n"
    "2. transcribe_audio for audio files (.mp3, .wav, .flac).\n"
    "3. analyze_youtube_video for YouTube URLs - provide a SPECIFIC question.\n"
    "4. If a media tool returns CACHED_IMAGE_ANALYSIS, VISION_UNAVAILABLE, or Error analyzing image, STOP; do not retry the same tool.\n"
    "5. Do not narrate hidden reasoning. Return concise findings only.\n"
    "6. Return structured, relevant findings.\n"
    "7. Say RESULT: <findings> when done.\n"
    "8. If media contains data that needs search/verification, add:\n"
    "   HANDOFF: WebSearcher - search for <info from media>\n"
    "EFFICIENCY: Call each tool ONCE with a precise query. Do NOT repeat calls."
    + _HANDOFF_INSTRUCTIONS
)

_CODE_EXECUTOR_SYSTEM_MESSAGE = (
    "You are CodeExecutor. Write and run Python code.\n\n"
    "Rules:\n"
    "1. ALWAYS use print() to output results.\n"
    "2. ALWAYS compute - never estimate or guess.\n"
    "3. Keep code concise and focused on the specific computation.\n"
    "4. If code fails, debug and retry with a different approach (max 1 retry).\n"
    "5. Import standard libraries as needed (math, datetime, re, json, etc.).\n"
    "6. If the task asks for scaled units (thousand/million/billion), report both raw value and requested scaled value.\n"
    "7. Use Previous results as your data source. If required data is missing, never infer or invent it.\n"
    "8. Say RESULT: <answer> when done.\n"
    "9. If you need external data to compute, add:\n"
    "   HANDOFF: WebSearcher - search for <missing data>\n"
    "EFFICIENCY: Write ONE comprehensive script. Do NOT split into multiple small calls."
    + _HANDOFF_INSTRUCTIONS
)


def create_web_searcher(tools: WorkerTools) -> WorkerConfig:
    return _worker_config(
        name="WebSearcher",
        description=(
            "Searches the web and Wikipedia for information. "
            "Use for external knowledge, facts, current events, or encyclopedic data."
        ),
        system_message=_WEB_SEARCHER_SYSTEM_MESSAGE,
        tools=tools,
    )


def create_web_browser(tools: WorkerTools) -> WorkerConfig:
    return _worker_config(
        name="WebBrowser",
        description=(
            "Fetches and browses web pages. "
            "Use for reading specific URLs, dynamic/JS pages, "
            "clicking elements, scrolling, and extracting page content."
        ),
        system_message=_WEB_BROWSER_SYSTEM_MESSAGE,
        tools=tools,
    )


def create_file_reader(tools: WorkerTools) -> WorkerConfig:
    return _worker_config(
        name="FileReader",
        description=(
            "Reads and parses files: Excel (.xlsx/.csv), Word (.docx), "
            "PowerPoint (.pptx), plain text, and generic attachments."
        ),
        system_message=_FILE_READER_SYSTEM_MESSAGE,
        tools=tools,
    )


def create_media_analyst(tools: WorkerTools) -> WorkerConfig:
    return _worker_config(
        name="MediaAnalyst",
        description=(
            "Analyzes images, transcribes audio, and analyzes YouTube videos. "
            "Use for visual questions, audio transcription, or video content analysis."
        ),
        system_message=_MEDIA_ANALYST_SYSTEM_MESSAGE,
        tools=tools,
    )


def create_code_executor(tools: WorkerTools) -> WorkerConfig:
    return _worker_config(
        name="CodeExecutor",
        description=(
            "Executes Python code for calculations, data analysis, "
            "string manipulation, date math, and any computation. "
            "Use whenever exact computation is needed."
        ),
        system_message=_CODE_EXECUTOR_SYSTEM_MESSAGE,
        tools=tools,
    )
