"""
配置文件 - 模型、路径、API 设置
适配 Microsoft Agent Framework (agent-framework) Python SDK
"""
import os
from typing import Any

from token_accounting import record_openai_response_usage

# ========================
# 路径配置
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAIA_DATA_DIR = os.path.join(BASE_DIR, "gaia_data")
GAIA_LEVEL1_PATH = os.path.join(GAIA_DATA_DIR, "gaia_level1.jsonl")
GAIA_ATTACHMENTS_DIR = os.path.join(GAIA_DATA_DIR, "attachments")
WORK_DIR = os.path.join(BASE_DIR, "gaia_work_dir")

# 确保工作目录存在
os.makedirs(WORK_DIR, exist_ok=True)

# ========================
# LLM 配置 (DeepSeek 官方 OpenAI-compatible API)
# ========================
# 文本主路径 API Key。不要把真实密钥写进仓库，优先通过环境变量提供。
API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ARK_API_KEY")
    or ""
)
# DeepSeek 官方 API 地址
BASE_URL = (
    os.environ.get("DEEPSEEK_BASE_URL")
    or "https://api.deepseek.com"
)

# 文本模型，默认使用 DeepSeek V4 Flash
TEXT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high").strip().lower()

# 多模态/音频任务可单独配置兼容供应商；默认跟随文本配置
VISION_API_KEY = os.environ.get("GAIA_VISION_API_KEY") or API_KEY
VISION_BASE_URL = os.environ.get("GAIA_VISION_BASE_URL") or BASE_URL
VISION_MODEL = os.environ.get("GAIA_VISION_MODEL", TEXT_MODEL)

# 向后兼容
MODEL_NAME = TEXT_MODEL


def get_chat_default_options(temperature: float = 0) -> dict[str, Any]:
    """Default Agent Framework chat options.

    DeepSeek V4 defaults to thinking mode. Thinking + tool calls requires
    preserving reasoning_content across the internal tool loop; the current
    Agent Framework adapter does not expose that DeepSeek-specific field, so
    benchmark agents use non-thinking mode unless explicitly overridden.
    """
    options: dict[str, Any] = {"temperature": temperature}
    is_deepseek = "deepseek" in BASE_URL.lower() or TEXT_MODEL.startswith("deepseek-")
    if is_deepseek:
        thinking = DEEPSEEK_THINKING if DEEPSEEK_THINKING in {"enabled", "disabled"} else "disabled"
        options["extra_body"] = {"thinking": {"type": thinking}}
        if thinking == "enabled" and DEEPSEEK_REASONING_EFFORT in {"high", "max"}:
            options["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    return options


def get_text_client():
    """获取文本模型的 Agent Framework OpenAIChatClient 实例"""
    from agent_framework.openai import OpenAIChatClient

    return OpenAIChatClient(
        model_id=TEXT_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def get_vision_client():
    """获取视觉模型的 Agent Framework OpenAIChatClient 实例"""
    from agent_framework.openai import OpenAIChatClient

    return OpenAIChatClient(
        model_id=VISION_MODEL,
        api_key=VISION_API_KEY,
        base_url=VISION_BASE_URL,
    )


def get_openai_client():
    """获取原生 OpenAI 兼容客户端（用于 Vision、Whisper 等直接调用）"""
    from openai import OpenAI

    return _TrackedOpenAIClient(OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL))


class _TrackedCreateProxy:
    def __init__(self, inner: Any, operation: str):
        self._inner = inner
        self._operation = operation

    def create(self, *args, **kwargs):
        response = self._inner.create(*args, **kwargs)
        model = kwargs.get("model") or getattr(response, "model", "") or ""
        record_openai_response_usage(
            response,
            model=str(model),
            operation=self._operation,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TrackedChatNamespace:
    def __init__(self, inner: Any):
        self._inner = inner
        self.completions = _TrackedCreateProxy(inner.completions, "chat.completions.create")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TrackedAudioNamespace:
    def __init__(self, inner: Any):
        self._inner = inner
        self.transcriptions = _TrackedCreateProxy(inner.transcriptions, "audio.transcriptions.create")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TrackedResponsesNamespace:
    def __init__(self, inner: Any):
        self._inner = inner

    def create(self, *args, **kwargs):
        response = self._inner.create(*args, **kwargs)
        model = kwargs.get("model") or getattr(response, "model", "") or ""
        record_openai_response_usage(
            response,
            model=str(model),
            operation="responses.create",
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TrackedOpenAIClient:
    def __init__(self, inner: Any):
        self._inner = inner
        if hasattr(inner, "chat"):
            self.chat = _TrackedChatNamespace(inner.chat)
        if hasattr(inner, "audio"):
            self.audio = _TrackedAudioNamespace(inner.audio)
        if hasattr(inner, "responses"):
            self.responses = _TrackedResponsesNamespace(inner.responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
