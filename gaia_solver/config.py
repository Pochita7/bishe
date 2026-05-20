"""
配置文件 - 模型、路径、API 设置
适配 Microsoft Agent Framework (agent-framework) Python SDK
"""
import os
from typing import Any

from token_accounting import record_openai_response_usage


def _load_dotenv_file(path: str) -> None:
    """Load KEY=VALUE pairs without requiring python-dotenv."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if value.startswith("${") and value.endswith("}"):
                    value = os.environ.get(value[2:-1], value)
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def _has_any_env(*names: str) -> bool:
    return any(bool(os.environ.get(name, "").strip()) for name in names)


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _looks_like_ark_model(model: str) -> bool:
    model_l = (model or "").lower()
    return model_l.startswith("ep-") or "doubao" in model_l or "ark" in model_l

# ========================
# 路径配置
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAIA_DATA_DIR = os.path.join(BASE_DIR, "gaia_data")
GAIA_LEVEL1_PATH = os.path.join(GAIA_DATA_DIR, "gaia_level1.jsonl")
GAIA_ATTACHMENTS_DIR = os.path.join(GAIA_DATA_DIR, "attachments")
WORK_DIR = os.path.join(BASE_DIR, "gaia_work_dir")

for _env_file in (".env", ".env.local"):
    _load_dotenv_file(os.path.join(BASE_DIR, _env_file))

# 确保工作目录存在
os.makedirs(WORK_DIR, exist_ok=True)

# ========================
# LLM 配置 (DeepSeek 官方 OpenAI-compatible API)
# ========================
ARK_BASE_URL_DEFAULT = "https://ark.cn-beijing.volces.com/api/v3"
LEGACY_ARK_TEXT_MODEL = "ep-20260212204648-nrlx2"

# 文本主路径 API Key。不要把真实密钥写进仓库，优先通过环境变量提供。
API_KEY = _env("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ARK_API_KEY")
# DeepSeek 官方 API 地址
BASE_URL = _env("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL", "ARK_BASE_URL", "VOLCENGINE_BASE_URL", default="")
if not BASE_URL:
    if _has_any_env("ARK_API_KEY", "VOLCENGINE_API_KEY") and not _has_any_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        BASE_URL = ARK_BASE_URL_DEFAULT
    else:
        BASE_URL = "https://api.deepseek.com"

# 文本模型，默认使用 DeepSeek V4 Flash
TEXT_MODEL = _env("DEEPSEEK_MODEL", "OPENAI_MODEL", "GAIA_MODEL", "ARK_MODEL", default="")
if not TEXT_MODEL:
    if BASE_URL == ARK_BASE_URL_DEFAULT and not _has_any_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        TEXT_MODEL = LEGACY_ARK_TEXT_MODEL
    else:
        TEXT_MODEL = "deepseek-v4-flash"
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high").strip().lower()

# 多模态/音频任务可单独配置兼容供应商；默认跟随文本配置
VISION_API_KEY = _env(
    "GAIA_VISION_API_KEY",
    "DOUBAO_API_KEY",
    "VOLCENGINE_API_KEY",
    "ARK_API_KEY",
    default=API_KEY,
)
VISION_MODEL = _env(
    "GAIA_VISION_MODEL",
    "DOUBAO_VISION_MODEL",
    "ARK_VISION_MODEL",
    "VOLCENGINE_VISION_MODEL",
    "DOUBAO_MODEL",
    "GAIA_MODEL",
    "ARK_MODEL",
    default="",
)
VISION_BASE_URL = _env(
    "GAIA_VISION_BASE_URL",
    "DOUBAO_BASE_URL",
    "ARK_BASE_URL",
    "VOLCENGINE_BASE_URL",
    default="",
)
if not VISION_BASE_URL and (
    _has_any_env("GAIA_VISION_MODEL", "DOUBAO_VISION_MODEL", "ARK_VISION_MODEL", "VOLCENGINE_VISION_MODEL", "DOUBAO_MODEL", "GAIA_MODEL", "ARK_MODEL", "DOUBAO_API_KEY", "VOLCENGINE_API_KEY", "ARK_API_KEY")
    or _looks_like_ark_model(VISION_MODEL)
):
    VISION_BASE_URL = ARK_BASE_URL_DEFAULT
if not VISION_BASE_URL:
    VISION_BASE_URL = BASE_URL
if not VISION_MODEL:
    VISION_MODEL = LEGACY_ARK_TEXT_MODEL if VISION_BASE_URL == ARK_BASE_URL_DEFAULT else TEXT_MODEL

AUDIO_API_KEY = _env(
    "GAIA_AUDIO_API_KEY",
    "DOUBAO_AUDIO_API_KEY",
    "GAIA_VISION_API_KEY",
    "DOUBAO_API_KEY",
    "VOLCENGINE_API_KEY",
    "ARK_API_KEY",
    default=VISION_API_KEY,
)
AUDIO_BASE_URL = _env(
    "GAIA_AUDIO_BASE_URL",
    "DOUBAO_AUDIO_BASE_URL",
    "GAIA_VISION_BASE_URL",
    "DOUBAO_BASE_URL",
    "ARK_BASE_URL",
    "VOLCENGINE_BASE_URL",
    default=VISION_BASE_URL,
)
AUDIO_MODEL = _env("GAIA_AUDIO_MODEL", "DOUBAO_AUDIO_MODEL", "ARK_AUDIO_MODEL", default=VISION_MODEL)
AUDIO_INPUT_MODE = _env(
    "GAIA_AUDIO_INPUT_MODE",
    "DOUBAO_AUDIO_INPUT_MODE",
    default="chat" if _looks_like_ark_model(AUDIO_MODEL) else "transcription",
).lower()

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

    return _TrackedOpenAIClient(
        OpenAI(
            api_key=VISION_API_KEY,
            base_url=VISION_BASE_URL,
            max_retries=_env_int("GAIA_VISION_MAX_RETRIES", 0, 0, 5),
        )
    )


def get_audio_client():
    """获取原生 OpenAI 兼容客户端（用于音频转写）。"""
    from openai import OpenAI

    return _TrackedOpenAIClient(
        OpenAI(
            api_key=AUDIO_API_KEY,
            base_url=AUDIO_BASE_URL,
            max_retries=_env_int("GAIA_AUDIO_MAX_RETRIES", 0, 0, 5),
        )
    )


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
