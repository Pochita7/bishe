"""
配置文件 - 模型、路径、API 设置
适配 Microsoft Agent Framework (agent-framework) Python SDK
"""
import os

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
# LLM 配置 (火山引擎)
# ========================
# 共同 API Key
API_KEY = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY", "d9452cdf-f0ec-41f7-9029-115170830afc")
# 火山引擎 API 地址
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

# 文本模型 DeepSeek-V3.2 接入点 ID
TEXT_MODEL = os.environ.get("GAIA_MODEL", "ep-20260212204648-nrlx2")

# 视觉模型 Doubao-Seed-1.8 接入点 ID
VISION_MODEL = os.environ.get("GAIA_VISION_MODEL", "ep-20260212191314-5hbv4")

# 向后兼容
MODEL_NAME = TEXT_MODEL


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
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def get_openai_client():
    """获取原生 OpenAI 兼容客户端（用于 Vision、Whisper 等直接调用）"""
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=BASE_URL)
