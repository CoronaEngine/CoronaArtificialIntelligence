"""
AI 配置模块 - 默认预设

此模块包含默认配置预设，按功能模块组织：
- base: 运行时基础配置
- providers: API 提供商配置（示例）
- network: 网络和轮询配置
- session: 会话管理配置
- llm: LLM 模型配置和系统提示词（示例）
- media: 媒体工具配置和提示词（示例）
- text: 文本生成工具提示词
- scene: 场景操作工具提示词
"""

from __future__ import annotations

from typing import Any, Dict

# ===========================================================================
# 默认预设配置
# ===========================================================================

from .base import RUNTIME_SETTINGS
from .network import NETWORK_SETTINGS, POLLING_SETTINGS
from .session import SESSION_SETTINGS
from .providers import PROVIDERS
from .llm import LLM_SETTINGS, DEFAULT_SYSTEM_PROMPT
from .media import (
    MEDIA_SETTINGS,
    TTS_SETTINGS,
    MUSIC_SETTINGS,
    IMAGE_PROMPTS,
    VIDEO_PROMPTS,
    SPEECH_PROMPTS,
    MUSIC_PROMPTS,
    OMNI_PROMPTS,
    DETECTION_PROMPTS,
    MEDIA_TOOL_PROMPTS,
)
from .text import (
    PRODUCT_TEXT_PROMPTS,
    MARKETING_TEXT_PROMPTS,
    CREATIVE_TEXT_PROMPTS,
    TEXT_TOOL_PROMPTS,
    PLATFORM_TIPS,
)
from .scene import (
    SCENE_QUERY_PROMPTS,
    SCENE_TRANSFORM_PROMPTS,
    SCENE_TOOL_PROMPTS,
)

# ===========================================================================
# 整合配置字典
# ===========================================================================

AI_SETTINGS: Dict[str, Any] = {
    "runtime": RUNTIME_SETTINGS,
    "network": NETWORK_SETTINGS,
    "polling": POLLING_SETTINGS,
    "session": SESSION_SETTINGS,
    "providers": PROVIDERS,
    "llm": LLM_SETTINGS,
    "media": MEDIA_SETTINGS,
    "tts": TTS_SETTINGS,
    "music": MUSIC_SETTINGS,
}

# ===========================================================================
# 导出
# ===========================================================================

__all__ = [
    # 整合配置
    "AI_SETTINGS",
    # 基础配置
    "RUNTIME_SETTINGS",
    "NETWORK_SETTINGS",
    "POLLING_SETTINGS",
    "SESSION_SETTINGS",
    # 敏感配置
    "PROVIDERS",
    "LLM_SETTINGS",
    "DEFAULT_SYSTEM_PROMPT",
    # 媒体配置
    "MEDIA_SETTINGS",
    "TTS_SETTINGS",
    "MUSIC_SETTINGS",
    "IMAGE_PROMPTS",
    "VIDEO_PROMPTS",
    "SPEECH_PROMPTS",
    "MUSIC_PROMPTS",
    "OMNI_PROMPTS",
    "DETECTION_PROMPTS",
    "MEDIA_TOOL_PROMPTS",
    # 文本生成配置
    "PRODUCT_TEXT_PROMPTS",
    "MARKETING_TEXT_PROMPTS",
    "CREATIVE_TEXT_PROMPTS",
    "TEXT_TOOL_PROMPTS",
    "PLATFORM_TIPS",
    # 场景配置
    "SCENE_QUERY_PROMPTS",
    "SCENE_TRANSFORM_PROMPTS",
    "SCENE_TOOL_PROMPTS",
]
