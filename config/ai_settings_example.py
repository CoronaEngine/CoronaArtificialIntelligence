"""
AI 配置模块 - 统一配置入口

配置结构：
- base: 运行时配置
- providers: API 提供商配置
- network: 网络和轮询配置
- session: 会话管理配置
- llm: LLM 模型配置和系统提示词
- media: 媒体工具配置和提示词
- text: 文本生成工具提示词
- scene: 场景操作工具提示词
"""

from __future__ import annotations

# 数据类
from .dataclasses.prompts import (
    ToolPromptConfig,
    TextToolPromptConfig,
    DetectionPromptConfig,
    MediaToolPrompts,
    TextToolPrompts,
    SceneToolPrompts,
)

# ===========================================================================
# 配置来源选择（一次性导入）
# ===========================================================================

_USE_PRIVATE_CONFIG = False

try:
    # 尝试从私有配置加载
    from ... import (
        # 基础配置
        RUNTIME_SETTINGS,
        NETWORK_SETTINGS,
        POLLING_SETTINGS,
        SESSION_SETTINGS,
        # 敏感配置
        PROVIDERS,
        LLM_SETTINGS,
        DEFAULT_SYSTEM_PROMPT,
        # 媒体配置
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
        # 文本生成配置
        PRODUCT_TEXT_PROMPTS,
        MARKETING_TEXT_PROMPTS,
        CREATIVE_TEXT_PROMPTS,
        TEXT_TOOL_PROMPTS,
        PLATFORM_TIPS,
        # 场景配置
        SCENE_QUERY_PROMPTS,
        SCENE_TRANSFORM_PROMPTS,
        SCENE_TOOL_PROMPTS,
        # 整合配置
        AI_SETTINGS,
    )

    _USE_PRIVATE_CONFIG = True

except ImportError:
    # 回退到默认预设
    from .settings import (
        # 基础配置
        RUNTIME_SETTINGS,
        NETWORK_SETTINGS,
        POLLING_SETTINGS,
        SESSION_SETTINGS,
        # 敏感配置
        PROVIDERS,
        LLM_SETTINGS,
        DEFAULT_SYSTEM_PROMPT,
        # 媒体配置
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
        # 文本生成配置
        PRODUCT_TEXT_PROMPTS,
        MARKETING_TEXT_PROMPTS,
        CREATIVE_TEXT_PROMPTS,
        TEXT_TOOL_PROMPTS,
        PLATFORM_TIPS,
        # 场景配置
        SCENE_QUERY_PROMPTS,
        SCENE_TRANSFORM_PROMPTS,
        SCENE_TOOL_PROMPTS,
        # 整合配置
        AI_SETTINGS,
    )


def has_private_config() -> bool:
    """检查是否加载了私有配置"""
    return _USE_PRIVATE_CONFIG


__all__ = [
    # 辅助函数
    "has_private_config",
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
    # 数据类
    "ToolPromptConfig",
    "TextToolPromptConfig",
    "DetectionPromptConfig",
    "MediaToolPrompts",
    "TextToolPrompts",
    "SceneToolPrompts",
]
