"""
AI 配置模块 - 统一配置入口

配置加载策略：
1. 客户端配置（providers/llm/media/network/session）: 从 settings/ 加载默认预设
2. 工具提示词（prompts）: 优先从 InnerAgentWorkflow/ai_config/ 加载，回退到 settings/

配置结构：
- providers: API 提供商配置
- network: 网络和轮询配置
- session: 会话管理配置
- llm: LLM 模型配置
- media: 媒体工具配置
- prompts: 各类工具提示词（可被 InnerAgentWorkflow 覆盖）
"""

from __future__ import annotations

# 数据类
from config.prompts import (
    ToolPromptConfig,
    TextToolPromptConfig,
    DetectionPromptConfig,
    MediaToolPrompts,
    TextToolPrompts,
    SceneToolPrompts,
)

# ===========================================================================
# 客户端配置（仅从 settings/ 加载，不再从 InnerAgentWorkflow 导入）
# ===========================================================================

# from .settings import (
#     # 基础配置
#     # RUNTIME_SETTINGS,
#     # NETWORK_SETTINGS,
#     # POLLING_SETTINGS,
#     # SESSION_SETTINGS,
#     # 敏感配置
#     # PROVIDERS,
#     LLM_SETTINGS,
#     DEFAULT_SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
#     # 媒体配置
#     MEDIA_SETTINGS,
#     TTS_SETTINGS,
#     MUSIC_SETTINGS,
#     # 整合配置
#     AI_SETTINGS,
# )

# ===========================================================================
# 工具提示词（优先从 InnerAgentWorkflow 加载）
# ===========================================================================
#
# _USE_PRIVATE_PROMPTS = False
#
# try:
#     # 尝试从 InnerAgentWorkflow 加载提示词
#     from InnerAgentWorkflow.ai_config import (
#         # 系统提示词
#         DEFAULT_SYSTEM_PROMPT,
#         # 媒体工具提示词
#         IMAGE_PROMPTS,
#         VIDEO_PROMPTS,
#         SPEECH_PROMPTS,
#         MUSIC_PROMPTS,
#         OMNI_PROMPTS,
#         DETECTION_PROMPTS,
#         # 文本生成提示词
#         PRODUCT_TEXT_PROMPTS,
#         MARKETING_TEXT_PROMPTS,
#         CREATIVE_TEXT_PROMPTS,
#         TEXT_TOOL_PROMPTS,
#         PLATFORM_TIPS,
#         # 场景提示词
#         SCENE_QUERY_PROMPTS,
#         SCENE_TRANSFORM_PROMPTS,
#         SCENE_TOOL_PROMPTS,
#     )
#
#     _USE_PRIVATE_PROMPTS = True
#
# except ImportError:
#     # 回退到默认预设
#     DEFAULT_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT
#     from .settings import (
#         IMAGE_PROMPTS,
#         VIDEO_PROMPTS,
#         SPEECH_PROMPTS,
#         MUSIC_PROMPTS,
#         OMNI_PROMPTS,
#         DETECTION_PROMPTS,
#         # 文本生成配置
#         PRODUCT_TEXT_PROMPTS,
#         MARKETING_TEXT_PROMPTS,
#         CREATIVE_TEXT_PROMPTS,
#         TEXT_TOOL_PROMPTS,
#         PLATFORM_TIPS,
#         # 场景配置
#         SCENE_QUERY_PROMPTS,
#         SCENE_TRANSFORM_PROMPTS,
#         SCENE_TOOL_PROMPTS,
#     )

# 整合的提示词容器（始终从 settings 导入，因为 InnerAgentWorkflow 不再导出此整合类型）
# from .settings import MEDIA_TOOL_PROMPTS

#
# def has_private_config() -> bool:
#     """检查是否加载了私有提示词配置"""
#     return _USE_PRIVATE_PROMPTS


__all__ = [
    # 辅助函数
    # "has_private_config",
    # 整合配置
    # "AI_SETTINGS",
    # 基础配置
    # "RUNTIME_SETTINGS",
    # "NETWORK_SETTINGS",
    # "POLLING_SETTINGS",
    # "SESSION_SETTINGS",
    # 敏感配置
    # "PROVIDERS",
    # "LLM_SETTINGS",
    # "DEFAULT_SYSTEM_PROMPT",
    # 媒体配置
    # "MEDIA_SETTINGS",
    # "TTS_SETTINGS",
    # "MUSIC_SETTINGS",
    # "IMAGE_PROMPTS",
    # "VIDEO_PROMPTS",
    # "SPEECH_PROMPTS",
    # "MUSIC_PROMPTS",
    # "OMNI_PROMPTS",
    # "DETECTION_PROMPTS",
    # "MEDIA_TOOL_PROMPTS",
    # 文本生成配置
    # "PRODUCT_TEXT_PROMPTS",
    # "MARKETING_TEXT_PROMPTS",
    # "CREATIVE_TEXT_PROMPTS",
    # "TEXT_TOOL_PROMPTS",
    # "PLATFORM_TIPS",
    # 场景配置
    # "SCENE_QUERY_PROMPTS",
    # "SCENE_TRANSFORM_PROMPTS",
    # "SCENE_TOOL_PROMPTS",
    # 数据类
    "ToolPromptConfig",
    "TextToolPromptConfig",
    "DetectionPromptConfig",
    "MediaToolPrompts",
    "TextToolPrompts",
    "SceneToolPrompts",
]
