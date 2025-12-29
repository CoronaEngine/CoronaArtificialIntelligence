"""
媒体工具配置模块

整合所有媒体生成相关的配置和提示词：
- image: 图像生成
- video: 视频生成
- speech: 语音合成 (TTS)
- music: 音乐生成
- omni: 多模态理解
- detection: 目标检测（默认占位配置）
"""

from __future__ import annotations

from typing import Any, Dict

# 子模块导入
from .image import IMAGE_SETTINGS, IMAGE_PROMPTS
from .video import VIDEO_SETTINGS, VIDEO_PROMPTS
from .speech import TTS_SETTINGS, SPEECH_PROMPTS
from .music import MUSIC_SETTINGS, MUSIC_PROMPTS
from .omni import OMNI_SETTINGS, OMNI_PROMPTS

# 数据类导入
from ...dataclasses.prompts import MediaToolPrompts, DetectionPromptConfig

# ===========================================================================
# 目标检测默认提示词（占位配置，实际配置在 InnerAgentWorkflow/ai_config/media/detection.py）
# ===========================================================================

DETECTION_PROMPTS = DetectionPromptConfig(
    tool_description="使用视觉语言模型进行图像目标检测。",
    fields={
        "image_url": "要检测的图片 URL",
        "target_description": "可选：要检测的目标类型描述",
    },
    detection_prompt=(
        "请检测图像中所有显著的主体对象，对于每个检测到的对象，"
        "请提供其类别和边界框，格式为：<bbox>x_min y_min x_max y_max</bbox>，"
        "坐标范围 0-1000。"
    ),
    target_prefix="请重点检测图片中的「{target_description}」。\n\n",
)

# ===========================================================================
# 整合的媒体配置
# ===========================================================================

MEDIA_SETTINGS: Dict[str, Any] = {
    "audio": {
        "sample_rate": 24000,
        "bitrate": 160,
    },
    "image": IMAGE_SETTINGS,
    "video": VIDEO_SETTINGS,
    "omni": OMNI_SETTINGS,
    # detection 配置已迁移到 InnerAgentWorkflow/ai_config/media/detection.py
}

# ===========================================================================
# 整合的媒体工具提示词
# ===========================================================================

MEDIA_TOOL_PROMPTS = MediaToolPrompts(
    image=IMAGE_PROMPTS,
    video=VIDEO_PROMPTS,
    speech=SPEECH_PROMPTS,
    music=MUSIC_PROMPTS,
    omni=OMNI_PROMPTS,
    detection=DETECTION_PROMPTS,
)

# ===========================================================================
# 导出
# ===========================================================================

__all__ = [
    # 配置
    "MEDIA_SETTINGS",
    "TTS_SETTINGS",
    "MUSIC_SETTINGS",
    # 单独的设置（供内部使用）
    "IMAGE_SETTINGS",
    "VIDEO_SETTINGS",
    "OMNI_SETTINGS",
    # 提示词
    "IMAGE_PROMPTS",
    "VIDEO_PROMPTS",
    "SPEECH_PROMPTS",
    "MUSIC_PROMPTS",
    "OMNI_PROMPTS",
    "DETECTION_PROMPTS",
    "MEDIA_TOOL_PROMPTS",
]
