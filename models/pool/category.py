"""
媒体类别定义

公共项目定义，与 InnerAgentWorkflow 保持一致。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class MediaCategory(Enum):
    """媒体生成类别"""

    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"
    SPEECH = "speech"
    AGENT = "agent"      # Agent 主推理用 LLM
    TEXT = "text"        # 文案工具用 LLM
    OMNI = "omni"        # 多模态理解 VLM
    DETECTION = "detection"  # 目标检测 VLM


# 类别 -> content_type 映射
CATEGORY_CONTENT_TYPE: Dict[MediaCategory, str] = {
    MediaCategory.IMAGE: "image",
    MediaCategory.VIDEO: "video",
    MediaCategory.MUSIC: "audio",
    MediaCategory.SPEECH: "audio",
    MediaCategory.AGENT: "text",
    MediaCategory.TEXT: "text",
    MediaCategory.OMNI: "text",       # VLM 返回文本分析
    MediaCategory.DETECTION: "detection",  # 目标检测返回检测结果
}


def get_content_type(category: MediaCategory) -> str:
    """获取类别对应的 content_type"""
    return CATEGORY_CONTENT_TYPE.get(category, "text")


__all__ = [
    "MediaCategory",
    "CATEGORY_CONTENT_TYPE",
    "get_content_type",
]
