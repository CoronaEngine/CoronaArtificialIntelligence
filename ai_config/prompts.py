"""
工具提示词配置数据类
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ToolPromptConfig:
    """单个工具的提示词配置"""

    tool_description: str  # 工具描述
    fields: Dict[str, str] = field(default_factory=dict)  # 字段名 -> 字段描述


@dataclass(frozen=True)
class TextToolPromptConfig:
    """文本生成工具的提示词配置（包含模板）"""

    tool_description: str
    fields: Dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""  # 系统提示词模板
    user_prompt: str = ""  # 用户提示词模板


@dataclass(frozen=True)
class DetectionPromptConfig:
    """目标检测工具的提示词配置（包含检测提示词模板）"""

    tool_description: str
    fields: Dict[str, str] = field(default_factory=dict)
    detection_prompt: str = ""  # 检测提示词模板
    target_prefix: str = ""  # 目标描述前缀模板


@dataclass(frozen=True)
class TextToolPrompts:
    """文本生成工具提示词集合"""

    product: TextToolPromptConfig
    marketing: TextToolPromptConfig
    creative: TextToolPromptConfig
    platform_tips: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneToolPrompts:
    """场景工具提示词集合"""

    query: ToolPromptConfig
    transform: ToolPromptConfig
