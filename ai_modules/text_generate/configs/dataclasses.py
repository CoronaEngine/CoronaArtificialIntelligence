"""
LLM 模型配置
"""

from dataclasses import dataclass


@dataclass(frozen=False)
class ChatModelConfig:
    """聊天模型配置"""

    provider: str
    model: str
    temperature: float
    request_timeout: float
    system_prompt: str = ""  # 默认值由加载器设置


@dataclass(frozen=False)
class ToolModelConfig:
    """工具模型配置"""

    provider: str
    model: str
