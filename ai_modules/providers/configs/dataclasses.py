"""
LLM 模型配置
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ProviderConfig:
    """AI 服务提供商配置"""

    name: str
    type: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    headers: Dict[str, str] = field(default_factory=dict)
