"""
视频生成配置
"""

from dataclasses import dataclass


@dataclass(frozen=False)
class VideoToolConfig:
    """视频生成工具配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
