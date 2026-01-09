"""
图像生成配置
"""

from dataclasses import dataclass


@dataclass(frozen=False)
class ImageConstraintsConfig:
    """图像尺寸约束配置"""

    max_size: int = 2000
    min_size: int = 360


@dataclass(frozen=False)
class ImageToolConfig:
    """图像生成工具配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
