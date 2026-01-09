"""
多模态理解模型配置
"""

from dataclasses import dataclass


@dataclass(frozen=False)
class OmniModelConfig:
    """多模态理解模型配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    max_frames: int = 16  # 视频最大帧数
    fps: float = 1.0  # 视频帧率
    image_detail: str = "high"  # 图片细节级别: low/high/auto
    request_timeout: float = 150.0  # 请求超时时间（秒）
