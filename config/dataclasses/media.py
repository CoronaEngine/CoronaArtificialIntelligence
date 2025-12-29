"""
媒体相关配置
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    """音频配置"""

    sample_rate: int = 24000
    bitrate: int = 160


@dataclass(frozen=True)
class ImageConstraintsConfig:
    """图像尺寸约束配置"""

    max_size: int = 2000
    min_size: int = 360


@dataclass(frozen=True)
class MediaToolConfig:
    """媒体工具配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class OmniModelConfig:
    """多模态理解模型配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    max_frames: int = 16  # 视频最大帧数
    fps: float = 1.0  # 视频帧率
    image_detail: str = "high"  # 图片细节级别: low/high/auto
    request_timeout: float = 150.0  # 请求超时时间（秒）


@dataclass(frozen=True)
class DetectionModelConfig:
    """目标检测模型配置"""

    enable: bool = False
    provider: str | None = None
    model: str | None = None
    image_detail: str = "high"  # 图片细节级别: low/high/auto
    request_timeout: float = 150.0  # 请求超时时间（秒）


@dataclass(frozen=True)
class MediaConfig:
    """媒体配置（图像/视频/音频）"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    image: MediaToolConfig = field(default_factory=MediaToolConfig)
    image_constraints: ImageConstraintsConfig = field(
        default_factory=ImageConstraintsConfig
    )
    video: MediaToolConfig = field(default_factory=MediaToolConfig)
    omni: OmniModelConfig = field(default_factory=OmniModelConfig)
    detection: DetectionModelConfig = field(default_factory=DetectionModelConfig)
