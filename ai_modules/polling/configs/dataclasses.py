"""
网络和轮询配置
"""

from dataclasses import dataclass

@dataclass(frozen=False)
class PollingConfig:
    """异步任务轮询配置"""

    max_wait_seconds: int = 150
    default_interval: float = 3.0
    speech_interval: float = 2.0
    music_interval: float = 5.0
    video_interval: float = 3.0

