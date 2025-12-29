"""
语音和音乐服务配置
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TTSConfig:
    """TTS 凭证配置"""

    appid: str | None = None
    token: str | None = None


@dataclass(frozen=True)
class MusicConfig:
    """音乐生成配置"""

    api_key: str | None = None
    base_url: str | None = None


@dataclass
class SpeechAppConfig:
    """火山引擎语音合成应用配置"""

    appid: str
    token: str
    cluster: str = "volcano_tts"
    uid: str = "default_user"


@dataclass
class SpeechAudioConfig:
    """火山引擎语音合成音频配置"""

    voice_type: str  # 音色类型
    encoding: str = "mp3"  # 音频编码格式: wav/pcm/ogg_opus/mp3
    speed_ratio: float = 1.0  # 语速 [0.1, 2]
    rate: int = 24000  # 采样率: 8000/16000/24000
    bitrate: int = 160  # 比特率 kb/s
    emotion: Optional[str] = None  # 音色情感
    enable_emotion: bool = False  # 是否启用情感
    emotion_scale: Optional[float] = None  # 情绪值 [1, 5]
    loudness_ratio: float = 1.0  # 音量调节 [0.5, 2]
    explicit_language: Optional[str] = None  # 明确语种
    context_language: Optional[str] = None  # 参考语种


@dataclass
class SpeechRequestConfig:
    """火山引擎语音合成请求配置（已废弃，保留用于兼容性）"""

    reqid: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    text_type: str = "plain"
    operation: str = "query"
    model: Optional[str] = None
    silence_duration: Optional[float] = None
    with_timestamp: Optional[int] = None
    extra_param: Optional[Dict[str, Any]] = None
