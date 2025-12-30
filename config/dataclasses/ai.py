"""
AI 配置主类
"""

from dataclasses import dataclass
from typing import Dict

from config.dataclasses.llm import ProviderConfig, ChatModelConfig, ToolModelConfig
from .media import MediaConfig
from .network import NetworkConfig, PollingConfig, SessionConfig
from .speech import TTSConfig, MusicConfig


@dataclass(frozen=True)
class AIConfig:
    """AI 配置"""

    providers: Dict[str, ProviderConfig]
    chat: ChatModelConfig
    tool_models: Dict[str, ToolModelConfig]
    network: NetworkConfig
    polling: PollingConfig
    session: SessionConfig
    media: MediaConfig
    tts: TTSConfig
    music: MusicConfig
