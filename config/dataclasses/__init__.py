"""
AI 配置数据类
"""

from .llm import ProviderConfig, ChatModelConfig, ToolModelConfig
from .media import (
    AudioConfig,
    ImageConstraintsConfig,
    MediaToolConfig,
    OmniModelConfig,
    DetectionModelConfig,
    MediaConfig,
)
from .network import NetworkConfig, PollingConfig, SessionConfig
from .speech import (
    TTSConfig,
    MusicConfig,
    SpeechAppConfig,
    SpeechAudioConfig,
    SpeechRequestConfig,
)
from .prompts import (
    ToolPromptConfig,
    TextToolPromptConfig,
    DetectionPromptConfig,
    MediaToolPrompts,
    TextToolPrompts,
    SceneToolPrompts,
)
from .ai import AIConfig

__all__ = [
    # LLM
    "ProviderConfig",
    "ChatModelConfig",
    "ToolModelConfig",
    # Media
    "AudioConfig",
    "ImageConstraintsConfig",
    "MediaToolConfig",
    "OmniModelConfig",
    "DetectionModelConfig",
    "MediaConfig",
    # Network
    "NetworkConfig",
    "PollingConfig",
    "SessionConfig",
    # Speech
    "TTSConfig",
    "MusicConfig",
    "SpeechAppConfig",
    "SpeechAudioConfig",
    "SpeechRequestConfig",
    # Prompts
    "ToolPromptConfig",
    "TextToolPromptConfig",
    "DetectionPromptConfig",
    "MediaToolPrompts",
    "TextToolPrompts",
    "SceneToolPrompts",
    # AI
    "AIConfig",
]
