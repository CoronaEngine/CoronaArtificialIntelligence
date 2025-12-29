"""
AI 专属配置
处理 LLM、图像生成、视频生成等 AI 相关配置

本模块是配置的入口点，实际实现分布在子模块中：
- dataclasses/: 所有配置数据类
- loaders/: 配置加载函数
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Any, Dict, Optional

try:
    from .ai_settings import AI_SETTINGS, DEFAULT_SYSTEM_PROMPT
except ImportError:
    AI_SETTINGS = {}
    DEFAULT_SYSTEM_PROMPT = ""

# ---------------------------------------------------------------------------
# 从子模块导入数据类
# ---------------------------------------------------------------------------

from .dataclasses import (
    # Provider
    ProviderConfig,
    # LLM
    ChatModelConfig,
    ToolModelConfig,
    # Media
    AudioConfig,
    ImageConstraintsConfig,
    MediaToolConfig,
    OmniModelConfig,
    DetectionModelConfig,
    MediaConfig,
    # Network
    NetworkConfig,
    PollingConfig,
    # Session
    SessionConfig,
    # Speech
    SpeechAppConfig,
    SpeechAudioConfig,
    SpeechRequestConfig,
    # Services
    TTSConfig,
    MusicConfig,
    # AI
    AIConfig,
)

# ---------------------------------------------------------------------------
# 从子模块导入加载函数
# ---------------------------------------------------------------------------

from .loaders import (
    _as_float,
    _load_providers,
    _load_tool_models,
    _load_media_config,
    _load_network_config,
    _load_polling_config,
    _load_session_config,
    _load_tts_config,
    _load_music_config,
)

# ---------------------------------------------------------------------------
# 模块级缓存
# ---------------------------------------------------------------------------

_AI_CACHE: Optional[AIConfig] = None
_AI_CONFIG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 配置加载辅助函数
# ---------------------------------------------------------------------------


def _apply_env_overrides(data: Dict[str, Any]) -> None:
    """应用环境变量覆盖"""
    overrides = {
        ("llm", "chat", "model"): os.getenv("CORONA_LLM_MODEL"),
        ("llm", "chat", "provider"): os.getenv("CORONA_LLM_PROVIDER"),
    }
    for path, value in overrides.items():
        if value is None:
            continue
        section = data
        for part in path[:-1]:
            section = section.setdefault(part, {})
        key = path[-1]
        section[key] = value


def _load_ai_config_data() -> Dict[str, Any]:
    """从 ai_settings 模块加载配置"""
    data = copy.deepcopy(AI_SETTINGS)
    _apply_env_overrides(data)
    return data


# ---------------------------------------------------------------------------
# 公共函数
# ---------------------------------------------------------------------------


def _build_ai_config() -> AIConfig:
    """构建 AI 配置"""
    raw = _load_ai_config_data()

    providers = _load_providers(raw.get("providers"))
    if not providers:
        raise RuntimeError("AI 配置中至少需要声明一个 provider")

    llm_section = raw.get("llm", {})
    chat_section = llm_section.get("chat", llm_section)
    chat = ChatModelConfig(
        provider=str(chat_section.get("provider", next(iter(providers.keys())))),
        model=str(chat_section.get("model", "Qwen/Qwen2.5-7B-Instruct")),
        temperature=_as_float(chat_section.get("temperature", 0.2), 0.2),
        request_timeout=_as_float(chat_section.get("request_timeout", 60), 60.0),
        system_prompt=str(chat_section.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
    )

    tool_models = _load_tool_models(llm_section.get("tool_models", {}))
    network = _load_network_config(raw.get("network"))
    polling = _load_polling_config(raw.get("polling"))
    session = _load_session_config(raw.get("session"))
    media = _load_media_config(raw.get("media", {}))
    tts = _load_tts_config(raw.get("tts"))
    music = _load_music_config(raw.get("music"))

    return AIConfig(
        providers=providers,
        chat=chat,
        tool_models=tool_models,
        network=network,
        polling=polling,
        session=session,
        media=media,
        tts=tts,
        music=music,
    )


def get_ai_config() -> AIConfig:
    """获取 AI 配置（单例，线程安全）"""
    global _AI_CACHE
    if _AI_CACHE is None:
        with _AI_CONFIG_LOCK:
            if _AI_CACHE is None:
                _AI_CACHE = _build_ai_config()
    return _AI_CACHE


def reload_ai_config() -> AIConfig:
    """重新加载 AI 配置（线程安全）"""
    global _AI_CACHE
    with _AI_CONFIG_LOCK:
        _AI_CACHE = _build_ai_config()
    return _AI_CACHE


__all__ = [
    # 数据类
    "AIConfig",
    "ProviderConfig",
    "ChatModelConfig",
    "ToolModelConfig",
    "MediaConfig",
    "MediaToolConfig",
    "OmniModelConfig",
    "DetectionModelConfig",
    "NetworkConfig",
    "PollingConfig",
    "AudioConfig",
    "ImageConstraintsConfig",
    "SessionConfig",
    "TTSConfig",
    "MusicConfig",
    "SpeechAppConfig",
    "SpeechAudioConfig",
    "SpeechRequestConfig",
    # 公共函数
    "get_ai_config",
    "reload_ai_config",
]
