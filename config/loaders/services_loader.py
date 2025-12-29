"""
外部服务配置加载器
"""

import os
from typing import Any, Mapping

from config.dataclasses.speech import (
    TTSConfig,
    MusicConfig,
)


def _load_tts_config(raw: Mapping[str, Any] | None) -> TTSConfig:
    """加载 TTS 配置"""
    if not isinstance(raw, Mapping):
        return TTSConfig()

    appid = raw.get("appid")
    appid_env = raw.get("appid_env")
    if appid_env:
        appid = os.getenv(str(appid_env), appid)

    token = raw.get("token")
    token_env = raw.get("token_env")
    if token_env:
        token = os.getenv(str(token_env), token)

    return TTSConfig(
        appid=appid,
        token=token,
    )


def _load_music_config(raw: Mapping[str, Any] | None) -> MusicConfig:
    """加载音乐生成配置"""
    if not isinstance(raw, Mapping):
        return MusicConfig()

    api_key = raw.get("api_key")
    api_key_env = raw.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)

    base_url = raw.get("base_url")

    return MusicConfig(
        api_key=api_key,
        base_url=base_url,
    )
