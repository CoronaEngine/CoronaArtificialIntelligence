"""
语音合成 (TTS) 配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict

from config.dataclasses.prompts import ToolPromptConfig

# ===========================================================================
# TTS 配置 - 默认预设
# 实际的 appid 和 token 应放在 InnerAgentWorkflow/ai_config/media/base.py 中
# ===========================================================================

TTS_SETTINGS: Dict[str, Any] = {
    # 火山引擎 TTS 配置
    # 获取方式: https://www.volcengine.com/docs/6561/196768
    "appid": "YOUR_APPID_HERE",
    "token": "YOUR_TOKEN_HERE",
}

# ===========================================================================
# 语音合成提示词
# ===========================================================================

SPEECH_PROMPTS = ToolPromptConfig(
    tool_description="使用火山引擎TTS将文本转换为语音。异步提交任务并返回音频URL，支持多种音色、语速、音量和格式调整。",
    fields={
        "text": "待合成的文本内容",
        "voice_type": "音色类型，例如：zh_female_cancan_mars_bigtts（女声）、zh_male_M392_conversation_wvae_bigtts（男声）",
        "speed_ratio": "语速比例，范围 [0.1, 2.0]，1.0 为正常速度",
        "loudness_ratio": "音量比例，范围 [0.5, 2.0]，1.0 为正常音量",
        "encoding": "音频格式，可选：mp3、wav、ogg_opus、pcm",
        "rate": "采样率（从CONFIG读取）",
        "max_wait_seconds": "最大等待时间（从CONFIG读取）",
        "poll_interval": "轮询间隔（从CONFIG读取）",
    },
)
