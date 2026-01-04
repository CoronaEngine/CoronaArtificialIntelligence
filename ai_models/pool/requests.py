"""
标准请求模型

定义各类别媒体生成的统一请求格式。
公共项目定义，与 InnerAgentWorkflow 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MediaRequest:
    """所有媒体生成请求的基类"""

    session_id: str  # 会话 ID，用于资源关联


@dataclass
class ImageRequest(MediaRequest):
    """
    图像生成请求

    支持两种模式：
    1. 纯文生图：仅提供 prompt
    2. 图生图/编辑：提供 prompt + image_urls
    """

    prompt: str  # 生成提示词
    resolution: str = "1:1"  # 图片比例 (1:1, 3:2, 4:3, 16:9, 21:9, 5:4, 2:1 及竖屏版本)
    image_size: str = "2K"  # 输出分辨率档位 (1K, 2K, 4K)
    image_urls: List[str] = field(
        default_factory=list
    )  # 输入图片 URL 列表（图生图模式）


@dataclass
class VideoRequest(MediaRequest):
    """
    视频生成请求（图生视频）

    基于起始帧图片和提示词生成动态视频。
    """

    prompt: str  # 视频生成提示词
    image_url: str  # 起始帧图片 URL
    resolution: str = "720P"  # 输出分辨率 (480P/720P/1080P)
    prompt_extend: bool = True  # 是否启用提示词扩展


@dataclass
class MusicRequest(MediaRequest):
    """
    音乐/BGM 生成请求

    根据文本描述生成背景音乐。
    """

    prompt: str  # 音乐描述/氛围提示词
    style: Optional[str] = None  # 风格标签 (lofi, ambient, epic, ...)
    duration: int = 20  # 期望时长（秒）
    model: str = "V5"  # 模型版本


@dataclass
class SpeechRequest(MediaRequest):
    """
    语音合成 (TTS) 请求

    将文本转换为语音。
    """

    text: str  # 待合成文本
    voice_type: str = "zh_female_cancan_mars_bigtts"  # 音色类型
    speed_ratio: float = 1.0  # 语速比例 [0.1, 2.0]
    loudness_ratio: float = 1.0  # 音量比例 [0.5, 2.0]
    encoding: str = "mp3"  # 输出格式 (mp3/wav/ogg_opus/pcm)
    sample_rate: int = 24000  # 采样率


@dataclass
class ChatRequest(MediaRequest):
    """
    对话/文本生成请求

    调用 LLM 进行对话或文本生成。
    """

    messages: List[Dict[str, str]]  # 消息列表 [{"role": "user", "content": "..."}]
    temperature: float = 0.7  # 温度参数
    max_tokens: Optional[int] = None  # 最大生成 token 数


@dataclass
class OmniRequest(MediaRequest):
    """
    多模态理解请求

    使用 VLM 分析图片、视频、音频内容。
    """

    prompt: str  # 分析提示词
    image_urls: List[str] = field(default_factory=list)  # 图片 URL 列表
    video_urls: List[str] = field(default_factory=list)  # 视频 URL 列表
    audio_urls: List[str] = field(default_factory=list)  # 音频 URL 列表


@dataclass
class DetectionRequest(MediaRequest):
    """
    目标检测请求

    使用 VLM 进行图像目标检测，返回边界框和描述。
    """

    image_url: str  # 待检测图片 URL
    target_description: str = ""  # 可选的目标描述


__all__ = [
    "MediaRequest",
    "ImageRequest",
    "VideoRequest",
    "MusicRequest",
    "SpeechRequest",
    "ChatRequest",
    "OmniRequest",
    "DetectionRequest",
]
