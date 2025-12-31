"""
媒体配置加载器
"""

from typing import Any, Mapping

from ..configs.dataclasses import (
    AudioConfig,
    ImageConstraintsConfig,
    MediaToolConfig,
    OmniModelConfig,
    DetectionModelConfig,
    MediaConfig,
)
from service.entrance import ai_entrance
from tools.helpers import _as_bool

@ai_entrance.collector.register_loader('media')
def _load_media_config(raw: Mapping[str, Any]) -> MediaConfig:
    """加载媒体配置"""

    def _load_media_tool(section: Mapping[str, Any] | None) -> MediaToolConfig:
        if not isinstance(section, Mapping):
            return MediaToolConfig()
        return MediaToolConfig(
            enable=_as_bool(section.get("enable"), False),
            provider=section.get("provider"),
            model=section.get("model"),
            base_url=section.get("base_url"),
        )

    audio_section = raw.get("audio", {})
    audio = (
        AudioConfig(
            sample_rate=int(audio_section.get("sample_rate", 24000)),
            bitrate=int(audio_section.get("bitrate", 160)),
        )
        if isinstance(audio_section, Mapping)
        else AudioConfig()
    )

    image = _load_media_tool(raw.get("image"))

    image_section = raw.get("image", {})
    image_constraints = (
        ImageConstraintsConfig(
            max_size=int(image_section.get("max_size", 2000)),
            min_size=int(image_section.get("min_size", 360)),
        )
        if isinstance(image_section, Mapping)
        else ImageConstraintsConfig()
    )

    video = _load_media_tool(raw.get("video"))

    # 加载 omni 配置
    omni_section = raw.get("omni", {})
    omni = (
        OmniModelConfig(
            enable=_as_bool(omni_section.get("enable"), False),
            provider=omni_section.get("provider"),
            model=omni_section.get("model"),
            max_frames=int(omni_section.get("max_frames", 16)),
            fps=float(omni_section.get("fps", 1.0)),
            image_detail=str(omni_section.get("image_detail", "high")),
            request_timeout=float(omni_section.get("request_timeout", 120.0)),
        )
        if isinstance(omni_section, Mapping)
        else OmniModelConfig()
    )

    # 加载 detection 配置
    detection_section = raw.get("detection", {})
    detection = (
        DetectionModelConfig(
            enable=_as_bool(detection_section.get("enable"), False),
            provider=detection_section.get("provider"),
            model=detection_section.get("model"),
            image_detail=str(detection_section.get("image_detail", "high")),
            request_timeout=float(detection_section.get("request_timeout", 120.0)),
        )
        if isinstance(detection_section, Mapping)
        else DetectionModelConfig()
    )

    return MediaConfig(
        audio=audio,
        image=image,
        image_constraints=image_constraints,
        video=video,
        omni=omni,
        detection=detection,
    )
