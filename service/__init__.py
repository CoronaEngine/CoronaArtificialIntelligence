"""
AI Service 子包

提供各类 AI 服务的入口函数。
为避免循环导入，这些函数通过延迟导入方式暴露。
"""


def __getattr__(name: str):
    """延迟导入以避免循环依赖"""
    if name == "handle_integrated_entrance":
        from service.integrated import (
            handle_integrated_entrance,
        )

        return handle_integrated_entrance
    elif name == "handle_integrated_entrance_stream":
        from service.integrated import (
            handle_integrated_entrance_stream,
        )

        return handle_integrated_entrance_stream
    elif name == "handle_image_generation":
        from service.image import (
            handle_image_generation,
        )

        return handle_image_generation
    elif name == "handle_video_generation":
        from service.video import (
            handle_video_generation,
        )

        return handle_video_generation
    elif name == "handle_text_generation":
        from service.text import handle_text_generation

        return handle_text_generation
    elif name == "handle_speech_generation":
        from service.speech import (
            handle_speech_generation,
        )

        return handle_speech_generation
    elif name == "handle_music_generation":
        from service.music import (
            handle_music_generation,
        )

        return handle_music_generation
    elif name == "handle_detection":
        from service.detection import (
            handle_detection,
        )

        return handle_detection
    elif name == "handle_whiteback_detection":
        from service.whiteback_detection import (
            handle_whiteback_detection,
        )

        return handle_whiteback_detection
    elif name == "get_concurrency_manager":
        from service.concurrency import (
            get_concurrency_manager,
        )

        return get_concurrency_manager
    elif name == "session_concurrency":
        from service.concurrency import (
            session_concurrency,
        )

        return session_concurrency
    elif name == "get_media_registry":
        from media_resource import (
            get_media_registry,
        )

        return get_media_registry
    elif name == "warmup_all":
        from service.warmup import warmup_all

        return warmup_all
    elif name == "warmup_minimal":
        from service.warmup import warmup_minimal

        return warmup_minimal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "handle_integrated_entrance",
    "handle_image_generation",
    "handle_video_generation",
    "handle_text_generation",
    "handle_speech_generation",
    "handle_music_generation",
    "handle_detection",
    "handle_whiteback_detection",
    "get_concurrency_manager",
    "session_concurrency",
    "get_media_registry",
    "warmup_all",
    "warmup_minimal",
]
