from .image_tools import load_image_tools
from .video_tools import load_video_tools
from .speech_tools import load_speech_tools
from .music_tools import load_music_tools
from .omni_tools import load_omni_tools

# detection_tools 已迁移到 InnerAgentWorkflow/ai_tools/

__all__ = [
    "load_image_tools",
    "load_video_tools",
    "load_speech_tools",
    "load_music_tools",
    "load_omni_tools",
]
