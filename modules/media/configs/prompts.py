from __future__ import annotations

from typing import Any, Dict


from config.prompts import MediaToolPrompts, DetectionPromptConfig
from modules.image_generate.configs.settings import IMAGE_SETTINGS, IMAGE_PROMPTS
from modules.video_generate.configs.settings import VIDEO_SETTINGS, VIDEO_PROMPTS
from modules.speech_generate.configs.settings import SPEECH_PROMPTS
from modules.music_generate.configs.settings import MUSIC_PROMPTS
from modules.omni.configs.settings import OMNI_SETTINGS, OMNI_PROMPTS
# ===========================================================================
# 目标检测默认提示词（占位配置，实际配置在 InnerAgentWorkflow/ai_config/omni/base.py）
# ===========================================================================

DETECTION_PROMPTS = DetectionPromptConfig(
    tool_description="使用视觉语言模型进行图像目标检测。",
    fields={
        "image_url": "要检测的图片 URL",
        "target_description": "可选：要检测的目标类型描述",
    },
    detection_prompt=(
        "请检测图像中所有显著的主体对象，对于每个检测到的对象，"
        "请提供其类别和边界框，格式为：<bbox>x_min y_min x_max y_max</bbox>，"
        "坐标范围 0-1000。"
    ),
    target_prefix="请重点检测图片中的「{target_description}」。\n\n",
)

# ===========================================================================
# 整合的媒体配置
# ===========================================================================


from service.entrance import ai_entrance

@ai_entrance.collector.register_setting("media")
def MEDIA_SETTINGS() -> Dict[str, Any]:
    return {
        "audio": {
            "sample_rate": 24000,
            "bitrate": 160,
        },
        "image": IMAGE_SETTINGS,
        "video": VIDEO_SETTINGS,
        "omni": OMNI_SETTINGS,
        # detection 配置已迁移到 InnerAgentWorkflow/ai_config/omni/base.py
    }

# ===========================================================================
# 整合的媒体工具提示词
# ===========================================================================

@ai_entrance.collector.register_prompts("media")
def MEDIA_TOOL_PROMPTS():
    return MediaToolPrompts(
        image=IMAGE_PROMPTS,
        video=VIDEO_PROMPTS,
        speech=SPEECH_PROMPTS,
        music=MUSIC_PROMPTS,
        omni=OMNI_PROMPTS,
        detection=DETECTION_PROMPTS,
    )