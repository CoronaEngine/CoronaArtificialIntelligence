from __future__ import annotations

from ...image_generate.configs.prompts import IMAGE_PROMPTS
from ...music_generate.configs.prompts import MUSIC_PROMPTS
from ...omni.configs.prompts import OMNI_PROMPTS
from ...speech_generate.configs.prompts import SPEECH_PROMPTS
from ...video_generate.configs.prompts import VIDEO_PROMPTS
from service.entrance import ai_entrance
from config.prompts import MediaToolPrompts, DetectionPromptConfig


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