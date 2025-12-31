from ...image_generate.configs.settings import IMAGE_SETTINGS
from ...omni.configs.settings import OMNI_SETTINGS
from ...video_generate.configs.settings import VIDEO_SETTINGS
from service.entrance import ai_entrance
from typing import Any, Dict


# ===========================================================================
# 整合的媒体配置
# ===========================================================================

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
