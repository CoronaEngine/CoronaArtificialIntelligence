"""
视频生成配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict


# ===========================================================================
# 视频生成配置 - 默认预设
# ===========================================================================

VIDEO_SETTINGS: Dict[str, Any] = {
    "enable": True,
    "provider": "example",
    "model": "video-model",
}
