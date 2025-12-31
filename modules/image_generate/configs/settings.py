"""
图像生成配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict

# ===========================================================================
# 图像生成配置 - 默认预设
# ===========================================================================

IMAGE_SETTINGS: Dict[str, Any] = {
    "enable": True,
    "provider": "example",
    "model": "image-model",
    "base_url": "https://api.example.com/v1/images/generations",
    "max_size": 2000,
    "min_size": 360,
}
