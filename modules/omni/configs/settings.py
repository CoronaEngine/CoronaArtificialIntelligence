"""
多模态理解配置和提示词
"""

from __future__ import annotations

from typing import Any, Dict

# ===========================================================================
# 多模态理解配置 - 默认预设
# ===========================================================================

OMNI_SETTINGS: Dict[str, Any] = {
    "enable": True,
    "provider": "example",
    "model": "omni-model",
}
