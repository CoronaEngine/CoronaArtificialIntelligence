from __future__ import annotations

from typing import Any, Dict
from ai_service.entrance import ai_entrance


@ai_entrance.collector.register_setting("scene_breakdown")
def SCENE_BREAKDOWN_SETTINGS() -> Dict[str, Any]:
    return {
        "enable": True,
        "temperature": 0.6,
        "request_timeout": 60.0,
        "default_style": "现代",
        "default_detail_level": "中等",
        "default_mesh_format": "glb",
        "default_texture_size": 2048,
    }
