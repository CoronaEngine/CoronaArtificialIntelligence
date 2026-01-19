"""3D生成配置和提示词"""
from __future__ import annotations

from typing import Any, Dict
import os

from ai_service.entrance import ai_entrance
@ai_entrance.collector.register_setting("rodin3d")
def RODIN_3D_SETTINGS() -> Dict[str, Any]:
    return {
        "base_url": "https://api.hyper3d.com",
        "api_key": "ew8ZMLzbHDCobj0OEaEbvkQcTHW2ecrHQFFdInEZyeckQc3jS9bQQBNumpXyCMSk",
       
        "generate_path": "/api/v2/rodin",
        "download_path": "/api/v2/download",
        "status_path": "/api/v2/status",
        "request_timeout": 300.0,
        "poll_interval": 2.0,
        "poll_timeout": 900.0,
    }


