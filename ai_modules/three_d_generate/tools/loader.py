"""
多模态理解配置加载器
"""

from typing import Any, Mapping

from ai_modules.three_d_generate.configs.dataclasses import Rodin3DSettings
from ai_service.entrance import ai_entrance
from ai_tools.helpers import _as_bool,_as_float

print("加载 Rodin 3D 配置加载器")
@ai_entrance.collector.register_loader("rodin3d")
def _load_rodin_3d_config(raw: Mapping[str, Any] | None) -> Rodin3DSettings:
    """加载多模态理解配置"""
    
    print("加载 Rodin 3D 配置加载器")
    if not isinstance(raw, Mapping):
        return Rodin3DSettings()

    return Rodin3DSettings(
        provider=raw.get("provider",''),       # providers 里的 key
        base_url=raw.get("base_url"),
        api_key=raw.get("api_key"),

        # Rodin API endpoints（官方文档：/api/v2/rodin, /api/v2/status, /api/v2/download）
        generate_path=raw.get("generate_path"), 
        status_path=raw.get("status_path"), 
        download_path=raw.get("download_path"), 

        request_timeout=_as_float(raw.get("request_timeout"), 300.0),
        poll_interval=_as_float(raw.get("poll_interval"), 2.0),
        poll_timeout=_as_float(raw.get("poll_timeout"), 900.0)
    )
