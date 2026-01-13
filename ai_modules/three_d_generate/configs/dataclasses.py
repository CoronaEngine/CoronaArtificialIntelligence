from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=False)
class Rodin3DSettings:
    """
    Rodin 3D 生成配置
    """

    provider: str       # providers 里的 key
    base_url: str
    api_key: str

    # Rodin API endpoints（官方文档：/api/v2/rodin, /api/v2/status, /api/v2/download）
    generate_path: str 
    status_path: str 
    download_path: str 

    request_timeout: float = 300.0
    poll_interval: float = 2.0
    poll_timeout: float = 900.0  
