"""
网络和轮询配置
"""

from __future__ import annotations

from typing import Any, Dict

# 网络请求配置
NETWORK_SETTINGS: Dict[str, Any] = {
    "request_timeout": 60,
    "download_timeout": 300,
    "download_chunk_size": 8192,
    "download_retries": 2,
    "download_backoff_factor": 0.5,
}

# 异步任务轮询配置
POLLING_SETTINGS: Dict[str, Any] = {
    "max_wait_seconds": 150,
    "default_interval": 3.0,
    "service_intervals": {
        "speech": 2.0,
        "music": 5.0,
        "video": 3.0,
    },
}
