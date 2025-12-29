"""
媒体资源组件配置

集中管理媒体资源相关的配置常量。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict

# ==============================================================================
# URL 有效期配置（小时）
# ==============================================================================

URL_TTL_HOURS: Dict[str, int] = {
    "image": 2,  # 图像 2 小时
    "video": 24,  # 视频 1 天
    "audio": 360,  # BGM 15 天
}

# ==============================================================================
# 内存缓存配置
# ==============================================================================

MEMORY_CACHE_CONFIG: Dict[str, int] = {
    "default_ttl_seconds": 3600,  # 默认过期时间：1 小时
    "max_items": 1000,  # 最大缓存条目数
    "cleanup_interval": 300,  # 清理间隔：5 分钟
}


# ==============================================================================
# 工具函数
# ==============================================================================


def calculate_expire_time(resource_type: str) -> int:
    """
    根据资源类型计算 URL 过期时间

    参数:
    - resource_type: 资源类型 (image/video/audio)

    返回:
    - 秒级时间戳
    """
    ttl_hours = URL_TTL_HOURS.get(resource_type, 2)  # 默认 2 小时
    expire_dt = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    return int(expire_dt.timestamp())


__all__ = [
    "URL_TTL_HOURS",
    "MEMORY_CACHE_CONFIG",
    "calculate_expire_time",
]
