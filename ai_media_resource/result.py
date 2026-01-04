"""
存储结果数据结构
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageResult:
    """存储结果"""

    url: str
    url_expire_time: Optional[int] = None  # URL 过期时间（秒级时间戳）


__all__ = ["StorageResult"]
