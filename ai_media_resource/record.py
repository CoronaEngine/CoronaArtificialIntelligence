"""
媒体资源记录数据结构
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MediaRecord:
    """
    媒体资源记录

    存储媒体资源的元数据和 URL 信息。
    """

    # 资源标识
    file_id: str
    session_id: str
    resource_type: str  # "image" | "video" | "audio"

    # URL 信息
    content_url: Optional[str] = None
    url_expire_time: Optional[int] = None  # URL 过期时间（秒级时间戳）

    # 媒体元数据
    content_text: str = ""
    source: str = "tool"  # "tool" | "upload"
    # 资源写入时间（秒级时间戳）
    timestamp: int = field(default_factory=lambda: int(time.time()))
    parameter: Dict[str, Any] = field(default_factory=dict)

    # 关联的异步任务 ID（如果有）
    task_id: Optional[str] = None
    error: Optional[str] = None

    def to_part(self, include_meta: bool = False) -> Dict[str, Any]:
        """
        转换为 part 结构（用于构建消息）

        参数:
        - include_meta: 是否包含 _meta 元数据
        """
        part: Dict[str, Any] = {
            "content_type": self.resource_type,
            "content_text": self.content_text,
            "file_id": self.file_id,
        }
        if self.content_url:
            part["content_url"] = self.content_url
        if self.url_expire_time:
            part["url_expire_time"] = self.url_expire_time
        if self.parameter:
            part["parameter"] = self.parameter
        if include_meta:
            part["_meta"] = {
                "source": self.source,
                "timestamp": self.timestamp,
            }
        return part


__all__ = ["MediaRecord"]
