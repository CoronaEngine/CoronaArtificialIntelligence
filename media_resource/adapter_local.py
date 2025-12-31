"""
本地存储适配器

下载资源到本地文件系统，返回 file:// URL。
"""

from __future__ import annotations

import logging
from typing import Optional

from .adapter_base import (
    StorageAdapter,
    normalize_to_data_uri,
)
from .result import StorageResult

logger = logging.getLogger(__name__)


class LocalStorageAdapter(StorageAdapter):
    """
    本地存储适配器

    下载资源到本地，返回 file:// URL。
    桥接 Backend.local_storage.MediaStore。
    """

    def __init__(self):
        self._store = None

    def _get_store(self):
        """延迟初始化 MediaStore"""
        if self._store is None:
            from Backend.local_storage.utils import get_media_store

            self._store = get_media_store()
        return self._store

    def save_from_url(
        self,
        cloud_url: str,
        session_id: str,
        resource_type: str,
        original_name: Optional[str] = None,
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """下载资源到本地，返回 file:// URL"""
        store = self._get_store()
        local_url = store.save_resource_from_url(
            session_id=session_id,
            url=cloud_url,
            resource_type=resource_type,
            original_name=original_name,
        )
        logger.debug(f"资源已下载到本地: {cloud_url} -> {local_url}")
        return StorageResult(url=local_url, url_expire_time=None)

    def save_from_base64(
        self,
        data_uri: str,
        session_id: str,
        resource_type: str,
        filename_prefix: str = "resource",
        url_expire_time: Optional[int] = None,
    ) -> StorageResult:
        """将 base64 数据保存到本地，返回 file:// URL"""
        normalized_data = normalize_to_data_uri(data_uri, resource_type)
        store = self._get_store()
        local_url = store.save_resource_from_base64(
            session_id=session_id,
            data_uri=normalized_data,
            resource_type=resource_type,
            filename_prefix=filename_prefix,
        )
        logger.debug(f"Base64 数据已保存到本地: {local_url}")
        return StorageResult(url=local_url, url_expire_time=None)


__all__ = ["LocalStorageAdapter"]
