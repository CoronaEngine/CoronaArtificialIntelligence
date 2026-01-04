"""
媒体资源组件（Media Resource Component）

统一管理媒体资源（图片、视频、音频）的存储和注册。

模块结构：
- config: 配置常量（TTL、缓存设置等）
- cache: 内存缓存实现（云端模式使用）
- result: 存储结果数据结构
- record: 媒体记录数据结构
- adapter_base: 存储适配器抽象基类
- adapter_local: 本地存储适配器
- adapter_cloud: 云端存储适配器
- adapter_factory: 适配器工厂
- task_executor: 异步任务执行器
- registry: 媒体资源注册表

快速使用：
>>> from ai_media_resource import (
...     get_media_registry,
...     get_storage_adapter,
...     get_task_executor,
... )
>>> registry = get_media_registry()
>>> adapter = get_storage_adapter()
>>> executor = get_task_executor()
"""

from __future__ import annotations

# 配置
from ai_media_resource.config import (
    MEMORY_CACHE_CONFIG,
    URL_TTL_HOURS,
    calculate_expire_time,
)

# 内存缓存
from ai_media_resource.cache import (
    CacheEntry,
    MemoryCache,
    get_memory_cache,
    reset_memory_cache,
)

# 数据结构
from ai_media_resource.result import StorageResult
from ai_media_resource.record import MediaRecord

# 存储适配器
from ai_media_resource.adapter_base import (
    StorageAdapter,
    normalize_to_data_uri,
)
from ai_media_resource.adapter_local import (
    LocalStorageAdapter,
)
from ai_media_resource.adapter_cloud import (
    CloudStorageAdapter,
)
from ai_media_resource.adapter_factory import (
    get_storage_adapter,
    reset_storage_adapter,
    resolve_cache_url,
)

# 异步任务执行器
from ai_media_resource.task_executor import (
    TaskExecutor,
    TaskRecord,
    TaskStatus,
    get_task_executor,
    reset_task_executor,
)

# 媒体注册表
from ai_media_resource.registry import (
    MediaResourceRegistry,
    get_media_registry,
    reset_media_registry,
)


__all__ = [
    # 配置
    "URL_TTL_HOURS",
    "MEMORY_CACHE_CONFIG",
    "calculate_expire_time",
    # 内存缓存
    "CacheEntry",
    "MemoryCache",
    "get_memory_cache",
    "reset_memory_cache",
    # 数据结构
    "StorageResult",
    "MediaRecord",
    # 存储适配器
    "StorageAdapter",
    "normalize_to_data_uri",
    "LocalStorageAdapter",
    "CloudStorageAdapter",
    "get_storage_adapter",
    "reset_storage_adapter",
    "resolve_cache_url",
    # 异步任务执行器
    "TaskExecutor",
    "TaskRecord",
    "TaskStatus",
    "get_task_executor",
    "reset_task_executor",
    # 媒体注册表
    "MediaResourceRegistry",
    "get_media_registry",
    "reset_media_registry",
]
