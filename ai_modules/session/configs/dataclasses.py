"""
网络和轮询配置
"""

from dataclasses import dataclass

@dataclass(frozen=False)
class SessionConfig:
    """会话管理配置"""

    ttl_seconds: int = 86400  # 24 hours
    max_sessions: int = 10000
    max_messages_per_session: int = 100
    max_concurrent_requests: int | None = None
    file_registry_max_workers: int | None = None
