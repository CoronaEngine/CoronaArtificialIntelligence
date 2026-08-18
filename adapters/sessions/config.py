"""Typed configuration owned by the optional session adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisSessionStoreConfig:
    url: str
    key_prefix: str = "quasar:session:"
    socket_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("Redis URL cannot be empty")
        if not self.key_prefix:
            raise ValueError("Redis key_prefix cannot be empty")
        if self.socket_timeout <= 0:
            raise ValueError("Redis socket_timeout must be greater than zero")


@dataclass(frozen=True)
class MongoSessionStoreConfig:
    uri: str
    database: str
    collection: str = "sessions"
    server_selection_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("MongoDB URI cannot be empty")
        if not self.database.strip():
            raise ValueError("MongoDB database cannot be empty")
        if not self.collection.strip():
            raise ValueError("MongoDB collection cannot be empty")
        if self.server_selection_timeout_ms <= 0:
            raise ValueError("MongoDB timeout must be greater than zero")


__all__ = ["MongoSessionStoreConfig", "RedisSessionStoreConfig"]
