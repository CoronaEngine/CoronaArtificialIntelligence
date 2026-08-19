"""Redis-backed SessionStore with lazy driver loading."""

from __future__ import annotations

import importlib
import json
import threading
from dataclasses import asdict
from typing import Any

from ...cai.capabilities import ComponentHealth, HealthStatus, SessionChange, SessionSnapshot
from ...cai.errors import AdapterConnectionError, HostIntegrationError, PersistenceError
from .config import RedisSessionStoreConfig


class RedisSessionStore:
    def __init__(self, config: RedisSessionStoreConfig, *, client: Any = None) -> None:
        self.config = config
        self._configured_client = client
        self._client: Any = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            try:
                client = self._configured_client
                if client is None:
                    redis = importlib.import_module("redis")
                    client = redis.Redis.from_url(
                        self.config.url,
                        socket_timeout=self.config.socket_timeout,
                        decode_responses=True,
                    )
                client.ping()
            except Exception as exc:
                raise AdapterConnectionError(
                    f"failed to connect Redis session store: {exc}",
                    component="session_store",
                    operation="start",
                    retryable=True,
                ) from exc
            self._client = client

    def create(self, session: SessionSnapshot) -> None:
        client = self._require_started("create")
        try:
            created = client.set(self._key(session.session_key), self._encode(session), nx=True)
        except Exception as exc:
            self._persistence_error("create", exc)
        if not created:
            raise ValueError(f"session already exists: {session.session_key}")

    def get(self, session_key: str) -> SessionSnapshot | None:
        client = self._require_started("get")
        try:
            payload = client.get(self._key(session_key))
        except Exception as exc:
            self._persistence_error("get", exc)
        if payload is None:
            return None
        return self._decode(payload)

    def update(self, change: SessionChange) -> None:
        current = self.get(change.session_key)
        if current is None:
            raise KeyError(change.session_key)
        snapshot = SessionSnapshot(
            change.session_key,
            change.state if change.state is not None else current.state,
            {**current.values, **change.values},
        )
        client = self._require_started("update")
        try:
            client.set(self._key(change.session_key), self._encode(snapshot))
        except Exception as exc:
            self._persistence_error("update", exc)

    def delete(self, session_key: str) -> bool:
        client = self._require_started("delete")
        try:
            return bool(client.delete(self._key(session_key)))
        except Exception as exc:
            self._persistence_error("delete", exc)

    def flush(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
                self._client = None

    def health(self) -> ComponentHealth:
        if self._client is None:
            return ComponentHealth(HealthStatus.DEGRADED, "adapter is not started")
        try:
            self._client.ping()
        except Exception as exc:
            return ComponentHealth(HealthStatus.UNAVAILABLE, str(exc))
        return ComponentHealth(HealthStatus.HEALTHY)

    def _key(self, session_key: str) -> str:
        return f"{self.config.key_prefix}{session_key}"

    @staticmethod
    def _encode(snapshot: SessionSnapshot) -> str:
        return json.dumps(asdict(snapshot), ensure_ascii=False)

    @staticmethod
    def _decode(payload: str | bytes) -> SessionSnapshot:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        return SessionSnapshot(data["session_key"], data["state"], data.get("values", {}))

    def _require_started(self, operation: str):
        if self._client is None:
            raise HostIntegrationError(
                "Redis session adapter is not started",
                component="session_store",
                operation=operation,
            )
        return self._client

    @staticmethod
    def _persistence_error(operation: str, exc: Exception):
        raise PersistenceError(
            f"Redis session {operation} failed: {exc}",
            component="session_store",
            operation=operation,
            retryable=True,
        ) from exc
