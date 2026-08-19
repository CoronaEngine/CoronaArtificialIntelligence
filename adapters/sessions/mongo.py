"""MongoDB-backed SessionStore with lazy driver loading."""

from __future__ import annotations

import importlib
import threading
from typing import Any

from ...cai.capabilities import ComponentHealth, HealthStatus, SessionChange, SessionSnapshot
from ...cai.errors import AdapterConnectionError, HostIntegrationError, PersistenceError
from .config import MongoSessionStoreConfig


class MongoSessionStore:
    def __init__(self, config: MongoSessionStoreConfig, *, client: Any = None) -> None:
        self.config = config
        self._configured_client = client
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            try:
                client = self._configured_client
                if client is None:
                    pymongo = importlib.import_module("pymongo")
                    client = pymongo.MongoClient(
                        self.config.uri,
                        serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
                    )
                client.admin.command("ping")
                collection = client[self.config.database][self.config.collection]
            except Exception as exc:
                raise AdapterConnectionError(
                    f"failed to connect MongoDB session store: {exc}",
                    component="session_store",
                    operation="start",
                    retryable=True,
                ) from exc
            self._client = client
            self._collection = collection

    def create(self, session: SessionSnapshot) -> None:
        collection = self._require_started("create")
        try:
            collection.insert_one(self._document(session))
        except Exception as exc:
            raise PersistenceError(
                f"MongoDB session create failed: {exc}",
                component="session_store",
                operation="create",
            ) from exc

    def get(self, session_key: str) -> SessionSnapshot | None:
        collection = self._require_started("get")
        try:
            document = collection.find_one({"session_key": session_key})
        except Exception as exc:
            self._persistence_error("get", exc)
        if document is None:
            return None
        return SessionSnapshot(
            document["session_key"], document.get("state", "created"), document.get("values", {})
        )

    def update(self, change: SessionChange) -> None:
        current = self.get(change.session_key)
        if current is None:
            raise KeyError(change.session_key)
        snapshot = SessionSnapshot(
            change.session_key,
            change.state if change.state is not None else current.state,
            {**current.values, **change.values},
        )
        collection = self._require_started("update")
        try:
            collection.replace_one({"session_key": change.session_key}, self._document(snapshot))
        except Exception as exc:
            self._persistence_error("update", exc)

    def delete(self, session_key: str) -> bool:
        collection = self._require_started("delete")
        try:
            return bool(collection.delete_one({"session_key": session_key}).deleted_count)
        except Exception as exc:
            self._persistence_error("delete", exc)

    def flush(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._collection = None

    def health(self) -> ComponentHealth:
        if self._client is None:
            return ComponentHealth(HealthStatus.DEGRADED, "adapter is not started")
        try:
            self._client.admin.command("ping")
        except Exception as exc:
            return ComponentHealth(HealthStatus.UNAVAILABLE, str(exc))
        return ComponentHealth(HealthStatus.HEALTHY)

    @staticmethod
    def _document(snapshot: SessionSnapshot) -> dict[str, Any]:
        return {
            "session_key": snapshot.session_key,
            "state": snapshot.state,
            "values": dict(snapshot.values),
        }

    def _require_started(self, operation: str):
        if self._collection is None:
            raise HostIntegrationError(
                "MongoDB session adapter is not started",
                component="session_store",
                operation=operation,
            )
        return self._collection

    @staticmethod
    def _persistence_error(operation: str, exc: Exception):
        raise PersistenceError(
            f"MongoDB session {operation} failed: {exc}",
            component="session_store",
            operation=operation,
            retryable=True,
        ) from exc
