"""SQLite vector-store adapter with explicit lifecycle management."""

from __future__ import annotations

import importlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from ...cai.capabilities import (
    ComponentHealth,
    HealthStatus,
    VectorMatch,
    VectorQuery,
    VectorRecord,
)
from ...cai.errors import AdapterConnectionError, HostIntegrationError, PersistenceError


class SQLiteVectorStore:
    """A lazy SQLite vector adapter.

    Construction only records configuration. Database access and optional
    dependency loading happen during :meth:`start`.
    """

    def __init__(self, database: str | Path, *, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be greater than zero")
        self.database = Path(database).expanduser().resolve()
        self.vector_dim = vector_dim
        self._connection: sqlite3.Connection | None = None
        self._sqlite_vec: Any | None = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            try:
                sqlite_vec = importlib.import_module("sqlite_vec")
                self.database.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.database, check_same_thread=False)
                connection.enable_load_extension(True)
                sqlite_vec.load(connection)
                connection.enable_load_extension(False)
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_key TEXT NOT NULL UNIQUE,
                        metadata TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vector_embeddings USING vec0(
                        record_rowid INTEGER PRIMARY KEY,
                        embedding FLOAT[{self.vector_dim}]
                    )
                    """
                )
                connection.commit()
            except Exception as exc:
                if "connection" in locals():
                    connection.close()
                raise AdapterConnectionError(
                    f"failed to start SQLite vector adapter: {exc}",
                    component="vector_store",
                    operation="start",
                ) from exc
            self._sqlite_vec = sqlite_vec
            self._connection = connection

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        connection = self._require_started("upsert")
        with self._lock:
            try:
                for record in records:
                    self._validate_vector(record.vector)
                    connection.execute(
                        """
                        INSERT INTO vector_metadata(record_key, metadata) VALUES (?, ?)
                        ON CONFLICT(record_key) DO UPDATE SET metadata = excluded.metadata
                        """,
                        (record.key, json.dumps(dict(record.metadata), ensure_ascii=False)),
                    )
                    row = connection.execute(
                        "SELECT id FROM vector_metadata WHERE record_key = ?",
                        (record.key,),
                    ).fetchone()
                    connection.execute(
                        "DELETE FROM vector_embeddings WHERE record_rowid = ?",
                        (row[0],),
                    )
                    connection.execute(
                        "INSERT INTO vector_embeddings(record_rowid, embedding) VALUES (?, ?)",
                        (row[0], self._serialize(record.vector)),
                    )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                if isinstance(exc, ValueError):
                    raise
                raise PersistenceError(
                    f"failed to upsert vectors: {exc}",
                    component="vector_store",
                    operation="upsert",
                ) from exc

    def search(self, query: VectorQuery) -> Sequence[VectorMatch]:
        connection = self._require_started("search")
        self._validate_vector(query.vector)
        if query.limit <= 0:
            raise ValueError("query limit must be greater than zero")
        with self._lock:
            try:
                rows = connection.execute(
                    """
                    SELECT metadata.record_key, metadata.metadata, nearest.distance
                    FROM (
                        SELECT record_rowid, distance
                        FROM vector_embeddings
                        WHERE embedding MATCH ? AND k = ?
                        ORDER BY distance
                    ) AS nearest
                    JOIN vector_metadata AS metadata ON metadata.id = nearest.record_rowid
                    ORDER BY nearest.distance
                    """,
                    (self._serialize(query.vector), query.limit),
                ).fetchall()
            except Exception as exc:
                raise PersistenceError(
                    f"failed to search vectors: {exc}",
                    component="vector_store",
                    operation="search",
                ) from exc
        return tuple(
            VectorMatch(key, 1.0 / (1.0 + float(distance)), json.loads(metadata))
            for key, metadata, distance in rows
        )

    def delete(self, record_keys: Sequence[str]) -> int:
        connection = self._require_started("delete")
        deleted = 0
        with self._lock:
            try:
                for key in record_keys:
                    row = connection.execute(
                        "SELECT id FROM vector_metadata WHERE record_key = ?", (key,)
                    ).fetchone()
                    if row is None:
                        continue
                    connection.execute(
                        "DELETE FROM vector_embeddings WHERE record_rowid = ?", (row[0],)
                    )
                    connection.execute("DELETE FROM vector_metadata WHERE id = ?", (row[0],))
                    deleted += 1
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise PersistenceError(
                    f"failed to delete vectors: {exc}",
                    component="vector_store",
                    operation="delete",
                ) from exc
        return deleted

    def flush(self, timeout: float | None = None) -> None:
        connection = self._require_started("flush")
        with self._lock:
            connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._sqlite_vec = None

    def health(self) -> ComponentHealth:
        if self._connection is not None:
            return ComponentHealth(HealthStatus.HEALTHY)
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message="adapter is not started",
        )

    def _serialize(self, vector: Sequence[float]) -> bytes:
        return self._sqlite_vec.serialize_float32(list(vector))

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"expected vector dimension {self.vector_dim}, received {len(vector)}"
            )

    def _require_started(self, operation: str) -> sqlite3.Connection:
        if self._connection is None:
            raise HostIntegrationError(
                "vector adapter is not started",
                component="vector_store",
                operation=operation,
            )
        return self._connection
