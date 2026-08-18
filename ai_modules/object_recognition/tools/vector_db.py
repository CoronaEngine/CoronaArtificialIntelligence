"""Object-recognition compatibility facade over the public VectorStore port.

New integrations inject a VectorStore. Supplying ``db_path`` retains the
legacy behavior through the optional SQLite adapter, started only on first use.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....adapters.vector import SQLiteVectorStore, SQLiteVectorStoreConfig
from ....cai.capabilities import VectorQuery, VectorRecord, VectorStore

logger = logging.getLogger(__name__)
_DB_INSTANCE_LOCK = threading.Lock()
_DB_INSTANCES: Dict[Tuple[str, int], "VectorDB"] = {}


class VectorDB:
    """Legacy object API backed by an injected VectorStore."""

    def __init__(
        self,
        db_path: str | None = None,
        vector_dim: int = 1024,
        *,
        vector_store: VectorStore | None = None,
    ) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be greater than zero")
        if vector_store is None and not db_path:
            raise ValueError("db_path or vector_store is required")
        self.db_path = os.path.abspath(db_path) if db_path else ""
        self.vector_dim = vector_dim
        self._store = vector_store or SQLiteVectorStore(
            SQLiteVectorStoreConfig(self.db_path, vector_dim=vector_dim)
        )
        self._started = False
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            start = getattr(self._store, "start", None)
            if callable(start):
                start()
            self._started = True

    def insert_object(
        self,
        object_id: str,
        embedding: np.ndarray,
        name: str = "",
        category: str = "",
        image_paths: Optional[List[str]] = None,
        description: str = "",
    ) -> int:
        self._validate(embedding)
        self.start()
        if self.get_object(object_id) is not None:
            raise ValueError(f"物体 '{object_id}' 已存在，请使用不同的 object_id")
        metadata = self._make_metadata(name, category, image_paths, description)
        self._store.upsert([VectorRecord(object_id, tuple(map(float, embedding)), metadata)])
        self._metadata[object_id] = metadata
        return self.count()

    def update_object(
        self,
        object_id: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        category: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> bool:
        self._validate(embedding)
        current = self.get_object(object_id)
        if current is None:
            return False
        metadata = {
            **current,
            **({"name": name} if name is not None else {}),
            **({"category": category} if category is not None else {}),
            **({"image_paths": image_paths} if image_paths is not None else {}),
            **({"description": description} if description is not None else {}),
        }
        metadata.pop("object_id", None)
        self._store.upsert([VectorRecord(object_id, tuple(map(float, embedding)), metadata)])
        self._metadata[object_id] = metadata
        return True

    def delete_object(self, object_id: str) -> bool:
        self.start()
        existed = self.get_object(object_id) is not None
        self._store.delete([object_id])
        self._metadata.pop(object_id, None)
        return existed

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        self._validate(query_embedding)
        self.start()
        matches = self._store.search(VectorQuery(tuple(map(float, query_embedding)), top_k))
        return [
            {
                "object_id": match.key,
                **dict(match.metadata),
                "distance": (1.0 / match.score - 1.0) if match.score > 0 else float("inf"),
            }
            for match in matches
        ]

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        self.start()
        get_metadata = getattr(self._store, "get_metadata", None)
        metadata = get_metadata(object_id) if callable(get_metadata) else self._metadata.get(object_id)
        return None if metadata is None else {"object_id": object_id, **dict(metadata)}

    def list_objects(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        self.start()
        list_metadata = getattr(self._store, "list_metadata", None)
        entries = list_metadata(limit=max(limit, 1000)) if callable(list_metadata) else tuple(self._metadata.items())
        results = [
            {"object_id": key, **dict(metadata)}
            for key, metadata in entries
            if category is None or metadata.get("category") == category
        ]
        return results[:limit]

    def count(self) -> int:
        self.start()
        count = getattr(self._store, "count", None)
        return int(count()) if callable(count) else len(self._metadata)

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            close = getattr(self._store, "close", None)
            if callable(close):
                close()
            self._started = False

    def _validate(self, embedding: np.ndarray) -> None:
        if embedding.shape != (self.vector_dim,):
            raise ValueError(
                f"向量维度不匹配: 期望 ({self.vector_dim},)，实际 {embedding.shape}"
            )

    @staticmethod
    def _make_metadata(name, category, image_paths, description) -> dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "image_paths": list(image_paths or []),
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def get_vector_db(db_path: str, vector_dim: int = 1024) -> VectorDB:
    """Deprecated cache for legacy callers; new code injects VectorStore."""
    normalized_path = os.path.abspath(db_path)
    cache_key = (normalized_path, vector_dim)
    instance = _DB_INSTANCES.get(cache_key)
    if instance is not None:
        return instance
    with _DB_INSTANCE_LOCK:
        instance = _DB_INSTANCES.get(cache_key)
        if instance is None:
            instance = VectorDB(normalized_path, vector_dim)
            _DB_INSTANCES[cache_key] = instance
    return instance


def _numpy_to_vec_string(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vec.astype(np.float32)) + "]"


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        logger.warning("输入零向量，跳过归一化")
        return vec
    return vec / norm


__all__ = ["VectorDB", "get_vector_db", "normalize_vector"]
