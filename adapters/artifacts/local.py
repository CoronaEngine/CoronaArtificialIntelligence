"""Explicit local-filesystem implementation of the ArtifactStore port."""
from __future__ import annotations

import secrets
import threading
from pathlib import Path

from ...cai.capabilities import ArtifactInput, ArtifactRef, ComponentHealth, HealthStatus
from ...cai.errors import HostIntegrationError


class LocalFileArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._started = False
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.root.is_dir():
                raise NotADirectoryError(self.root)
            self._started = True

    def put(self, artifact: ArtifactInput) -> ArtifactRef:
        self._require_started("put")
        suffix = Path(artifact.name or "").suffix
        reference = ArtifactRef(secrets.token_urlsafe(24), artifact.name, artifact.media_type)
        target = self._path(reference.key, suffix)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_bytes(bytes(artifact.content))
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return reference

    def open(self, artifact_ref: ArtifactRef):
        self._require_started("open")
        suffix = Path(artifact_ref.name or "").suffix
        target = self._path(artifact_ref.key, suffix)
        if not target.is_file():
            raise FileNotFoundError(artifact_ref.key)
        return target.open("rb")

    def delete(self, artifact_ref: ArtifactRef) -> bool:
        self._require_started("delete")
        suffix = Path(artifact_ref.name or "").suffix
        target = self._path(artifact_ref.key, suffix)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def flush(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self._started = False

    def health(self) -> ComponentHealth:
        if not self._started:
            return ComponentHealth(HealthStatus.DEGRADED, "adapter is not started")
        if not self.root.is_dir():
            return ComponentHealth(HealthStatus.UNAVAILABLE, "artifact root is unavailable")
        return ComponentHealth(HealthStatus.HEALTHY)

    def _path(self, key: str, suffix: str) -> Path:
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in key):
            raise ValueError("invalid artifact key")
        target = (self.root / f"{key}{suffix}").resolve()
        if self.root not in target.parents:
            raise ValueError("artifact path escapes configured root")
        return target

    def _require_started(self, operation: str) -> None:
        if not self._started:
            raise HostIntegrationError(
                "artifact adapter is not started",
                component="artifact_store",
                operation=operation,
            )
