"""Artifact storage adapters."""

from .local import LocalFileArtifactStore, LocalFileArtifactStoreConfig

__all__ = ["LocalFileArtifactStore", "LocalFileArtifactStoreConfig"]
