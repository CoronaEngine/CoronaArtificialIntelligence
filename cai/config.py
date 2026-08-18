"""Typed, host-independent runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigurationError


@dataclass(frozen=True)
class RuntimeConfig:
    request_timeout: float = 60.0
    shutdown_timeout: float = 30.0
    max_concurrency: int = 8
    log_level: str = "INFO"
    persistence_policy: str = "memory"

    def __post_init__(self) -> None:
        if self.request_timeout <= 0:
            self._invalid("request_timeout must be greater than zero")
        if self.shutdown_timeout < 0:
            self._invalid("shutdown_timeout cannot be negative")
        if self.max_concurrency <= 0:
            self._invalid("max_concurrency must be greater than zero")
        normalized_level = self.log_level.upper()
        if normalized_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            self._invalid(f"unsupported log_level: {self.log_level}")
        if not self.persistence_policy:
            self._invalid("persistence_policy cannot be empty")
        object.__setattr__(self, "log_level", normalized_level)

    @staticmethod
    def _invalid(message: str) -> None:
        raise ConfigurationError(message, component="config", operation="validate")


__all__ = ["RuntimeConfig"]
