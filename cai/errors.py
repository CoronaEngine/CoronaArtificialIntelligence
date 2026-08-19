"""Stable, transport-neutral errors exposed by Quasar Core."""
from __future__ import annotations

from typing import Any


class QuasarRuntimeError(RuntimeError):
    code = "quasar_runtime_error"

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        operation: str | None = None,
        retryable: bool = False,
        trace_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.component = component
        self.operation = operation
        self.retryable = retryable
        self.trace_key = trace_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "component": self.component,
            "operation": self.operation,
            "retryable": self.retryable,
            "trace_key": self.trace_key,
        }


class ConfigurationError(QuasarRuntimeError):
    code = "configuration_error"


class CapabilityUnavailableError(QuasarRuntimeError):
    code = "capability_unavailable_error"


class AdapterConnectionError(QuasarRuntimeError):
    code = "adapter_connection_error"


class PersistenceError(QuasarRuntimeError):
    code = "persistence_error"


class BufferFlushError(QuasarRuntimeError):
    code = "buffer_flush_error"


class ToolExecutionError(QuasarRuntimeError):
    code = "tool_execution_error"


class HostIntegrationError(QuasarRuntimeError):
    code = "host_integration_error"


__all__ = [
    "QuasarRuntimeError",
    "ConfigurationError",
    "CapabilityUnavailableError",
    "AdapterConnectionError",
    "PersistenceError",
    "BufferFlushError",
    "ToolExecutionError",
    "HostIntegrationError",
]
