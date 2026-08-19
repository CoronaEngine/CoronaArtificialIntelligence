"""Fluent, non-global runtime assembly helpers."""

from __future__ import annotations

from typing import Any

from .capabilities import Capability
from .config import RuntimeConfig
from .runtime import CAIRuntime
from .tools import CapabilityToolRegistry, ToolSpec


class RuntimeBuilder:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self._config = config or RuntimeConfig()
        self._capabilities: dict[str | Capability, Any] = {}
        self._plugins: list[Any] = []
        self._tools: list[ToolSpec] = []

    def use_core_defaults(self) -> "RuntimeBuilder":
        return self

    def use_capability(self, capability: str | Capability, value: Any) -> "RuntimeBuilder":
        self._capabilities[capability] = value
        return self

    def use_model(self, model: Any) -> "RuntimeBuilder":
        return self.use_capability(Capability.MODEL, model)

    def use_session_store(self, store: Any) -> "RuntimeBuilder":
        return self.use_capability(Capability.SESSION_STORE, store)

    def use_artifact_store(self, store: Any) -> "RuntimeBuilder":
        return self.use_capability(Capability.ARTIFACT_STORE, store)

    def use_vector_store(self, store: Any) -> "RuntimeBuilder":
        return self.use_capability(Capability.VECTOR_STORE, store)

    def install(self, plugin: Any) -> "RuntimeBuilder":
        self._plugins.append(plugin)
        return self

    def add_tool(self, spec: ToolSpec) -> "RuntimeBuilder":
        self._tools.append(spec)
        return self

    def build(self) -> CAIRuntime:
        capabilities = dict(self._capabilities)
        capabilities[Capability.CONFIG] = self._config
        runtime = CAIRuntime(capabilities=capabilities)
        for plugin in self._plugins:
            runtime.register_plugin(plugin)
        registry = CapabilityToolRegistry(runtime.capabilities)
        for spec in self._tools:
            registry.register(spec)
        registry.validate()
        runtime.set_registry("tool", registry)
        runtime.set_capability(Capability.TOOL, registry)
        return runtime


__all__ = ["RuntimeBuilder"]
