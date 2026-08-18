"""Host-independent Quasar runtime and lifecycle owner."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .capabilities import (
    Capability,
    ComponentHealth,
    HealthStatus,
    RuntimeHealth,
    create_core_defaults,
)
from .errors import (
    AdapterConnectionError,
    BufferFlushError,
    CapabilityUnavailableError,
    ConfigurationError,
    HostIntegrationError,
)


class LazyRegistryRef:
    def __init__(self, module_name: str, getter_name: str):
        self._module_name = module_name
        self._getter_name = getter_name
        self._value = None

    def resolve(self):
        if self._value is None:
            package = __package__ if self._module_name.startswith(".") else None
            module = import_module(self._module_name, package)
            self._value = getattr(module, self._getter_name)()
        return self._value

    def __getattr__(self, name: str):
        return getattr(self.resolve(), name)


def _load_default_ai_entrance():
    from ..ai_service import entrance

    entrance_cls = entrance.ai_entrance
    if not entrance_cls.if_import:
        entrance_cls.reimport()
    return entrance_cls


def _capability_name(name: str | Capability) -> str:
    return name.value if isinstance(name, Capability) else str(name)


@dataclass(frozen=True)
class _CapabilitySpec:
    required: bool = True
    depends_on: tuple[str, ...] = ()


class CAIRuntime:
    """Own all runtime capabilities and their deterministic lifecycle."""

    def __init__(
        self,
        ai_entrance_provider: Callable[[], Any] | None = None,
        registries: dict[str, Any] | None = None,
        capabilities: dict[str | Capability, Any] | None = None,
    ) -> None:
        self._ai_entrance_provider = ai_entrance_provider or _load_default_ai_entrance
        self.metadata: dict[str, Any] = {}
        self.entrance_handlers: dict[str, Any] = {}
        self._state = "new"
        self._started_components: list[str] = []
        self.capabilities: dict[str, Any] = create_core_defaults()
        self._capability_specs: dict[str, _CapabilitySpec] = {
            name: _CapabilitySpec(required=True) for name in self.capabilities
        }
        for name, value in (capabilities or {}).items():
            self.set_capability(name, value, replace=True)

        self.registries = self._create_default_registries()
        if registries:
            self.registries.update(registries)

        from .plugins import PluginManager

        self.plugin_manager = PluginManager(self)

    @property
    def state(self) -> str:
        return self._state

    def get_ai_entrance(self):
        return self._ai_entrance_provider()

    def chat_stream(self, payload: dict) -> Iterator[str]:
        if self._state == "closed":
            raise HostIntegrationError(
                "runtime is closed", component="runtime", operation="chat_stream"
            )
        if self._state == "new":
            self.start()
        handler = self.get_entrance_handler("handle_integrated_entrance_stream")
        yield from handler(payload)

    def register_entrance_handler(self, name: str, handler: Any) -> None:
        if self._state == "closed":
            raise HostIntegrationError(
                "runtime is closed", component="runtime", operation="register_entrance_handler"
            )
        self.entrance_handlers[name] = handler

    def get_entrance_handler(self, name: str):
        handler = self.entrance_handlers.get(name)
        if handler is not None:
            return handler
        return getattr(self.get_ai_entrance(), name)

    def get_registry(self, name: str):
        registry = self.registries[name]
        if isinstance(registry, LazyRegistryRef):
            return registry.resolve()
        return registry

    def set_registry(self, name: str, registry: Any) -> None:
        self.registries[name] = registry

    def set_capability(
        self,
        name: str | Capability,
        value: Any,
        *,
        depends_on: tuple[str | Capability, ...] = (),
        required: bool = True,
        replace: bool = False,
    ) -> None:
        if self._state == "closed":
            raise HostIntegrationError(
                "runtime is closed", component="runtime", operation="set_capability"
            )
        normalized = _capability_name(name)
        if normalized in self.capabilities and not replace:
            raise ConfigurationError(
                f"capability is already configured: {normalized}",
                component=normalized,
                operation="register",
            )
        self.capabilities[normalized] = value
        self._capability_specs[normalized] = _CapabilitySpec(
            required=required,
            depends_on=tuple(_capability_name(dependency) for dependency in depends_on),
        )

    def get_capability(self, name: str | Capability, default: Any = None) -> Any:
        return self.capabilities.get(_capability_name(name), default)

    def require_capability(self, name: str | Capability) -> Any:
        normalized = _capability_name(name)
        try:
            return self.capabilities[normalized]
        except KeyError as exc:
            raise CapabilityUnavailableError(
                f"capability is not configured: {normalized}",
                component=normalized,
                operation="resolve",
            ) from exc

    def register_tool_loader_registrar(self, registrar: Callable[[Any], None]) -> None:
        registrars = self.capabilities.setdefault("tool_loader_registrars", [])
        if registrar not in registrars:
            registrars.append(registrar)

    def register_plugin(self, plugin: Any) -> None:
        self.plugin_manager.register(plugin)

    def start(self) -> None:
        if self._state == "closed":
            raise HostIntegrationError(
                "runtime is closed", component="runtime", operation="start"
            )
        if self._state == "started":
            return

        ordered = self._ordered_lifecycle_components()
        try:
            for name in ordered:
                component = self.capabilities[name]
                start = getattr(component, "start", None)
                if callable(start):
                    start()
                self._started_components.append(name)
        except Exception as exc:
            self._close_started_components()
            raise AdapterConnectionError(
                str(exc), component=name, operation="start", retryable=False
            ) from exc
        self._state = "started"

    def close(self, timeout: float | None = None) -> None:
        if self._state == "closed":
            return
        started = list(reversed(self._started_components))
        flush_error: tuple[str, Exception] | None = None
        for name in started:
            flush = getattr(self.capabilities[name], "flush", None)
            if callable(flush):
                try:
                    flush(timeout=timeout)
                except Exception as exc:
                    if flush_error is None:
                        flush_error = (name, exc)
        self._close_started_components()
        self.plugin_manager.shutdown()
        self._state = "closed"
        if flush_error is not None:
            name, exc = flush_error
            raise BufferFlushError(
                str(exc),
                component=name,
                operation="flush",
                retryable=True,
            ) from exc

    def shutdown(self) -> None:
        """Compatibility alias for older hosts."""
        self.close()

    def health(self) -> RuntimeHealth:
        components: dict[str, ComponentHealth] = {}
        overall = HealthStatus.HEALTHY
        for name, value in self.capabilities.items():
            check = getattr(value, "health", None)
            try:
                result = check() if callable(check) else ComponentHealth(HealthStatus.HEALTHY)
                if isinstance(result, HealthStatus):
                    result = ComponentHealth(result)
                elif not isinstance(result, ComponentHealth):
                    result = ComponentHealth(HealthStatus(str(result)))
            except Exception as exc:
                result = ComponentHealth(HealthStatus.UNAVAILABLE, str(exc))
            components[name] = result
            required = self._capability_specs.get(name, _CapabilitySpec()).required
            if result.status is HealthStatus.UNAVAILABLE:
                overall = HealthStatus.UNAVAILABLE if required else max(
                    overall, HealthStatus.DEGRADED, key=_health_rank
                )
            elif result.status is HealthStatus.DEGRADED and overall is HealthStatus.HEALTHY:
                overall = HealthStatus.DEGRADED
        return RuntimeHealth(overall, components)

    def _ordered_lifecycle_components(self) -> list[str]:
        lifecycle_names = {
            name for name, value in self.capabilities.items()
            if any(callable(getattr(value, method, None)) for method in ("start", "flush", "close"))
        }
        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name not in lifecycle_names:
                return
            if name in visiting:
                raise ConfigurationError(
                    "capability dependency cycle", component=name, operation="validate"
                )
            visiting.add(name)
            spec = self._capability_specs.get(name, _CapabilitySpec())
            for dependency in spec.depends_on:
                if dependency not in self.capabilities:
                    raise CapabilityUnavailableError(
                        f"capability dependency is missing: {dependency}",
                        component=name,
                        operation="validate",
                    )
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for name in self.capabilities:
            visit(name)
        return result

    def _close_started_components(self) -> None:
        for name in reversed(self._started_components):
            close = getattr(self.capabilities[name], "close", None)
            if callable(close):
                close()
        self._started_components.clear()

    @staticmethod
    def _create_default_registries() -> dict[str, Any]:
        return {
            "config": LazyRegistryRef("..ai_config.ai_config", "get_ai_config"),
            "tool": LazyRegistryRef("..ai_tools.registry", "get_tool_registry"),
            "workflow": LazyRegistryRef("..ai_workflow.registry", "get_workflow_registry"),
            "workflow_command": LazyRegistryRef(
                "..ai_workflow.command_registry", "get_workflow_command_registry"
            ),
            "media": LazyRegistryRef("..ai_media_resource", "get_media_registry"),
            "conversation": LazyRegistryRef(
                "..ai_agent.conversation_store", "get_conversation_store"
            ),
            "model": LazyRegistryRef("..ai_models.base_pool", "get_pool_registry"),
        }


def _health_rank(status: HealthStatus) -> int:
    return {
        HealthStatus.HEALTHY: 0,
        HealthStatus.DEGRADED: 1,
        HealthStatus.UNAVAILABLE: 2,
    }[status]


_DEFAULT_RUNTIME: CAIRuntime | None = None


def get_default_runtime() -> CAIRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = CAIRuntime()
    return _DEFAULT_RUNTIME


def set_default_runtime(runtime: CAIRuntime | None) -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = runtime


__all__ = ["CAIRuntime", "LazyRegistryRef", "get_default_runtime", "set_default_runtime"]
