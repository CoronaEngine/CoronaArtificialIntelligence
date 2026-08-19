"""Capability-aware tool declarations and registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .capabilities import Capability
from .errors import CapabilityUnavailableError, ConfigurationError, QuasarRuntimeError, ToolExecutionError


def _name(capability: str | Capability) -> str:
    return capability.value if isinstance(capability, Capability) else str(capability)


class ToolAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ToolContext:
    """Read-only capability view provided to a tool factory."""

    capabilities: Mapping[str, Any]

    def __init__(self, capabilities: Mapping[str | Capability, Any]) -> None:
        normalized = {_name(key): value for key, value in capabilities.items()}
        object.__setattr__(self, "capabilities", MappingProxyType(normalized))

    def get(self, capability: str | Capability, default: Any = None) -> Any:
        return self.capabilities.get(_name(capability), default)

    def require(self, capability: str | Capability) -> Any:
        normalized = _name(capability)
        try:
            return self.capabilities[normalized]
        except KeyError as exc:
            raise CapabilityUnavailableError(
                f"capability is not configured: {normalized}",
                component=normalized,
                operation="resolve_tool_dependency",
            ) from exc


@dataclass(frozen=True)
class ToolSpec:
    name: str
    factory: Callable[[ToolContext], Any]
    requires: frozenset[str | Capability] = frozenset()
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError(
                "tool name cannot be empty", component="tool", operation="validate"
            )
        if not callable(self.factory):
            raise ConfigurationError(
                f"tool factory is not callable: {self.name}",
                component=self.name,
                operation="validate",
            )


@dataclass(frozen=True)
class ToolRegistration:
    spec: ToolSpec
    availability: ToolAvailability
    missing: frozenset[str] = frozenset()
    instance: Any = None


class CapabilityToolRegistry:
    def __init__(self, capabilities: Mapping[str | Capability, Any]) -> None:
        self._capabilities = capabilities
        self._registrations: dict[str, ToolRegistration] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolRegistration:
        if spec.name in self._registrations and not replace:
            raise ConfigurationError(
                f"tool is already registered: {spec.name}",
                component=spec.name,
                operation="register",
            )
        normalized_capabilities = {_name(key): value for key, value in self._capabilities.items()}
        missing = frozenset(
            _name(requirement)
            for requirement in spec.requires
            if _name(requirement) not in normalized_capabilities
        )
        if missing:
            registration = ToolRegistration(spec, ToolAvailability.UNAVAILABLE, missing)
        else:
            try:
                instance = spec.factory(ToolContext(normalized_capabilities))
            except QuasarRuntimeError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    f"failed to create tool '{spec.name}': {exc}",
                    component=spec.name,
                    operation="create",
                ) from exc
            registration = ToolRegistration(spec, ToolAvailability.AVAILABLE, instance=instance)
        self._registrations[spec.name] = registration
        return registration

    def get(self, name: str) -> Any:
        try:
            registration = self._registrations[name]
        except KeyError as exc:
            raise KeyError(name) from exc
        if registration.availability is ToolAvailability.UNAVAILABLE:
            raise CapabilityUnavailableError(
                f"tool is unavailable: {name}; missing: {', '.join(sorted(registration.missing))}",
                component=name,
                operation="resolve_tool",
            )
        return registration.instance

    def status(self, name: str) -> ToolRegistration:
        return self._registrations[name]

    def validate(self) -> None:
        unavailable = [
            registration for registration in self._registrations.values()
            if registration.spec.required
            and registration.availability is ToolAvailability.UNAVAILABLE
        ]
        if unavailable:
            details = "; ".join(
                f"{item.spec.name} missing {', '.join(sorted(item.missing))}"
                for item in unavailable
            )
            raise ConfigurationError(
                f"required tools are unavailable: {details}",
                component="tool_registry",
                operation="validate",
            )

    def registrations(self) -> tuple[ToolRegistration, ...]:
        return tuple(self._registrations.values())


__all__ = [
    "CapabilityToolRegistry",
    "ToolAvailability",
    "ToolContext",
    "ToolRegistration",
    "ToolSpec",
]
