from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import (
    Capability,
    CapabilityToolRegistry,
    ConfigurationError,
    RuntimeBuilder,
    RuntimeConfig,
    ToolAvailability,
    ToolExecutionError,
    ToolSpec,
)


def test_runtime_config_is_typed_and_validated():
    config = RuntimeConfig(request_timeout=12.5, max_concurrency=4, log_level="debug")

    assert config.request_timeout == 12.5
    assert config.max_concurrency == 4
    assert config.log_level == "DEBUG"
    with pytest.raises(ConfigurationError):
        RuntimeConfig(max_concurrency=0)


def test_legacy_config_adapter_maps_old_runtime_keys_with_a_warning():
    from Quasar.compat.v1 import LegacyConfigAdapter

    with pytest.warns(DeprecationWarning):
        config = LegacyConfigAdapter.from_dict({
            "chat": {"request_timeout": 25},
            "runtime": {"max_workers": 3, "shutdown_timeout": 7},
            "log_level": "warning",
        })

    assert config == RuntimeConfig(
        request_timeout=25,
        shutdown_timeout=7,
        max_concurrency=3,
        log_level="WARNING",
    )


def test_optional_tool_is_not_instantiated_when_a_capability_is_missing():
    created = []
    registry = CapabilityToolRegistry({})
    spec = ToolSpec(
        name="recognize_object",
        factory=lambda context: created.append(context) or object(),
        requires=frozenset({Capability.VECTOR_STORE}),
    )

    registration = registry.register(spec)

    assert registration.availability is ToolAvailability.UNAVAILABLE
    assert registration.missing == frozenset({Capability.VECTOR_STORE.value})
    assert registration.instance is None
    assert created == []


def test_required_tool_fails_validation_and_duplicate_replacement_is_explicit():
    registry = CapabilityToolRegistry({})
    spec = ToolSpec(
        name="required_tool",
        factory=lambda context: object(),
        requires=frozenset({Capability.MODEL}),
        required=True,
    )
    registry.register(spec)

    with pytest.raises(ConfigurationError, match="required_tool"):
        registry.validate()
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(spec)

    replacement = ToolSpec("required_tool", lambda context: "replacement")
    assert registry.register(replacement, replace=True).instance == "replacement"
    registry.validate()


def test_tool_factory_only_reads_explicit_context_capabilities():
    model = object()
    registry = CapabilityToolRegistry({Capability.MODEL.value: model})
    registration = registry.register(
        ToolSpec("summarize", lambda context: context.require(Capability.MODEL))
    )

    assert registration.instance is model
    assert registration.availability is ToolAvailability.AVAILABLE


def test_tool_factory_failure_is_reported_as_a_structured_runtime_error():
    def fail(context):
        raise RuntimeError("provider failed")

    registry = CapabilityToolRegistry({})

    with pytest.raises(ToolExecutionError) as raised:
        registry.register(ToolSpec("broken", fail))

    assert raised.value.component == "broken"
    assert raised.value.operation == "create"


def test_runtime_builder_assembles_capabilities_tools_and_plugins_without_starting_io():
    model = object()
    session_store = object()

    class Plugin:
        name = "sample"
        enabled = True

        def register(self, runtime):
            runtime.metadata["plugin_registered"] = True

    runtime = (
        RuntimeBuilder(RuntimeConfig(max_concurrency=2))
        .use_core_defaults()
        .use_model(model)
        .use_session_store(session_store)
        .add_tool(ToolSpec("model_tool", lambda context: context.require(Capability.MODEL)))
        .install(Plugin())
        .build()
    )

    assert runtime.state == "new"
    assert runtime.require_capability(Capability.MODEL) is model
    assert runtime.require_capability(Capability.SESSION_STORE) is session_store
    assert runtime.require_capability(Capability.CONFIG).max_concurrency == 2
    assert runtime.get_registry("tool").get("model_tool") is model
    assert runtime.metadata["plugin_registered"] is True
