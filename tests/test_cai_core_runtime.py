from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import (
    AdapterConnectionError,
    ArtifactInput,
    BufferFlushError,
    CAIRuntime,
    Capability,
    ComponentHealth,
    ConfigurationError,
    ConversationSnapshot,
    DomainEvent,
    HealthStatus,
    HostIntegrationError,
    MemoryArtifactStore,
    MemoryConversationStore,
    MemorySessionStore,
    SessionChange,
    SessionSnapshot,
)


def test_default_runtime_uses_functional_in_memory_capabilities():
    before = {thread.ident for thread in threading.enumerate()}
    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    assert isinstance(runtime.require_capability(Capability.CONVERSATION_STORE), MemoryConversationStore)
    assert isinstance(runtime.require_capability(Capability.SESSION_STORE), MemorySessionStore)
    assert isinstance(runtime.require_capability(Capability.ARTIFACT_STORE), MemoryArtifactStore)
    assert {thread.ident for thread in threading.enumerate()} == before


def test_core_import_does_not_load_optional_packages():
    repository = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(repository)!r})
import Quasar.cai
forbidden = ('yaml', 'sqlalchemy', 'redis', 'pymongo', 'sqlite_vec', 'torch', 'PIL', 'httpx')
loaded = [name for name in forbidden if name in sys.modules]
assert loaded == [], loaded
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_memory_capabilities_implement_public_contracts():
    conversations = MemoryConversationStore()
    conversations.save(ConversationSnapshot("conversation-1", ("hello",)))
    assert conversations.load("conversation-1").messages == ("hello",)
    assert conversations.delete("conversation-1") is True

    sessions = MemorySessionStore()
    sessions.create(SessionSnapshot("session-1", state="created"))
    sessions.update(SessionChange("session-1", state="running", values={"step": 2}))
    assert sessions.get("session-1") == SessionSnapshot("session-1", "running", {"step": 2})

    artifacts = MemoryArtifactStore()
    reference = artifacts.put(ArtifactInput(b"content", "result.txt", "text/plain"))
    assert artifacts.open(reference).read() == b"content"
    assert artifacts.delete(reference) is True


def test_runtime_lifecycle_is_dependency_ordered_and_idempotent():
    events: list[str] = []

    class Component:
        def __init__(self, name: str): self.name = name
        def start(self): events.append(f"start:{self.name}")
        def flush(self, timeout=None): events.append(f"flush:{self.name}")
        def close(self): events.append(f"close:{self.name}")

    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    runtime.set_capability("storage", Component("storage"))
    runtime.set_capability("buffer", Component("buffer"), depends_on=("storage",))
    runtime.start()
    runtime.close()
    runtime.close()
    assert events == [
        "start:storage", "start:buffer", "flush:buffer", "flush:storage",
        "close:buffer", "close:storage",
    ]


def test_capability_replacement_must_be_explicit():
    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    with pytest.raises(ConfigurationError):
        runtime.set_capability(Capability.SESSION_STORE, MemorySessionStore())
    replacement = MemorySessionStore()
    runtime.set_capability(Capability.SESSION_STORE, replacement, replace=True)
    assert runtime.require_capability(Capability.SESSION_STORE) is replacement


def test_flush_failure_closes_component_and_raises_structured_error():
    events: list[str] = []

    class Component:
        def start(self): events.append("start")
        def flush(self, timeout=None):
            events.append("flush")
            raise RuntimeError("write failed")
        def close(self): events.append("close")

    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    runtime.set_capability("buffer", Component())
    runtime.start()
    with pytest.raises(BufferFlushError) as raised:
        runtime.close(timeout=1)
    assert raised.value.component == "buffer"
    assert runtime.state == "closed"
    assert events == ["start", "flush", "close"]


def test_health_aggregates_required_and_optional_components():
    class Unavailable:
        def health(self): return ComponentHealth(HealthStatus.UNAVAILABLE, "not configured")

    optional = CAIRuntime(ai_entrance_provider=lambda: None)
    optional.set_capability("optional", Unavailable(), required=False)
    assert optional.health().status is HealthStatus.DEGRADED

    required = CAIRuntime(ai_entrance_provider=lambda: None)
    required.set_capability("required", Unavailable(), required=True)
    assert required.health().status is HealthStatus.UNAVAILABLE


def test_closed_runtime_rejects_requests():
    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    runtime.close()
    with pytest.raises(HostIntegrationError):
        list(runtime.chat_stream({"message": "hello"}))


def test_runtime_errors_are_transport_neutral():
    error = AdapterConnectionError(
        "cannot connect", component="session_store", operation="start",
        retryable=True, trace_key="trace-1",
    )
    assert error.to_dict() == {
        "code": "adapter_connection_error", "message": "cannot connect",
        "component": "session_store", "operation": "start", "retryable": True,
        "trace_key": "trace-1",
    }


def test_event_bus_records_domain_events():
    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    event_bus = runtime.require_capability(Capability.EVENT_BUS)
    event = DomainEvent("session.updated", {"state": "running"}, trace_key="trace-1")
    event_bus.publish(event)
    assert event_bus.events() == (event,)
