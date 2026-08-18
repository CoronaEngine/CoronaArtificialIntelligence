"""Host-independent capability protocols and in-memory defaults."""
from __future__ import annotations

import io
import secrets
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence, runtime_checkable


class Capability(str, Enum):
    MODEL = "model"
    TOOL = "tool"
    WORKFLOW = "workflow"
    CONFIG = "config"
    CONVERSATION_STORE = "conversation_store"
    SESSION_STORE = "session_store"
    ARTIFACT_STORE = "artifact_store"
    VECTOR_STORE = "vector_store"
    EVENT_BUS = "event_bus"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ComponentHealth:
    status: HealthStatus
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeHealth:
    status: HealthStatus
    components: Mapping[str, ComponentHealth] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationSnapshot:
    conversation_key: str
    messages: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSnapshot:
    session_key: str
    state: str = "created"
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionChange:
    session_key: str
    state: str | None = None
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactInput:
    content: bytes
    name: str | None = None
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    name: str | None = None
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class VectorRecord:
    key: str
    vector: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorQuery:
    vector: tuple[float, ...]
    limit: int = 10


@dataclass(frozen=True)
class VectorMatch:
    key: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainEvent:
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    trace_key: str | None = None


@runtime_checkable
class ConversationStore(Protocol):
    def load(self, conversation_key: str) -> ConversationSnapshot: ...
    def save(self, snapshot: ConversationSnapshot) -> None: ...
    def delete(self, conversation_key: str) -> bool: ...


@runtime_checkable
class SessionStore(Protocol):
    def create(self, session: SessionSnapshot) -> None: ...
    def get(self, session_key: str) -> SessionSnapshot | None: ...
    def update(self, change: SessionChange) -> None: ...
    def delete(self, session_key: str) -> bool: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, artifact: ArtifactInput) -> ArtifactRef: ...
    def open(self, artifact_ref: ArtifactRef) -> BinaryIO: ...
    def delete(self, artifact_ref: ArtifactRef) -> bool: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> None: ...
    def search(self, query: VectorQuery) -> Sequence[VectorMatch]: ...
    def delete(self, record_keys: Sequence[str]) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    def publish(self, event: DomainEvent) -> None: ...


@runtime_checkable
class LifecycleComponent(Protocol):
    def start(self) -> None: ...
    def flush(self, timeout: float | None = None) -> None: ...
    def close(self) -> None: ...


class MemoryConversationStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, ConversationSnapshot] = {}
        self._lock = threading.RLock()

    def load(self, conversation_key: str) -> ConversationSnapshot:
        with self._lock:
            return self._snapshots.get(conversation_key, ConversationSnapshot(conversation_key))

    def save(self, snapshot: ConversationSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.conversation_key] = snapshot

    def delete(self, conversation_key: str) -> bool:
        with self._lock:
            return self._snapshots.pop(conversation_key, None) is not None


class MemorySessionStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, SessionSnapshot] = {}
        self._lock = threading.RLock()

    def create(self, session: SessionSnapshot) -> None:
        with self._lock:
            if session.session_key in self._snapshots:
                raise ValueError(f"session already exists: {session.session_key}")
            self._snapshots[session.session_key] = session

    def get(self, session_key: str) -> SessionSnapshot | None:
        with self._lock:
            return self._snapshots.get(session_key)

    def update(self, change: SessionChange) -> None:
        with self._lock:
            current = self._snapshots.get(change.session_key)
            if current is None:
                raise KeyError(change.session_key)
            values = {**current.values, **change.values}
            self._snapshots[change.session_key] = replace(
                current,
                state=change.state if change.state is not None else current.state,
                values=values,
            )

    def delete(self, session_key: str) -> bool:
        with self._lock:
            return self._snapshots.pop(session_key, None) is not None


class MemoryArtifactStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def put(self, artifact: ArtifactInput) -> ArtifactRef:
        reference = ArtifactRef(secrets.token_urlsafe(24), artifact.name, artifact.media_type)
        with self._lock:
            self._artifacts[reference.key] = bytes(artifact.content)
        return reference

    def open(self, artifact_ref: ArtifactRef) -> BinaryIO:
        with self._lock:
            try:
                content = self._artifacts[artifact_ref.key]
            except KeyError as exc:
                raise FileNotFoundError(artifact_ref.key) from exc
        return io.BytesIO(content)

    def delete(self, artifact_ref: ArtifactRef) -> bool:
        with self._lock:
            return self._artifacts.pop(artifact_ref.key, None) is not None


class InProcessEventBus:
    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._subscribers: list[Callable[[DomainEvent], None]] = []
        self._lock = threading.RLock()

    def publish(self, event: DomainEvent) -> None:
        with self._lock:
            self._events.append(event)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber(event)

    def subscribe(self, subscriber: Callable[[DomainEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(subscriber)

    def events(self) -> tuple[DomainEvent, ...]:
        with self._lock:
            return tuple(self._events)


def create_core_defaults() -> dict[str, Any]:
    return {
        Capability.CONVERSATION_STORE.value: MemoryConversationStore(),
        Capability.SESSION_STORE.value: MemorySessionStore(),
        Capability.ARTIFACT_STORE.value: MemoryArtifactStore(),
        Capability.EVENT_BUS.value: InProcessEventBus(),
    }


__all__ = [
    "Capability", "HealthStatus", "ComponentHealth", "RuntimeHealth",
    "ConversationSnapshot", "SessionSnapshot", "SessionChange",
    "ArtifactInput", "ArtifactRef", "VectorRecord", "VectorQuery", "VectorMatch",
    "DomainEvent", "ConversationStore", "SessionStore", "ArtifactStore", "VectorStore",
    "EventBus", "LifecycleComponent", "MemoryConversationStore", "MemorySessionStore",
    "MemoryArtifactStore", "InProcessEventBus", "create_core_defaults",
]
