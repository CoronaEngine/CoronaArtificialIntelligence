from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import (
    CAIRuntime,
    Capability,
    InProcessEventBus,
    MemorySessionStore,
)
from Quasar.ai_tools.session_tracking.cache import SessionCacheManager


def test_session_cache_manager_uses_injected_store_without_database_or_threads():
    store = MemorySessionStore()
    events = InProcessEventBus()
    before_threads = {thread.ident for thread in threading.enumerate()}

    manager = SessionCacheManager(session_store=store, event_bus=events)
    manager.init_session("session-1", "chat", {"message": "hello"})
    manager.update_state("session-1", "running")
    manager.update_progress("session-1", 1, 2, "prepare", "working")

    snapshot = store.get("session-1")
    assert snapshot is not None
    assert snapshot.state == "running"
    assert snapshot.values["input_parameters"] == {"message": "hello"}
    assert snapshot.values["progress"]["current_step"] == 1
    assert [event.name for event in events.events()] == [
        "session.created", "session.state_changed", "session.progress_updated"
    ]
    assert "database" not in sys.modules
    assert {thread.ident for thread in threading.enumerate()} == before_threads


def test_legacy_session_api_delegates_to_default_runtime_and_warns():
    from Quasar.cai.runtime import set_default_runtime
    from Quasar.compat.v1 import init_session, update_session_state

    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    set_default_runtime(runtime)
    try:
        with pytest.warns(DeprecationWarning):
            init_session("legacy-1", "workflow", {"prompt": "hello"})
        with pytest.warns(DeprecationWarning):
            update_session_state("legacy-1", "completed")
    finally:
        set_default_runtime(None)

    snapshot = runtime.require_capability(Capability.SESSION_STORE).get("legacy-1")
    assert snapshot is not None
    assert snapshot.state == "completed"


def test_existing_session_tracking_import_path_is_a_compatibility_facade():
    from Quasar.ai_tools.session_tracking import get_session_cache_manager, init_session
    from Quasar.cai.runtime import set_default_runtime

    runtime = CAIRuntime(ai_entrance_provider=lambda: None)
    set_default_runtime(runtime)
    try:
        with pytest.warns(DeprecationWarning):
            init_session("legacy-import", "chat", {})
        with pytest.warns(DeprecationWarning):
            assert get_session_cache_manager() is not None
    finally:
        set_default_runtime(None)

    assert runtime.require_capability(Capability.SESSION_STORE).get("legacy-import") is not None
