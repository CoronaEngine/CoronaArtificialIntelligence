"""Legacy session-tracking facade backed by the Quasar SessionStore port."""
from __future__ import annotations

import copy
import threading
import time
import weakref
from datetime import datetime
from typing import Any, Mapping

from ...cai.capabilities import (
    Capability,
    DomainEvent,
    EventBus,
    SessionChange,
    SessionSnapshot,
    SessionStore,
)
from .models import DeadlineInfo


def _now_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time()).isoformat()


def _empty_progress() -> dict[str, Any]:
    return {
        "current_step": 0,
        "total_steps": 0,
        "step_name": "",
        "step_message": "",
        "progress_percent": 0.0,
        "is_retrying": False,
        "current_attempt": 1,
        "max_attempts": 3,
        "steps_history": [],
        "total_retries": 0,
        "estimated_remaining_seconds": None,
    }


class SessionCacheManager:
    """Compatibility API that persists every change through ``SessionStore``."""

    def __init__(
        self,
        session_store: SessionStore | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        if session_store is None or event_bus is None:
            from ...cai.runtime import get_default_runtime

            runtime = get_default_runtime()
            session_store = session_store or runtime.require_capability(Capability.SESSION_STORE)
            event_bus = event_bus or runtime.require_capability(Capability.EVENT_BUS)
        self._session_store = session_store
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._known_keys: set[str] = set()

    def init_session(
        self,
        session_id: str,
        input_type: str,
        parameters: Mapping[str, Any],
        workflow_state: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.time()
        workflow = dict(workflow_state or {})
        values = {
            "created_at": now,
            "updated_at": now,
            "input_type": input_type,
            "input_parameters": copy.deepcopy(dict(parameters)),
            "function_id": workflow.get("function_id"),
            "prompt": workflow.get("prompt", ""),
            "images": copy.deepcopy(workflow.get("images", [])),
            "additional_type": copy.deepcopy(workflow.get("additional_type")),
            "bounding_box": copy.deepcopy(workflow.get("bounding_box")),
            "resolution": workflow.get("resolution", "1:1"),
            "image_size": workflow.get("image_size", "2K"),
            "metadata": copy.deepcopy(workflow.get("metadata", {})),
            "progress": _empty_progress(),
            "outputs": [],
            "error_message": None,
            "account_usages": [],
            "deadline_info": None,
        }
        with self._lock:
            if self._session_store.get(session_id) is not None:
                self._session_store.delete(session_id)
            self._session_store.create(SessionSnapshot(session_id, "idle", values))
            self._known_keys.add(session_id)
        self._publish("session.created", session_id, {"input_type": input_type})

    def update_state(self, session_id: str, state: str) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        values["updated_at"] = time.time()
        self._save(session_id, values, state=state)
        self._publish("session.state_changed", session_id, {"state": state})

    def update_progress(
        self,
        session_id: str,
        current_step: int,
        total_steps: int,
        step_name: str,
        message: str,
        progress_percent: float | None = None,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        progress = values["progress"]
        progress.update({
            "current_step": current_step,
            "total_steps": total_steps,
            "step_name": step_name,
            "step_message": message,
            "progress_percent": progress_percent if progress_percent is not None else (
                current_step / total_steps * 100.0 if total_steps else 0.0
            ),
        })
        values["updated_at"] = time.time()
        self._save(session_id, values)
        self._publish("session.progress_updated", session_id, copy.deepcopy(progress))

    def record_step_start(
        self,
        session_id: str,
        step_name: str,
        step_number: int,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        progress = values["progress"]
        step = self._find_step(progress, step_name, step_number)
        if step is None:
            step = {
                "step": step_number, "name": step_name, "status": "running",
                "started_at": _now_iso(), "completed_at": None,
                "duration_ms": None, "retry_info": None, "metadata": {},
            }
            progress["steps_history"].append(step)
        if attempt > 1:
            step["status"] = "retrying"
            retry = step.get("retry_info") or {
                "attempt_count": attempt, "max_attempts": max_attempts,
                "last_error": None, "retry_history": [],
            }
            retry.update({"attempt_count": attempt, "max_attempts": max_attempts})
            step["retry_info"] = retry
        else:
            step.update({"status": "running", "started_at": _now_iso()})
        progress.update({
            "is_retrying": attempt > 1,
            "current_attempt": attempt,
            "max_attempts": max_attempts,
        })
        values["updated_at"] = time.time()
        self._save(session_id, values)

    def record_step_retry(
        self, session_id: str, step_name: str, step_number: int,
        error: str, next_attempt: int,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        progress = values["progress"]
        step = self._find_step(progress, step_name, step_number)
        if step is not None:
            retry = step.get("retry_info") or {
                "attempt_count": next_attempt - 1, "max_attempts": 3,
                "last_error": None, "retry_history": [],
            }
            retry["last_error"] = error
            retry["retry_history"].append({
                "attempt": next_attempt - 1, "failed_at": _now_iso(), "error": error,
            })
            step.update({"status": "retrying", "retry_info": retry})
            progress["total_retries"] += 1
        values["updated_at"] = time.time()
        self._save(session_id, values)

    def record_step_complete(
        self, session_id: str, step_name: str, step_number: int,
        success: bool, error: str | None = None,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        progress = values["progress"]
        step = self._find_step(progress, step_name, step_number)
        if step is not None:
            completed = _now_iso()
            step.update({"status": "completed" if success else "failed", "completed_at": completed})
            if step.get("started_at"):
                try:
                    start = datetime.fromisoformat(step["started_at"])
                    step["duration_ms"] = int((datetime.fromisoformat(completed) - start).total_seconds() * 1000)
                except (TypeError, ValueError):
                    pass
            if not success and error:
                retry = step.get("retry_info") or {
                    "attempt_count": 1, "max_attempts": 1, "retry_history": [],
                }
                retry["last_error"] = error
                step["retry_info"] = retry
        progress.update({"is_retrying": False, "current_attempt": 1})
        values["updated_at"] = time.time()
        self._save(session_id, values)

    def append_output(self, session_id: str, output_type: str, content: Mapping[str, Any]) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        now = time.time()
        values["outputs"].append({
            "type": output_type,
            "content": copy.deepcopy(dict(content)),
            "created_at": now,
            "created_at_iso": _now_iso(now),
        })
        values["updated_at"] = now
        self._save(session_id, values)
        self._publish("session.output_appended", session_id, {"type": output_type})

    def set_error(self, session_id: str, error: str) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        values.update({"error_message": error, "updated_at": time.time()})
        self._save(session_id, values)

    def record_account_usage(
        self, session_id: str, account_id: str, account_name: str,
        model: str | None, price: float, latency_ms: float, success: bool,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        values["account_usages"].append({
            "account_id": account_id, "account_name": account_name, "model": model,
            "timestamp": time.time(), "price": price,
            "latency_ms": round(latency_ms, 2), "success": success,
        })
        values["updated_at"] = time.time()
        self._save(session_id, values)

    def record_deadline_info(
        self, session_id: str, deadline: float, start_time: float,
        stage_timings: list, success: bool, is_timeout: bool,
    ) -> None:
        values = self._load_values(session_id)
        if values is None:
            return
        values["deadline_info"] = {
            "deadline": deadline,
            "start_time": start_time,
            "deadline_seconds": deadline - start_time,
            "is_timeout": is_timeout,
            "elapsed_ms": round((time.time() - start_time) * 1000, 2),
            "stage_timings": copy.deepcopy(stage_timings),
            "success": success,
        }
        values["updated_at"] = time.time()
        self._save(session_id, values)

    def get_deadline_info(self, session_id: str) -> DeadlineInfo | None:
        values = self._load_values(session_id)
        info = values.get("deadline_info") if values else None
        if not info:
            return None
        return DeadlineInfo(**{key: info[key] for key in (
            "deadline", "start_time", "deadline_seconds", "is_timeout",
            "elapsed_ms", "stage_timings",
        )})

    def get_status(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        values, progress = snapshot.values, snapshot.values["progress"]
        usage = values["account_usages"]
        return self._ok({
            "session_id": session_id, "state": snapshot.state,
            "created_at": values["created_at"], "updated_at": values["updated_at"],
            "current_step": progress["step_name"], "total_steps": progress["total_steps"],
            "progress_percent": round(progress["progress_percent"], 2),
            "error_message": values["error_message"],
            "total_cost": round(sum(item["price"] for item in usage if item["success"]), 4),
            "total_calls": len(usage),
            "successful_calls": sum(1 for item in usage if item["success"]),
        })

    def get_progress(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        progress = copy.deepcopy(snapshot.values["progress"])
        usage = snapshot.values["account_usages"]
        progress.update({
            "session_id": session_id,
            "total_cost": round(sum(item["price"] for item in usage if item["success"]), 4),
            "total_calls": len(usage),
            "successful_calls": sum(1 for item in usage if item["success"]),
        })
        return self._ok(progress)

    def get_input(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        values = snapshot.values
        return self._ok({
            "session_id": session_id, "input_type": values["input_type"],
            "parameters": copy.deepcopy(values["input_parameters"]),
            "submitted_at": values["created_at"],
        })

    def get_output(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        outputs = copy.deepcopy(snapshot.values["outputs"])
        status = "completed" if snapshot.state == "completed" else (
            "failed" if snapshot.state == "failed" else "partial" if outputs else "pending"
        )
        return self._ok({
            "session_id": session_id, "status": status, "outputs": outputs,
            "total_outputs": len(outputs),
            "completed_at": snapshot.values["updated_at"] if snapshot.state == "completed" else None,
        })

    def get_snapshot(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        values = copy.deepcopy(dict(snapshot.values))
        result = {
            "session_id": session_id, "state": snapshot.state,
            "created_at": _now_iso(values.pop("created_at")),
            "updated_at": _now_iso(values.pop("updated_at")),
            **values,
        }
        result["usage_summary"] = self._usage_summary(result["account_usages"])
        return self._ok(result)

    def get_accounts(self, session_id: str) -> dict[str, Any]:
        snapshot = self._session_store.get(session_id)
        if snapshot is None:
            return self._not_found()
        usage = copy.deepcopy(snapshot.values["account_usages"])
        return self._ok({
            "session_id": session_id,
            "account_usages": usage,
            "usage_summary": self._usage_summary(usage),
        })

    def exists(self, session_id: str) -> bool:
        return self._session_store.get(session_id) is not None

    def clear(self, session_id: str) -> None:
        self._session_store.delete(session_id)
        self._known_keys.discard(session_id)

    def clear_all(self) -> None:
        for session_id in tuple(self._known_keys):
            self._session_store.delete(session_id)
        self._known_keys.clear()

    def get_stats(self) -> dict[str, int]:
        snapshots = [self._session_store.get(key) for key in self._known_keys]
        existing = [snapshot for snapshot in snapshots if snapshot is not None]
        return {
            "total_sessions": len(existing),
            "running_sessions": sum(1 for snapshot in existing if snapshot.state == "running"),
            "completed_sessions": sum(1 for snapshot in existing if snapshot.state == "completed"),
            "failed_sessions": sum(1 for snapshot in existing if snapshot.state == "failed"),
        }

    def _load_values(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._session_store.get(session_id)
            return copy.deepcopy(dict(snapshot.values)) if snapshot is not None else None

    def _save(self, session_id: str, values: dict[str, Any], state: str | None = None) -> None:
        with self._lock:
            self._session_store.update(SessionChange(session_id, state=state, values=values))

    def _publish(self, name: str, session_id: str, payload: Mapping[str, Any]) -> None:
        self._event_bus.publish(DomainEvent(name, {"session_key": session_id, **payload}))

    @staticmethod
    def _find_step(progress: dict[str, Any], name: str, number: int) -> dict[str, Any] | None:
        return next((step for step in progress["steps_history"] if step["name"] == name and step["step"] == number), None)

    @staticmethod
    def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [record for record in records if record["success"]]
        return {
            "total_calls": len(records), "successful_calls": len(successful),
            "failed_calls": len(records) - len(successful),
            "total_cost": round(sum(record["price"] for record in successful), 4),
        }

    @staticmethod
    def _not_found() -> dict[str, Any]:
        return {"code": 404, "msg": "会话不存在", "data": None}

    @staticmethod
    def _ok(data: Any) -> dict[str, Any]:
        return {"code": 0, "msg": "ok", "data": data}


_MANAGERS: "weakref.WeakKeyDictionary[Any, SessionCacheManager]" = weakref.WeakKeyDictionary()
_MANAGER_LOCK = threading.Lock()


def get_session_cache_manager(runtime=None) -> SessionCacheManager:
    from ...cai.runtime import get_default_runtime

    runtime = runtime or get_default_runtime()
    with _MANAGER_LOCK:
        manager = _MANAGERS.get(runtime)
        if manager is None:
            manager = SessionCacheManager(
                session_store=runtime.require_capability(Capability.SESSION_STORE),
                event_bus=runtime.require_capability(Capability.EVENT_BUS),
            )
            _MANAGERS[runtime] = manager
        return manager


__all__ = ["SessionCacheManager", "get_session_cache_manager"]
