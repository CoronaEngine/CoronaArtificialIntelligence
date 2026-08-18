"""One-generation compatibility facade for legacy session tracking."""
from __future__ import annotations

import warnings
from typing import Any

from ...cai.config import RuntimeConfig


class LegacyConfigAdapter:
    """Translate the supported legacy dictionary shape into RuntimeConfig."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RuntimeConfig:
        warnings.warn(
            "LegacyConfigAdapter is deprecated; construct RuntimeConfig explicitly",
            DeprecationWarning,
            stacklevel=2,
        )
        chat = data.get("chat") if isinstance(data.get("chat"), dict) else {}
        runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
        values: dict[str, Any] = {}
        request_timeout = chat.get("request_timeout", data.get("request_timeout"))
        if request_timeout is not None:
            values["request_timeout"] = request_timeout
        shutdown_timeout = runtime.get("shutdown_timeout", data.get("shutdown_timeout"))
        if shutdown_timeout is not None:
            values["shutdown_timeout"] = shutdown_timeout
        max_concurrency = runtime.get(
            "max_concurrency",
            runtime.get("max_workers", data.get("max_concurrency", data.get("max_workers"))),
        )
        if max_concurrency is not None:
            values["max_concurrency"] = max_concurrency
        for key in ("log_level", "persistence_policy"):
            value = runtime.get(key, data.get(key))
            if value is not None:
                values[key] = value
        return RuntimeConfig(**values)

def _manager():
    return _resolve_manager()


def _resolve_manager(runtime=None):
    from ...ai_tools.session_tracking.cache import get_session_cache_manager as resolve

    return resolve(runtime)


def get_session_cache_manager(runtime=None):
    _warn("get_session_cache_manager")
    return _resolve_manager(runtime)


def _warn(name: str) -> None:
    warnings.warn(
        f"Quasar legacy session API '{name}' is deprecated; use the Runtime SessionStore capability",
        DeprecationWarning,
        stacklevel=2,
    )


def init_session(session_id: str, input_type: str, parameters: dict,
                 workflow_state: dict | None = None) -> None:
    _warn("init_session")
    _manager().init_session(session_id, input_type, parameters, workflow_state)


def update_session_state(session_id: str, state: str) -> None:
    _warn("update_session_state")
    _manager().update_state(session_id, state)


def update_session_progress(session_id: str, current_step: int, total_steps: int,
                            step_name: str, message: str,
                            progress_percent: float | None = None) -> None:
    _warn("update_session_progress")
    _manager().update_progress(session_id, current_step, total_steps, step_name, message, progress_percent)


def record_step_start(session_id: str, step_name: str, step_number: int,
                      attempt: int = 1, max_attempts: int = 3) -> None:
    _warn("record_step_start")
    _manager().record_step_start(session_id, step_name, step_number, attempt, max_attempts)


def record_step_retry(session_id: str, step_name: str, step_number: int,
                      error: str, next_attempt: int) -> None:
    _warn("record_step_retry")
    _manager().record_step_retry(session_id, step_name, step_number, error, next_attempt)


def record_step_complete(session_id: str, step_name: str, step_number: int,
                         success: bool, error: str | None = None) -> None:
    _warn("record_step_complete")
    _manager().record_step_complete(session_id, step_name, step_number, success, error)


def append_session_output(session_id: str, output_type: str, content: dict) -> None:
    _warn("append_session_output")
    _manager().append_output(session_id, output_type, content)


def set_session_error(session_id: str, error: str) -> None:
    _warn("set_session_error")
    _manager().set_error(session_id, error)


def record_account_usage_to_session(
    session_id: str, account_id: str, account_name: str, model: str | None,
    price: float, latency_ms: float, success: bool,
) -> None:
    _warn("record_account_usage_to_session")
    _manager().record_account_usage(
        session_id, account_id, account_name, model, price, latency_ms, success
    )


def record_deadline_info(session_id: str, deadline: float, start_time: float,
                         stage_timings: list, success: bool, is_timeout: bool) -> None:
    _warn("record_deadline_info")
    _manager().record_deadline_info(
        session_id, deadline, start_time, stage_timings, success, is_timeout
    )


def get_deadline_info(session_id: str):
    _warn("get_deadline_info")
    return _manager().get_deadline_info(session_id)


def get_session_status(session_id: str) -> dict[str, Any]:
    _warn("get_session_status")
    return _manager().get_status(session_id)


def get_session_progress(session_id: str) -> dict[str, Any]:
    _warn("get_session_progress")
    return _manager().get_progress(session_id)


def get_session_input(session_id: str) -> dict[str, Any]:
    _warn("get_session_input")
    return _manager().get_input(session_id)


def get_session_output(session_id: str) -> dict[str, Any]:
    _warn("get_session_output")
    return _manager().get_output(session_id)


def get_session_snapshot(session_id: str) -> dict[str, Any]:
    _warn("get_session_snapshot")
    return _manager().get_snapshot(session_id)


def get_session_accounts(session_id: str) -> dict[str, Any]:
    _warn("get_session_accounts")
    return _manager().get_accounts(session_id)


__all__ = [
    "LegacyConfigAdapter",
    "init_session", "update_session_state", "update_session_progress",
    "record_step_start", "record_step_retry", "record_step_complete",
    "append_session_output", "set_session_error", "record_account_usage_to_session",
    "record_deadline_info", "get_deadline_info", "get_session_status",
    "get_session_progress", "get_session_input", "get_session_output",
    "get_session_snapshot", "get_session_accounts", "get_session_cache_manager",
]
