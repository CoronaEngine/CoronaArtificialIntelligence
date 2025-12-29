from __future__ import annotations

from contextvars import ContextVar, Token
from uuid import uuid4

_BOOT_SESSION_ID = uuid4().hex
_SESSION_CONTEXT: ContextVar[str] = ContextVar("ai_session_id", default=_BOOT_SESSION_ID)


def set_current_session(session_id: str) -> Token:
    return _SESSION_CONTEXT.set(session_id or "default")


def reset_current_session(token: Token) -> None:
    _SESSION_CONTEXT.reset(token)


def get_current_session() -> str:
    return _SESSION_CONTEXT.get()


def get_boot_session_id() -> str:
    return _BOOT_SESSION_ID


__all__ = ["set_current_session", "reset_current_session", "get_current_session", "get_boot_session_id"]
