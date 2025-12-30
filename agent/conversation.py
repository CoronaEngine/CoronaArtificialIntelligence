from __future__ import annotations

from typing import Dict, List, Sequence
from langchain_core.messages import BaseMessage

from tools.context import (
    get_boot_session_id,
    get_current_session,
)
from agent.conversation_store import (
    get_conversation_store,
)


# 保留旧类名以兼容，但建议使用线程安全版本
class ConversationStore:
    """
    Legacy 会话存储（非线程安全）

    警告：此类仅为向后兼容保留，多用户并发场景请使用 ThreadSafeConversationStore
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, List[BaseMessage]] = {}

    def snapshot(self, session_id: str) -> List[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def update(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        self._sessions[session_id] = list(messages)


def get_history(session_id: str) -> List[BaseMessage]:
    """获取会话历史（线程安全）"""
    return get_conversation_store().snapshot(session_id)


def update_history(session_id: str, messages: Sequence[BaseMessage]) -> None:
    """更新会话历史（线程安全）"""
    get_conversation_store().update(session_id, messages)


def default_session_id() -> str:
    return get_current_session() or get_boot_session_id()


__all__ = ["ConversationStore", "get_history", "update_history", "default_session_id"]
