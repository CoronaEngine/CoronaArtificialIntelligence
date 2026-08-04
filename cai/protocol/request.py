from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatRequest:
    payload: dict[str, Any]
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, request: "ChatRequest | dict[str, Any]") -> "ChatRequest":
        if isinstance(request, cls):
            return request
        if not isinstance(request, dict):
            raise TypeError("ChatRequest requires a dict or ChatRequest")
        return cls.from_legacy(request)

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> "ChatRequest":
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        session_id = payload.get("session_id")
        return cls(
            payload=dict(payload),
            session_id=session_id if isinstance(session_id, str) else None,
            metadata=dict(metadata),
        )

    @classmethod
    def from_flask_request(cls, flask_request) -> "ChatRequest":
        """Build a request from a Flask request without depending on Flask."""
        data = flask_request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise TypeError("Chat request JSON body must be an object")
        return cls.from_legacy(data)

    @classmethod
    def from_text(
        cls,
        text: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatRequest":
        payload = {
            "session_id": session_id,
            "llm_content": [
                {
                    "role": "user",
                    "interface_type": "integrated",
                    "part": [
                        {
                            "content_type": "text",
                            "content_text": text,
                        }
                    ],
                }
            ],
            "metadata": metadata or {},
        }
        return cls.from_legacy(payload)

    def to_legacy_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(self.metadata)
        payload["metadata"] = metadata
        return payload

    @property
    def message(self) -> str:
        """Return the conventional text field used by host applications."""
        value = self.payload.get("message", "")
        if isinstance(value, str) and value.strip():
            return value.strip()

        # Also support Quasar's structured integrated-message format.
        for entry in self.payload.get("llm_content", []) or []:
            for part in entry.get("part", []) if isinstance(entry, dict) else []:
                text = part.get("content_text") if isinstance(part, dict) else None
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    @property
    def course_id(self) -> Any:
        """Return an optional host-specific course context identifier."""
        return self.payload.get("course_id")
