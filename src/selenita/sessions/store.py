"""Persistencia ligera de sesiones de selenita (mensajes user/agent).

For the echo a flat JSONL history suffices (one line per message). Future tasks
(``sessions-persistence``) will extend it with thread metadata, tool calls and ACP replay.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# IDs generados por selenita: ``sess_`` + hex uuid; aceptamos la misma forma validada por CLI.
_SESSION_ID_RE = re.compile(r"^sess_[A-Za-z0-9_-]{4,}$")


def new_session_id() -> str:
    """Generate a valid ACP session identifier (``sess_`` + hex uuid)."""
    return f"sess_{uuid.uuid4().hex}"


def validate_session_id(session_id: str) -> str:
    """Lanza ``ValueError`` si el id no tiene la forma esperada (protege rutas del store)."""
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"invalid session_id: {session_id!r} (expected: sess_<alphanumeric chars>)"
        )
    return session_id


@dataclass(frozen=True)
class SessionMessage:
    """One conversation message: what the user says or the agent answers."""

    role: str  # "user" | "agent"
    text: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.role not in ("user", "agent"):
            raise ValueError(f"invalid role: {self.role!r}")


class SessionStore(Protocol):
    """Minimal contract the core needs from any session backend."""

    def exists(self, session_id: str) -> bool:
        """Whether there is saved history for the session."""
        ...

    def load(self, session_id: str) -> list[SessionMessage]:
        """Devuelve el historial completo en orden."""
        ...

    def append(self, session_id: str, message: SessionMessage) -> None:
        """Append a message to the session history (creating it when missing)."""
        ...


class InMemorySessionStore:
    """Store en memoria: base para tests unitarios y para ejecuciones desechables."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[SessionMessage]] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def load(self, session_id: str) -> list[SessionMessage]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, message: SessionMessage) -> None:
        self._sessions.setdefault(session_id, []).append(message)


class JsonlSessionStore:
    """On-disk store: one JSONL file per session inside the configured directory.

    Each line is JSON with role/text/created_at. Operations re-read the file so runs of
    previous binary executions still show (``--session-id`` / ``session/load``).
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = sessions_dir

    @classmethod
    def from_settings(cls, settings: Any) -> JsonlSessionStore:
        return cls(settings.sessions_dir)

    def _path(self, session_id: str) -> Path:
        validate_session_id(session_id)
        return self._dir / f"{session_id}.jsonl"

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).is_file()

    def load(self, session_id: str) -> list[SessionMessage]:
        path = self._path(session_id)
        if not path.is_file():
            return []
        messages: list[SessionMessage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            messages.append(SessionMessage(role=record["role"], text=record["text"]))
        return messages

    def append(self, session_id: str, message: SessionMessage) -> None:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
