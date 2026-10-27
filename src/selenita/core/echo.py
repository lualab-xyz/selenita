"""Single agent core of selenita: the same logic for every entry-point mode.

The three entry points (ACP server, one-shot CLI and interactive TUI) call this module and
must stay thin adapters over it (decision D012). Sessions/history are shared, so e.g. a TUI
resume with ``--session-id`` sees the same conversation as ACP/CLI runs.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from selenita.sessions.store import (
    InMemorySessionStore,
    JsonlSessionStore,
    SessionMessage,
    SessionStore,
    new_session_id,
)

from .chunks import CoreChunk

logger = logging.getLogger(__name__)


class EchoCore:
    """Echo-phase core of selenita (MVP): receives the user text and echoes it back.

    It stores each exchange in the session store so all modes share the same history
    (e.g. TUI resume via ``--session-id``).
    """

    # No tool executions to report in the echo phase; adapters may still set this.
    on_tool_event: Callable[[Any], None] | None = None

    def __init__(self, store: SessionStore | None = None, settings: Any = None) -> None:
        if store is not None:
            self._store: SessionStore = store
        elif settings is not None:
            self._store = JsonlSessionStore.from_settings(settings)
        else:
            self._store = InMemorySessionStore()

    @property
    def store(self) -> SessionStore:
        """Underlying session store (entry points use it to build the welcome screen)."""
        return self._store

    def new_session_id(self) -> str:
        """Fresh session identifier (``sess_`` + uuid)."""
        return new_session_id()

    async def aclose(self) -> None:
        """Echo owns no external resources; nothing to close (protocol parity)."""
        return None

    async def attach_session_mcp(self, session_id: str, specs: Any) -> None:
        """MCP is an LLM-core feature; the echo core only counts the specs (parity)."""
        logger.info("echo core ignores %d MCP spec(s) for session %s", len(list(specs)), session_id)

    async def respond(self, session_id: str, user_text: str) -> AsyncIterator[CoreChunk]:
        """Process the user text and emit the response as streamed chunks.

        In the echo phase the response is the very same text as a single chunk. Entry-point
        modes consume the generator and map it to their transport (ACP notification, stdout,
        …). The full response also goes to the store for the persisted history.
        """
        if not user_text:
            return
        _persist(self._store, session_id, user_text, "user")
        _persist(self._store, session_id, user_text, "agent")
        yield CoreChunk(kind="answer", text=user_text)


def _persist(store: SessionStore, session_id: str, text: str, role: str) -> None:
    """Persist a message, degrading gracefully when the disk fails (turn must survive)."""
    try:
        store.append(session_id, SessionMessage(role=role, text=text))
    except (OSError, ValueError) as exc:
        logger.warning("could not persist %s message (session %s): %s", role, session_id, exc)
