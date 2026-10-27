"""Agent-core protocol and configuration-driven factory (D012/D006).

Every entry-point mode (ACP server, one-shot CLI and interactive TUI) consumes a single
``AgentCore``: current implementations are ``LlmCore`` (MAF) and ``EchoCore`` (echo).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Protocol

from selenita.config import Settings
from selenita.sessions.store import SessionStore, new_session_id

from .chunks import CoreChunk
from .echo import EchoCore
from .llm import LlmConfigError, LlmCore
from .tools import Toolkit

logger = logging.getLogger(__name__)


class AgentCore(Protocol):
    """Contract consumed by the entry-point modes (does not change between cores)."""

    on_tool_event: Callable[[Any], None] | None
    """Live tool-execution feedback callback (None keeps the events invisible)."""

    @property
    def store(self) -> SessionStore:  # pragma: no cover - declarative protocol
        """Underlying session store."""
        ...

    def new_session_id(self) -> str:  # pragma: no cover - declarative protocol
        """Fresh session identifier (``sess_*``)."""
        ...

    async def aclose(self) -> None:  # pragma: no cover - declarative protocol
        """Release external resources (e.g. MCP tools); idempotent, safe to call twice."""
        ...

    async def attach_session_mcp(
        self, session_id: str, specs: Sequence[Any]
    ) -> None:  # pragma: no cover - declarative protocol
        """Attach client-declared MCP servers to one session (ACP ``session/new``).

        Ignored by cores without MCP support (echo); LLM-scoped servers are stdio specs.
        """
        ...

    def respond(self, session_id: str, user_text: str) -> AsyncIterator[CoreChunk]:
        """Emit the agent's streaming output as typed chunks (answer/reasoning)."""
        ...


def create_core(settings: Settings, *, toolkit: Toolkit | None = None) -> AgentCore:
    """Core according to ``SELENITA_AGENT_MODE`` ('llm' | 'echo') with graceful degradation.

    When the real core cannot be built (missing API key, unknown provider…), it falls
    back to the echo core and logs a warning: CLI/TUI/ACP keep working without breaking
    the turn (D008). ``toolkit`` describes which built-in tools the core offers and how
    approvals are collected (ignored by the echo core).
    """
    if settings.agent_mode == "echo":
        return EchoCore(settings=settings)
    try:
        return LlmCore.build(settings, toolkit=toolkit)
    except LlmConfigError as exc:
        logger.warning(
            "LLM unavailable (%s): selenita starts in echo mode. "
            "Set SELENITA_AGENT_MODE/SELENITA_LLM_PROVIDER and the credentials.",
            exc,
        )
        return EchoCore(settings=settings)


__all__ = ["AgentCore", "CoreChunk", "EchoCore", "LlmCore", "create_core", "new_session_id"]
