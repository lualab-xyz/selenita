"""Typed streaming chunk emitted by every agent-core implementation.

Adapters translate these chunks into their own transports: ACP maps ``answer`` to
``agent_message_chunk`` and ``reasoning`` to ``agent_thought_chunk``; the CLI/TUI render
them with different styling (reasoning in a muted style, answers with the ¤ marker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["answer", "reasoning"]


@dataclass(frozen=True)
class CoreChunk:
    """One piece of the agent's streaming output, labeled by kind."""

    kind: ChunkKind
    text: str
