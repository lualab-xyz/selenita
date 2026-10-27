"""One-shot CLI mode: a single run (``selenita "prompt"``).

Thin adapter over the core (D012): streams the core output — reasoning to stderr (muted,
visible in TTYs), answer text to stdout — then prints the ``session-id`` so the user can
reuse the session later (TUI or a future ACP ``session/load``).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any

from selenita.config import Settings
from selenita.core import AgentCore, create_core
from selenita.core.permissions import (
    DECISION_ALWAYS,
    DECISION_APPROVE,
    DECISION_REJECT,
    ToolEvent,
)
from selenita.core.tools import Toolkit, summarize_tool_call
from selenita.sessions.store import validate_session_id


def run_terminal_prompt(prompt: str, session_id: str | None, settings: Settings) -> int:
    """Run one conversation turn and return the process exit code."""
    try:
        resolved_id = validate_session_id(session_id) if session_id else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Mutating built-in tools need a confirmation; over a non-TTY stdin there is no
    # one to ask, so those tools are simply not offered to the model (tools epic).
    is_tty = sys.stdin.isatty()
    asker = _terminal_asker() if is_tty else None
    toolkit = Toolkit(include_mutating=is_tty, include_ask_user=False, asker=asker)
    core = create_core(settings, toolkit=toolkit)
    core.on_tool_event = _tool_event_writer
    resolved_id = resolved_id or core.new_session_id()

    async def _run_and_close() -> str:
        try:
            return await _turn(core, resolved_id, prompt)
        finally:
            await core.aclose()

    last_char = asyncio.run(_run_and_close())
    # Close the streamed line only when the core output does not end with a newline.
    if last_char and last_char != "\n":
        sys.stdout.write("\n")
    print(f"[session: {resolved_id}]")
    return 0


def _tool_event_writer(event: ToolEvent) -> None:
    """Plain tool-execution blocks on stderr (``$`` name, arguments, ✓/✗)."""

    if event.kind == "call":
        stream = f"$ {event.title}\n"
        if event.detail:
            stream += f"  {event.detail}\n"
        sys.stderr.write(stream)
    else:
        glyph = "✓" if event.status == "done" else "✗"
        label = event.status or ""
        if event.status == "done" and event.duration is not None:
            label = f"{event.duration:.1f}s"
        sys.stderr.write(f"  {glyph} {label}\n")
    sys.stderr.flush()


def _terminal_asker() -> Any:
    """Permission asker for the one-shot CLI: prompts on stderr, reads stdin.

    Blocking stdin reads run in a worker thread so the MAF event loop is not frozen.
    The tool name is one line; its arguments — the full command when applicable — are
    echoed on the line below, so long calls stay readable.
    """

    async def asker(tool_name: str, arguments: Mapping[str, Any]) -> str:
        detail = summarize_tool_call(tool_name, arguments)
        sys.stderr.write(f"\n{tool_name} needs your permission\n  {detail}\n")
        sys.stderr.write("  [y]es · [a]lways · [n]o > ")
        sys.stderr.flush()
        answer = await asyncio.to_thread(input, "")
        choice = (answer or "").strip().lower()
        if choice in ("a", "always"):
            return DECISION_ALWAYS
        if choice in ("y", "yes", ""):
            return DECISION_APPROVE
        return DECISION_REJECT

    return asker


async def _turn(core: AgentCore, session_id: str, user_text: str) -> str:
    """Consume the core chunks and stream them to the right channels."""
    close_reasoning = False  # reasoning streams on stderr; close the line when answer starts
    return_char = ""
    async for chunk in core.respond(session_id, user_text):
        return_char = chunk.text[-1] if chunk.text else return_char
        if chunk.kind == "reasoning":
            close_reasoning = True
            stream = f"\x1b[2m{chunk.text}\x1b[0m" if sys.stderr.isatty() else chunk.text
            sys.stderr.write(stream)
            sys.stderr.flush()
        elif close_reasoning:
            # Close the reasoning line and leave a blank gap before the answer.
            sys.stderr.write("\n\n")
            close_reasoning = False
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        else:
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
    return return_char
