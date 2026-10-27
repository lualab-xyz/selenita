"""Built-in tool permission policy and MAF middleware gate (tools epic).

Selenita ships a few of its own tools (see ``core/tools.py``). Read-only ones run
without confirmation; everything else goes through an :class:`ApprovalPolicy`:

- The policy remembers tools the user approved *always* for this process. For
  ``run_command`` the decision is scoped to the exact command: "always" for ``ls``
  must never silently approve ``rm -rf``.
- Any other tool is sent to an *asker* callback provided by the adapter (the TUI
  draws a prompt, the one-shot CLI reads stdin). When no asker is wired the call is
  denied fail-closed (e.g. an ACP transport without a permission channel yet).

The gate itself is an MAF :class:`~agent_framework.FunctionMiddleware`, the framework's
official extension point for tool policy. Unlike ``ToolApprovalMiddleware`` it does not
need an ``AgentSession``, so it works with our stateless core (session=None, history
replayed from our store, D009/D014).

Permission decisions never short-circuit the MAF function loop: a denied call sets
``context.result`` to a clear message and the model sees it and adapts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import agent_framework as af
from pydantic import BaseModel

from .tools import DETAIL_LIMIT, summarize_tool_call, tool_title

logger = logging.getLogger(__name__)


def _result_to_text(result: object) -> str:
    """Flatten a tool result to plain text.

    MAF hands ``context.result`` back as a list of ``Content`` objects (even for plain
    string-returning tools); flatten them to their text so the adapters show the tool's
    real output instead of a ``Content`` repr.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, Sequence) and not isinstance(result, (bytes, bytearray)):
        texts = [getattr(item, "text", None) for item in result]
        joined = "\n".join(t for t in texts if isinstance(t, str) and t)
        if joined:
            return joined
    return str(result)

# Tools that run without asking the user: read-only operations plus ``ask_user``,
# whose whole purpose is the user's own input (asking permission for it would be a
# prompt inside a prompt).
SAFE_TOOLS = frozenset({"selenita_read_file", "selenita_web_fetch", "selenita_ask_user"})

# Name prefix of selenita's own built-in tools; anything that does not carry it is a
# client-provided tool (e.g. ACP-declared MCP servers — trusted, the client attached it).
_BUILTIN_PREFIX = "selenita_"

# Decision strings the asker may return.
DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
DECISION_ALWAYS = "always"

# Asker callback: (tool_name, arguments) -> one of the DECISION_* values.
Asker = Callable[[str, Mapping[str, Any]], Awaitable[str]]

_RUN_TOOL = "selenita_run_command"

# Cap of the result text carried in a result ToolEvent (adapters may show it; ACP maps
# it to ``rawOutput``). Big command outputs must not flood the UI or the wire.
_OUTPUT_LIMIT = 4_000


def always_key(tool_name: str, arguments: Mapping[str, Any] | None = None) -> str:
    """Key remembering an "always" decision: the whole tool, or one exact command.

    ``run_command`` is the only tool where a tool-wide "always" would be dangerous
    (approving ``git status`` once must never auto-approve ``rm -rf /``), so its key
    embeds the command itself; everything else is remembered per tool name.
    """
    if tool_name == _RUN_TOOL:
        command = str((arguments or {}).get("command", "")).strip()
        return f"{_RUN_TOOL}:{command}"
    return tool_name


# Live event emitted for every tool execution: ``call`` right before it runs (after the
# permission decision), ``result`` when it settles. Adapters that wire ``on_tool_event``
# render them as compact ``$`` blocks (Copilot-CLI-style) or as ACP ``tool_call``/
# ``tool_call_update`` session updates.
@dataclass(frozen=True)
class ToolEvent:
    """One tool-execution step shown in the adapters' output."""

    kind: Literal["call", "result"]
    tool: str
    title: str
    detail: str | None = None  # compact arguments summary (call)
    status: str | None = None  # "done" | "denied" | "failed: …" (result)
    duration: float | None = None  # seconds (result)
    arguments: Mapping[str, Any] | None = None  # raw call arguments (call, ACP rawInput)
    output: str | None = None  # result text, capped (result, ACP rawOutput)


class ApprovalPolicy:
    """In-process rules deciding whether a built-in tool runs without asking.

    - ``SAFE_TOOLS`` run automatically (read-only operations and ``ask_user``).
    - Tools the user approved "always" run automatically too; for ``run_command`` the
      decision is scoped to the exact command (:func:`always_key`).
    - Anything else asks the ``asker``; with no asker wired it is denied fail-closed.
    """

    def __init__(self, asker: Asker | None = None) -> None:
        self._asker = asker
        self._always_approved: set[str] = set()

    @property
    def asker(self) -> Asker | None:
        """The callback the adapters wire to prompt the user (or None)."""
        return self._asker

    @asker.setter
    def asker(self, value: Asker | None) -> None:
        self._asker = value

    def is_safe(self, tool_name: str) -> bool:
        return tool_name in SAFE_TOOLS

    def is_always_approved(
        self, tool_name: str, arguments: Mapping[str, Any] | None = None
    ) -> bool:
        return always_key(tool_name, arguments) in self._always_approved

    def remember_always(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> None:
        self._always_approved.add(always_key(tool_name, arguments))

    async def ask(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        """Ask the user (when possible); a missing asker denies the call."""
        if self._asker is None:
            logger.warning("no permission asker wired: denying %s call", tool_name)
            return DECISION_REJECT
        return await self._asker(tool_name, dict(arguments))


class ApprovalMiddleware(af.FunctionMiddleware):
    """Gate on built-in tool calls decided by an :class:`ApprovalPolicy`.

    Safe and always-approved tools run through; everything else prompts the user (when
    an asker exists) or is denied. Denial short-circuits by setting ``context.result``
    without calling ``call_next()``: the function never executes and MAF feeds the
    message back to the model, which keeps going.

    Optional ``emit`` callback receives a :class:`ToolEvent` before and after every
    execution (call/result), so adapters can show what the agent is doing. The events
    must stay harmless: they are rendered, not enforced.

    Scope: only selenita's own built-in tools (``selenita_*`` prefix) are gated. Tools the
    client attaches itself (e.g. the MCP servers declared in an ACP ``session/new``) run
    without a permission prompt — the client chose to attach them — but their execution is
    still reported through the same events so adapters can show the work.
    """

    def __init__(
        self,
        policy: ApprovalPolicy,
        emit: Callable[[ToolEvent], None] | None = None,
    ) -> None:
        self._policy = policy
        self._emit = emit

    async def process(
        self,
        context: af.FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        tool = context.function.name
        # ``context.arguments`` can be a validated pydantic model or a plain mapping
        # depending on the provider; normalize so asker and detail always see a dict.
        arguments = context.arguments
        arg_dict = arguments.model_dump() if isinstance(arguments, BaseModel) else dict(arguments)
        detail = summarize_tool_call(tool, arg_dict) or None
        if tool == "selenita_ask_user":
            detail = None  # the interactive question renders itself; the args line is noise
        self._emit_event(
            ToolEvent(kind="call", tool=tool, title=tool_title(tool), detail=detail,
                      arguments=arg_dict)
        )
        started = time.monotonic()
        try:
            builtin = tool.startswith(_BUILTIN_PREFIX)
            if builtin and not (
                self._policy.is_safe(tool) or self._policy.is_always_approved(tool, arg_dict)
            ):
                decision = await self._policy.ask(tool, arg_dict)
                if decision == DECISION_ALWAYS:
                    self._policy.remember_always(tool, arg_dict)
                elif decision != DECISION_APPROVE:
                    # The function never runs; the model sees this message and adapts.
                    context.result = (
                        f"Permission denied by the user: the {tool} tool was not executed "
                        f"(arguments: {arguments}). Do not retry it unless the user asks."
                    )
                    self._emit_event(
                        ToolEvent(kind="result", tool=tool, title=tool_title(tool),
                                  status="denied", duration=time.monotonic() - started)
                    )
                    return
            await call_next()
        except Exception as exc:
            label = f"failed: {str(exc)[:DETAIL_LIMIT]}"
            self._emit_event(
                ToolEvent(kind="result", tool=tool, title=tool_title(tool),
                          status=label, duration=time.monotonic() - started)
            )
            raise
        if tool == "selenita_ask_user":
            # The wait here is the user's own thinking time, not execution work: the
            # result event carries no duration (adapters skip the timer row for it).
            self._emit_event(
                ToolEvent(kind="result", tool=tool, title=tool_title(tool), status="done")
            )
            return
        # Selenita's tools return strings; the MAF function loop wraps both them and
        # richer client-tool payloads into ``Content`` lists — flatten them all to text
        # so ACP/TUI adapters render the tool's actual output, not a repr.
        output = _result_to_text(context.result)
        if len(output) > _OUTPUT_LIMIT:
            output = output[:_OUTPUT_LIMIT] + "…"
        self._emit_event(
            ToolEvent(kind="result", tool=tool, title=tool_title(tool),
                      status="done", duration=time.monotonic() - started,
                      output=output)
        )

    def _emit_event(self, event: ToolEvent) -> None:
        if self._emit is not None:
            self._emit(event)


__all__ = [
    "ApprovalMiddleware",
    "ApprovalPolicy",
    "Asker",
    "DECISION_ALWAYS",
    "DECISION_APPROVE",
    "DECISION_REJECT",
    "SAFE_TOOLS",
    "always_key",
]
