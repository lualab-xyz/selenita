"""ACP server of selenita: the real agent over the Agent Client Protocol (v1).

``SelenitaAgent`` translates the ACP surface to the selenita core (D012) — one core,
every entry point. The mappings:

- **Streaming**: ``answer`` → ``agent_message_chunk``, ``reasoning`` → ``agent_thought_chunk``.
- **Tools** (D phase): the core's :class:`~selenita.core.permissions.ToolEvent`s become
  ``tool_call`` notifications (start) and ``tool_call_update`` (result), with the raw
  arguments and output, so editors render live tool blocks.
- **Permissions** (D phase): mutating built-in tools ask through
  ``session/request_permission`` (allow once / always / reject). A client that cannot
  answer permission requests is denied fail-closed; safe tools keep running.
- **MCP** (D phase): the ``mcpServers`` declared in ``session/new`` are attached to that
  session only and run permission-free (the client attached them on purpose).

Everything else (``session/load``, fork/resume, list, modes, auth, commands) is backlog:
unsupported methods raise a clean, explicit error instead of failing silently.

References: ``.context/design.md`` §3 and the ``agent-client-protocol`` SDK examples.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
)
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ForkSessionResponse,
    ListSessionsResponse,
    PermissionOption,
    ResumeSessionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallStatus,
    ToolCallUpdate,
    ToolKind,
)

from selenita.config import Settings
from selenita.core import AgentCore, CoreChunk, create_core
from selenita.core.mcp import specs_from_acp
from selenita.core.permissions import (
    DECISION_ALWAYS,
    DECISION_APPROVE,
    DECISION_REJECT,
    ToolEvent,
)
from selenita.core.tools import Toolkit, tool_title

logger = logging.getLogger(__name__)

_METHOD_NOT_IMPLEMENTED = "selenita: this ACP protocol method is not supported yet"

# ACP ``ToolKind`` of every built-in tool (the SDK enum; the client picks icons).
_TOOL_KINDS: dict[str, ToolKind] = {
    "selenita_read_file": "read",
    "selenita_web_fetch": "fetch",
    "selenita_write_file": "edit",
    "selenita_edit_file": "edit",
    "selenita_run_command": "execute",
    "selenita_ask_user": "think",  # not offered over ACP (needs a prompt inside a prompt)
}

# ACP permission options offered for mutating calls: once / always / reject. There is no
# "reject always": repeated calls keep asking turn by turn.
_PERMISSION_OPTIONS = (
    PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
    PermissionOption(option_id="allow_always", name="Always allow", kind="allow_always"),
    PermissionOption(option_id="reject_once", name="Reject once", kind="reject_once"),
)

# ACP option ids the asker recognizes → selenita permission decisions.
_DECISIONS_BY_OPTION: dict[str, str] = {
    "allow_once": DECISION_APPROVE,
    "allow_always": DECISION_ALWAYS,
}


def _block_text(block: Any) -> str:
    """Extract the text of a ContentBlock (the SDK yields objects or dicts per version)."""
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def _tool_kind(tool: str) -> ToolKind:
    """ACP ``ToolKind`` of a tool call (unknown tools, e.g. MCP, are ``other``)."""
    return _TOOL_KINDS.get(tool, "other")


class SelenitaAgent(Agent):
    """ACP adapter of the selenita core: prompt turns → protocol updates.

    Subclass of ``acp.Agent`` (SDK interface); ``conn`` gets injected on connect. One
    turn runs at a time (a single shared core; ACP clients naturally serialize prompts
    per session) — concurrency is serialized for a deterministic tool-call stream.
    """

    def __init__(self, core: AgentCore | None = None) -> None:
        # Without an injected core, resolve it from the environment: mutating tools in,
        # ask_user out (the interactive question is a TUI-only feature; over ACP the
        # model can only ask through its protocol surface). The asker is a bound method:
        # it reads the current connection/session lazily, on the call itself.
        self._core = core if core is not None else create_core(
            Settings.from_env(),
            toolkit=Toolkit(asker=self._ask_tool_permission),
        )
        self._conn: Client | None = None
        self._session: str | None = None  # session owning the running turn (or None)
        self._turn_lock = asyncio.Lock()
        self._tool_seq = 0
        self._tool_ids: list[str] = []  # open tool calls, in emission order (this turn)
        self._outbox: list[asyncio.Task] = []  # in-flight session/update tasks
        self._core.on_tool_event = self._on_tool_event

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # ACP lifecycle supported in this phase
    # ------------------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        logger.info(
            "initialize: protocol=%s client=%s", protocol_version, getattr(client_info, "name", "?")
        )
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a session, attaching the MCP servers the client declared (per session)."""
        session_id = self._core.new_session_id()
        specs = specs_from_acp(mcp_servers)
        if specs:
            await self._core.attach_session_mcp(session_id, specs)
        logger.info("new_session: cwd=%s id=%s mcp=%d", cwd, session_id, len(specs))
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: Any,
        **kwargs: Any,
    ) -> PromptResponse:
        if self._conn is None:
            raise RuntimeError("prompt received without an active ACP connection")
        logger.info("prompt: session=%s blocks=%d", session_id, len(prompt or []))
        # One turn at a time over the shared core: concurrent prompts queue up.
        async with self._turn_lock:
            self._session = session_id
            try:
                for block in prompt or []:
                    text = _block_text(block)
                    if not text:
                        continue
                    async for chunk in self._core.respond(session_id, text):
                        await self._send_chunk(session_id, chunk)
            finally:
                self._session = None
        await self.drain_pending_updates()  # updates must not outlive the prompt response
        return PromptResponse(stop_reason="end_turn")

    # ------------------------------------------------------------------
    # Tools (D phase): ToolEvent → tool_call / tool_call_update
    # ------------------------------------------------------------------

    def _on_tool_event(self, event: ToolEvent) -> None:
        """Adapter hook: convert a core ToolEvent and push it to the client."""
        if self._conn is None:  # no connection: tool feedback would be noise
            return
        session = self._session
        if session is None:  # an event outside a prompt turn cannot be addressed
            logger.warning("dropping tool event outside a turn: %s", event.tool)
            return
        update: ToolCallStart | ToolCallProgress
        if event.kind == "call":
            tool_call_id = self._new_tool_id()
            self._tool_ids.append(tool_call_id)
            update = ToolCallStart(
                session_update="tool_call",
                tool_call_id=tool_call_id,
                title=event.title,
                kind=_tool_kind(event.tool),
                status="in_progress",
                raw_input=dict(event.arguments) if event.arguments is not None else None,
            )
        else:
            update = ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id=self._pop_tool_id(),
                status=self._call_status(event),
                raw_output=event.output,
            )
        self._enqueue_update(session, update)

    def _enqueue_update(
        self,
        session_id: str,
        update: ToolCallStart | ToolCallProgress | AgentMessageChunk | AgentThoughtChunk,
    ) -> None:
        """Queue a notification for the connection; FIFO tasks keep the wire order."""
        assert self._conn is not None  # callers check (or events do not reach here)
        task = asyncio.ensure_future(
            self._conn.session_update(session_id=session_id, update=update)
        )
        self._outbox.append(task)

    async def drain_pending_updates(self) -> None:
        """Await the in-flight session/update tasks (useful in tests and on shutdown)."""
        pending: list[asyncio.Task] = list(self._outbox)
        self._outbox.clear()
        for task in pending:
            await task
            exc = task.exception()
            if exc is not None:
                logger.warning("acp: session/update task failed: %s", exc)

    @staticmethod
    def _call_status(event: ToolEvent) -> ToolCallStatus:
        """Map a result event's status to an ACP ``ToolCallStatus``.

        ACP has no ``denied`` status: a refused call is reported as ``failed`` (the
        model also receives a denial message and keeps the turn going).
        """
        return "completed" if event.status == "done" else "failed"

    def _new_tool_id(self) -> str:
        self._tool_seq += 1
        return f"tool_{self._tool_seq}"

    def _pop_tool_id(self) -> str:
        """Close the most recent open tool call (one tool runs at a time per turn)."""
        if self._tool_ids:
            return self._tool_ids.pop()
        return self._new_tool_id()  # defensive: a result without a start still flows

    def _current_tool_id(self) -> str:
        """Tool call id in flight; used as the ``request_permission`` subject."""
        if self._tool_ids:
            return self._tool_ids[-1]
        return self._new_tool_id()

    # ------------------------------------------------------------------
    # Permissions (D phase): session/request_permission
    # ------------------------------------------------------------------

    async def _ask_tool_permission(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        """ACP asker: one ``session/request_permission`` per mutating tool call.

        Returns one of the ``DECISION_*`` strings of the permission policy. Any client
        failure (no transport, unsupported method, closed session) denies the call
        fail-closed: a client without the permission channel only gets safe tools.
        """
        if self._conn is None or self._session is None:
            logger.warning("no ACP permission channel: denying %s", tool_name)
            return DECISION_REJECT
        tool_call = ToolCallUpdate(
            tool_call_id=self._current_tool_id(),
            title=tool_title(tool_name),
            raw_input=dict(arguments),
        )
        try:
            response = await self._conn.request_permission(
                session_id=self._session, tool_call=tool_call, options=list(_PERMISSION_OPTIONS)
            )
        except Exception as exc:
            logger.warning(
                "permission request unsupported or failed (%s): denying %s",
                type(exc).__name__,
                tool_name,
            )
            return DECISION_REJECT
        if getattr(response.outcome, "outcome", None) != "selected":
            return DECISION_REJECT
        option_id = getattr(response.outcome, "option_id", None)
        if option_id is None:
            return DECISION_REJECT
        return _DECISIONS_BY_OPTION.get(str(option_id), DECISION_REJECT)

    # ------------------------------------------------------------------
    # Streaming mapping
    # ------------------------------------------------------------------

    async def _send_chunk(self, session_id: str, chunk: CoreChunk) -> None:
        """Send one core chunk as the matching ACP ``session/update`` notification.

        ``agent_message_chunk`` carries answer text; ``agent_thought_chunk`` carries the
        model's reasoning. The discriminant ``session_update`` is mandatory; the SDK
        serializes the model (JSON with camelCase aliases) into the notification.
        """
        if self._conn is None:
            raise RuntimeError(
                "prompt received without an active ACP connection (missing on_connect)"
            )
        # Both chunk types share the same TextContentBlock body but carry different
        # session_update literals, which is how ACP editors tell talk from thinking.
        update: AgentMessageChunk | AgentThoughtChunk
        if chunk.kind == "reasoning":
            update = AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=TextContentBlock(type="text", text=chunk.text),
            )
        else:
            update = AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text=chunk.text),
            )
        self._enqueue_update(session_id, update)

    # ------------------------------------------------------------------
    # ACP protocol methods not supported yet (backlog)
    # ------------------------------------------------------------------

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: Any = None,
        additional_directories: Any = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> None:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> None:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        # History is kept in the store (sessions are durable); per-session MCP tools
        # stay attached until the server exits (MVP).
        logger.info("close_session: history kept (session=%s)", session_id)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        # Real cancellation (aborting a running turn) is backlog: prompts are serialized,
        # so a cancel lets the current turn finish and the request has no effect yet.
        logger.info("cancel received (session=%s): the turn keeps running (backlog)", session_id)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(_METHOD_NOT_IMPLEMENTED)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        # Fire-and-forget notification: ignored outside the MVP scope.
        logger.info("ext_notification ignored: method=%s", method)

    async def aclose(self) -> None:
        """Release the core's external resources (MCP clients); safe to call twice."""
        await self._core.aclose()


async def run_acp_server() -> None:
    """Run the selenita ACP agent in stdio server mode (blocking)."""
    agent = SelenitaAgent()
    try:
        await run_agent(agent)
    finally:
        await agent.aclose()


__all__ = ["SelenitaAgent", "run_acp_server"]
