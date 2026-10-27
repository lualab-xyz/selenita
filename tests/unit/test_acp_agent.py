"""Unit tests of the SelenitaAgent — no I/O, fake connection (D008: mocks from day one).

The core is always a fake (echo core or a scripted core emitting ToolEvents): the ACP
layer is the unit under test (chunk mapping, tool-call updates, permission askers).
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator

from acp import InitializeResponse, NewSessionResponse, PromptResponse  # noqa: F401
from acp.schema import TextContentBlock, ToolCallProgress, ToolCallStart
from agent_framework.exceptions import ToolException

from selenita.acp_server.agent import SelenitaAgent
from selenita.core.chunks import CoreChunk
from selenita.core.echo import EchoCore
from selenita.core.mcp import McpServerSpec
from selenita.core.permissions import (
    DECISION_ALWAYS,
    DECISION_REJECT,
    ToolEvent,
)
from selenita.sessions.store import InMemorySessionStore


class RecordingConn:
    """Fake ACP client recording session_update and request_permission activity."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.permission_requests: list[tuple[str, object, list]] = []
        self.next_outcome: object | None = None
        self.fail_permanently = False

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        # The transport flushes in order; draining once keeps events sequenced for tests.
        self.updates.append((session_id, update))

    async def request_permission(self, session_id: str, tool_call, options, **kwargs):
        if self.fail_permanently:
            raise RuntimeError("client does not support request_permission")
        if self.next_outcome is None:
            raise RuntimeError("client has no permission UI")
        self.permission_requests.append((session_id, tool_call, list(options)))
        return self.next_outcome


def make_agent(core=None) -> tuple[SelenitaAgent, RecordingConn]:
    """Agent with a fake core (echo by default) and a recording connection attached."""
    agent = SelenitaAgent(core=core or EchoCore(store=InMemorySessionStore()))
    conn = RecordingConn()
    agent.on_connect(conn)  # type: ignore[arg-type]
    return agent, conn


class ScriptedCore:
    """Fake core that replays scripted ToolEvents and answers with a fixed text."""

    on_tool_event = None

    def __init__(self, script: list[ToolEvent]) -> None:
        self._script = script
        self._store = InMemorySessionStore()
        self.attached: list[tuple[str, list[McpServerSpec]]] = []

    @property
    def store(self):
        return self._store

    def new_session_id(self) -> str:
        return "sess_scripted"

    async def aclose(self) -> None:
        return None

    async def attach_session_mcp(self, session_id: str, specs) -> None:
        self.attached.append((session_id, list(specs)))

    async def respond(self, session_id: str, user_text: str) -> AsyncIterator[CoreChunk]:
        for event in self._script:
            assert self.on_tool_event is not None
            self.on_tool_event(event)
        yield CoreChunk(kind="answer", text="done")


# ---------------------------------------------------------------------------
# Streaming chunks (existing behavior, renamed adapter)
# ---------------------------------------------------------------------------

async def test_initialize_echoes_protocol_version() -> None:
    agent, _ = make_agent()
    response: InitializeResponse = await agent.initialize(protocol_version=1)
    assert response.protocol_version == 1


async def test_new_session_returns_prefixed_uuid() -> None:
    agent, _ = make_agent()
    response: NewSessionResponse = await agent.new_session(cwd="/tmp/project")
    assert response.session_id.startswith("sess_")
    assert len(response.session_id) > len("sess_")


async def test_prompt_streams_answer_chunks() -> None:
    agent, conn = make_agent()
    response: PromptResponse = await agent.prompt(
        "sess_x1",
        [TextContentBlock(type="text", text="hello"), {"type": "text", "text": "world"}],
    )
    assert response.stop_reason == "end_turn"
    texts = [update.content.text for _, update in conn.updates]
    assert texts == ["hello", "world"]
    assert all(sid == "sess_x1" for sid, _ in conn.updates)


# ---------------------------------------------------------------------------
# D1: ToolEvents → tool_call / tool_call_update notifications
# ---------------------------------------------------------------------------

async def test_tool_events_become_tool_call_notifications() -> None:
    call = ToolEvent(kind="call", tool="selenita_run_command", title="Shell",
                     detail="git status", arguments={"command": "git status"})
    done = ToolEvent(kind="result", tool="selenita_run_command", title="Shell",
                     status="done", duration=0.2, output="On branch main")
    agent, conn = await _turn_with_sync([call, done])
    kinds = [(u.session_update, type(u).__name__) for _, u in conn.updates]
    assert kinds == [("tool_call", "ToolCallStart"), ("tool_call_update", "ToolCallProgress"),
                     ("agent_message_chunk", "AgentMessageChunk")]
    start = conn.updates[0][1]
    assert start.tool_call_id == "tool_1"
    assert start.status == "in_progress"
    assert start.kind == "execute"
    assert start.raw_input == {"command": "git status"}
    progress = conn.updates[1][1]
    assert progress.tool_call_id == "tool_1"
    assert progress.status == "completed"
    assert progress.raw_output == "On branch main"


async def test_denied_and_failed_results_map_to_failed_status() -> None:
    denied = ToolEvent(kind="result", tool="selenita_write_file", title="Write file",
                       status="denied", duration=0.1)
    failed = ToolEvent(kind="result", tool="selenita_read_file", title="Read file",
                       status=f"failed: {ToolException('boom')}", duration=0.1)
    agent, conn = await _turn_with_sync([denied])
    assert updates_status(conn) == ["failed"]
    agent2, conn2 = await _turn_with_sync([failed])
    assert updates_status(conn2) == ["failed"]


def updates_status(conn: RecordingConn) -> list:
    return [u.status for _, u in conn.updates if isinstance(u, ToolCallProgress)]


async def _turn_with_sync(events: list[ToolEvent]) -> tuple[SelenitaAgent, RecordingConn]:
    agent, conn = make_agent(ScriptedCore(events))
    await agent.prompt("sess_tools", [TextContentBlock(type="text", text="go")])
    await agent.drain_pending_updates()
    return agent, conn


async def test_each_call_gets_its_own_tool_id() -> None:
    script = [
        ToolEvent(kind="call", tool="selenita_read_file", title="Read file"),
        ToolEvent(kind="result", tool="selenita_read_file", title="Read file", status="done",
                  output="a"),
        ToolEvent(kind="call", tool="selenita_edit_file", title="Edit file"),
        ToolEvent(kind="result", tool="selenita_edit_file", title="Edit file", status="done",
                  output="b"),
    ]
    _, conn = await _turn_with_sync(script)
    tool_calls = [u for _, u in conn.updates if isinstance(u, ToolCallStart)]
    updates = [u for _, u in conn.updates if isinstance(u, ToolCallProgress)]
    assert [c.tool_call_id for c in tool_calls] == ["tool_1", "tool_2"]
    assert [p.tool_call_id for p in updates] == ["tool_1", "tool_2"]


async def _turn_with(events: list[ToolEvent]) -> tuple[SelenitaAgent, RecordingConn]:
    agent, conn = make_agent(ScriptedCore(events))
    await agent.prompt("sess_tools", [TextContentBlock(type="text", text="go")])
    await agent.drain_pending_updates()
    return agent, conn


# ---------------------------------------------------------------------------
# D2: request_permission → policy decisions
# ---------------------------------------------------------------------------

async def test_permission_always_outcome_maps_to_always() -> None:
    agent, conn = make_agent()
    conn.next_outcome = _selected("allow_always")
    agent._session = "sess_perm"
    decision = await agent._ask_tool_permission("selenita_run_command", {"command": "ls"})
    assert decision == DECISION_ALWAYS
    session_id, tool_call, options = conn.permission_requests[0]
    assert session_id == "sess_perm"
    assert tool_call.title == "Shell"
    assert options[0].kind == "allow_once" and options[2].kind == "reject_once"


async def test_permission_cancelled_or_broken_client_denies_fail_closed() -> None:
    agent, conn = make_agent()
    conn.next_outcome = _cancelled()
    agent._session = "sess_perm"
    assert await agent._ask_tool_permission("selenita_write_file", {}) == DECISION_REJECT

    agent2, conn2 = make_agent()
    conn2.fail_permanently = True  # client without the permission method support
    agent2._session = "sess_perm"
    assert await agent2._ask_tool_permission("selenita_edit_file", {}) == DECISION_REJECT

    agent3, conn3 = make_agent()
    agent3._session = None  # out of any turn: no channel, no approval
    assert await agent3._ask_tool_permission("selenita_edit_file", {}) == DECISION_REJECT


async def test_permission_unknown_option_denies() -> None:
    agent, conn = make_agent()
    conn.next_outcome = _selected("allow_everything_forever")  # not recognized
    agent._session = "sess_perm"
    assert await agent._ask_tool_permission("selenite_run_command", {}) == DECISION_REJECT


def _selected(option_id: str):
    from acp.schema import AllowedOutcome, RequestPermissionResponse

    outcome = AllowedOutcome(outcome="selected", option_id=option_id)
    return RequestPermissionResponse(outcome=outcome)


def _cancelled():
    from acp.schema import DeniedOutcome, RequestPermissionResponse

    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


# ---------------------------------------------------------------------------
# D3: mcpServers of session/new → per-session attach
# ---------------------------------------------------------------------------

async def test_new_session_attaches_declared_mcp_servers() -> None:
    core = ScriptedCore([])
    agent, _ = make_agent(core)
    response = await agent.new_session(
        cwd="/tmp",
        mcp_servers=[{"name": "demo", "command": "python", "args": ["-u", "demo.py"]}],
    )
    assert response.session_id == "sess_scripted"
    assert core.attached == [("sess_scripted", [McpServerSpec(name="demo", command="python",
                                                             args=("-u", "demo.py"))])]


async def test_new_session_without_mcp_servers_attaches_nothing() -> None:
    core = ScriptedCore([])
    agent, _ = make_agent(core)
    await agent.new_session(cwd="/tmp")
    assert core.attached == []


# ---------------------------------------------------------------------------
# Prompt serialization: one turn at a time over the shared core
# ---------------------------------------------------------------------------

async def test_concurrent_prompts_are_serialized() -> None:
    class SlowCore(ScriptedCore):
        def __init__(self) -> None:
            super().__init__([])
            self.active = 0
            self.max_active = 0

        async def respond(self, session_id: str, user_text: str) -> AsyncIterator[CoreChunk]:
            import asyncio

            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            yield CoreChunk(kind="answer", text=user_text)

    core = SlowCore()
    agent, conn = make_agent(core)
    import asyncio

    await asyncio.gather(
        agent.prompt("sess_p1", [{"type": "text", "text": "one"}]),
        agent.prompt("sess_p2", [{"type": "text", "text": "two"}]),
    )
    assert core.max_active == 1


# ---------------------------------------------------------------------------
# Protocol shape (regression guard of the adapter surface)
# ---------------------------------------------------------------------------

def test_agent_methods_are_coroutines() -> None:
    agent, _ = make_agent()
    for method in ("initialize", "new_session", "prompt", "drain_pending_updates",
                   "_ask_tool_permission", "aclose"):
        assert inspect.iscoroutinefunction(getattr(agent, method)), method
