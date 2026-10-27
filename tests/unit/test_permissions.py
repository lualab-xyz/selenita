"""Tests of the permission gate (tools epic): policy, middleware and the full MAF tool loop.

The tool loop is exercised with ``ToolLoopClient`` (``tests/_fakes.py``), a
``FakeChatClient`` wrapped in MAF's ``FunctionInvocationLayer`` exactly like the real
OpenAI Responses client, so unit tests cover the whole call chain
agent → client loop → middleware gate → tool → function_result → model → final text.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import agent_framework as af
import pytest

from selenita.core.llm import LlmCore
from selenita.core.permissions import (
    DECISION_ALWAYS,
    DECISION_APPROVE,
    DECISION_REJECT,
    SAFE_TOOLS,
    ApprovalMiddleware,
    ApprovalPolicy,
    ToolEvent,
)
from selenita.core.tools import Toolkit
from selenita.sessions.store import InMemorySessionStore
from tests._fakes import ToolLoopClient


def _ctx(name: str, arguments: dict | None = None) -> SimpleNamespace:
    """Minimal stand-in for ``af.FunctionInvocationContext`` (process only uses these)."""
    return SimpleNamespace(
        function=SimpleNamespace(name=name),
        arguments=arguments or {},
        result=None,
    )


async def _noop_next() -> None:
    """call_next spy: records nothing, returns immediately."""


# ---------------------------------------------------------------------------
# ApprovalPolicy
# ---------------------------------------------------------------------------


def test_policy_marks_safe_tools_without_asking() -> None:
    policy = ApprovalPolicy()

    assert frozenset(
        {"selenita_read_file", "selenita_web_fetch", "selenita_ask_user"}
    ) == SAFE_TOOLS
    for name in SAFE_TOOLS:
        assert policy.is_safe(name)
    assert not policy.is_safe("selenita_write_file")


@pytest.mark.asyncio
async def test_middleware_emits_call_and_done_events() -> None:
    """Every execution emits ToolEvents: call first, then the result with its duration."""

    async def _always(*_args: object) -> str:
        return DECISION_ALWAYS

    policy = ApprovalPolicy(asker=_always)
    events: list[ToolEvent] = []
    middleware = ApprovalMiddleware(policy, emit=events.append)
    work_calls: list[int] = []

    async def work() -> None:
        work_calls.append(1)

    await middleware.process(_ctx("selenita_write_file", {"path": "f.txt"}), work)

    assert work_calls == [1]
    assert [(e.kind, e.title, e.detail, e.status) for e in events] == [
        ("call", "Write file", "path='f.txt'", None),
        ("result", "Write file", None, "done"),
    ]
    assert events[1].duration is not None and events[1].duration >= 0.0


@pytest.mark.asyncio
async def test_middleware_emits_denied_event_without_running_the_tool() -> None:

    async def _reject(*_args: object) -> str:
        return DECISION_REJECT

    policy = ApprovalPolicy(asker=_reject)
    events: list[ToolEvent] = []
    middleware = ApprovalMiddleware(policy, emit=events.append)
    ran: list[int] = []

    async def work() -> None:
        ran.append(1)

    await middleware.process(_ctx("selenita_run_command", {"command": "rm x"}), work)

    assert ran == []  # call_next never invoked
    assert [(e.kind, e.status) for e in events] == [("call", None), ("result", "denied")]


@pytest.mark.asyncio
async def test_middleware_emits_failed_event_and_reraises() -> None:
    """Errors surface as a failed result event and still propagate."""
    policy = ApprovalPolicy()
    events: list[ToolEvent] = []
    middleware = ApprovalMiddleware(policy, emit=events.append)

    async def boom() -> None:
        raise RuntimeError("the disk is gone")

    with pytest.raises(RuntimeError):
        # read_file is SAFE: no asker involved, the error comes from the execution.
        await middleware.process(_ctx("selenita_read_file", {"path": "f.txt"}), boom)

    assert [(e.kind, e.status) for e in events] == [
        ("call", None),
        ("result", "failed: the disk is gone"),
    ]


@pytest.mark.asyncio
async def test_middleware_hides_ask_user_arguments_from_the_detail_line() -> None:
    """ask_user renders no detail: the interactive question already shows it."""
    policy = ApprovalPolicy()
    events: list[ToolEvent] = []
    middleware = ApprovalMiddleware(policy, emit=events.append)

    async def work() -> None:
        return None

    await middleware.process(
        _ctx("selenita_ask_user", {"question": "Pick one", "options": ["a", "b"]}), work
    )

    assert [e.detail for e in events] == [None, None]
async def test_policy_ask_without_asker_denies_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = ApprovalPolicy()

    with caplog.at_level("WARNING", logger="selenita.core.permissions"):
        decision = await policy.ask("selenita_write_file", {"path": "x"})

    assert decision == DECISION_REJECT
    assert "denying selenita_write_file call" in caplog.text


@pytest.mark.asyncio
async def test_policy_ask_delegates_to_asker() -> None:
    seen: list[tuple[str, dict]] = []

    async def asker(tool: str, args: dict) -> str:
        seen.append((tool, args))
        return DECISION_APPROVE

    policy = ApprovalPolicy(asker)

    decision = await policy.ask("selenita_write_file", {"path": "x"})

    assert decision == DECISION_APPROVE
    assert seen == [("selenita_write_file", {"path": "x"})]


def test_policy_remember_always() -> None:
    policy = ApprovalPolicy()

    assert not policy.is_always_approved("selenita_write_file")
    policy.remember_always("selenita_write_file")
    assert policy.is_always_approved("selenita_write_file")


def test_policy_always_scopes_run_command_to_the_exact_command() -> None:
    """An "always" for run_command approves only that command, not any future one."""
    policy = ApprovalPolicy()

    policy.remember_always("selenita_run_command", {"command": " git status "})

    # Same command (whitespace-normalized) runs without asking again...
    assert policy.is_always_approved("selenita_run_command", {"command": "git status"})
    # ...but a different command with the same tool still asks.
    assert not policy.is_always_approved("selenita_run_command", {"command": "rm -rf /"})
    assert not policy.is_always_approved("selenita_run_command", {"command": ""})
    # Other tools keep their tool-wide scoping.
    policy.remember_always("selenita_write_file")
    assert policy.is_always_approved("selenita_write_file", {"path": "any.txt"})


def test_policy_asker_setter_wires_or_clears() -> None:
    async def asker(tool: str, args: dict) -> str:
        return DECISION_REJECT

    policy = ApprovalPolicy()
    assert policy.asker is None
    policy.asker = asker
    assert policy.asker is asker
    policy.asker = None
    assert policy.asker is None


# ---------------------------------------------------------------------------
# ApprovalMiddleware process (unit level, stubbed context + call_next)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_runs_safe_tool_without_asking() -> None:
    executed = False

    async def call_next() -> None:
        nonlocal executed
        executed = True

    middleware = ApprovalMiddleware(ApprovalPolicy())
    ctx = _ctx("selenita_read_file", {"path": "a.txt"})

    await middleware.process(ctx, call_next)

    assert executed
    assert ctx.result is None


@pytest.mark.asyncio
async def test_middleware_deny_short_circuits_without_asker() -> None:
    executed = False

    async def call_next() -> None:
        nonlocal executed
        executed = True

    middleware = ApprovalMiddleware(ApprovalPolicy())
    ctx = _ctx("selenita_write_file", {"path": "x"})

    await middleware.process(ctx, call_next)

    assert not executed  # the function never ran: short-circuited via context.result
    assert "Permission denied" in ctx.result
    assert "selenita_write_file" in ctx.result


@pytest.mark.asyncio
async def test_middleware_approves_when_asker_approves() -> None:
    executed = False

    async def call_next() -> None:
        nonlocal executed
        executed = True

    async def asker(tool: str, args: dict) -> str:
        return DECISION_APPROVE

    middleware = ApprovalMiddleware(ApprovalPolicy(asker))
    ctx = _ctx("selenita_write_file", {"path": "x"})

    await middleware.process(ctx, call_next)

    assert executed
    assert ctx.result is None


@pytest.mark.asyncio
async def test_middleware_approves_and_remembers_always() -> None:
    asker_calls: list[str] = []

    async def asker(tool: str, args: dict) -> str:
        asker_calls.append(tool)
        return DECISION_ALWAYS

    policy = ApprovalPolicy(asker)
    middleware = ApprovalMiddleware(policy)
    ctx = _ctx("selenita_write_file")

    await middleware.process(ctx, _noop_next)
    await middleware.process(ctx, _noop_next)

    assert policy.is_always_approved("selenita_write_file")
    assert asker_calls == ["selenita_write_file"]  # only the first call asks


@pytest.mark.asyncio
async def test_middleware_rejects_when_asker_says_no() -> None:
    async def call_next() -> None:
        pytest.fail("call_next must not run on rejection")

    async def asker(tool: str, args: dict) -> str:
        return DECISION_REJECT

    middleware = ApprovalMiddleware(ApprovalPolicy(asker))

    await middleware.process(_ctx("selenita_run_command"), call_next)


# ---------------------------------------------------------------------------
# Full MAF tool loop (ToolLoopClient = FunctionInvocationLayer + FakeChatClient)
# ---------------------------------------------------------------------------


def _call_content(name: str, call_id: str, arguments: dict) -> af.Content:
    return af.Content(type="function_call", name=name, call_id=call_id, arguments=arguments)


def _model_saw_result(client: ToolLoopClient, snippet: str) -> bool:
    """True when the re-requested messages carried a function_result with ``snippet``.

    MAF places the tool's string output on the content's ``result`` field (``output``
    is a separate rich-result attribute that stays None for plain text results).
    """
    return any(
        getattr(c, "type", None) == "function_result"
        and snippet in (getattr(c, "result", "") or "")
        for m in client.last_messages
        for c in (m.contents or [])
    )


async def _collect(core: LlmCore, session_id: str) -> str:
    return "".join(c.text for c in [chunk async for chunk in core.respond(session_id, "hi")])


@pytest.mark.asyncio
async def test_tool_loop_runs_safe_read_tool_without_asking(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("file payload", encoding="utf-8")
    client = ToolLoopClient(
        replay="CONTENT",
        streaming_chunks=["CONTENT"],
        function_call=_call_content("selenita_read_file", "call_1", {"path": str(target)}),
    )
    # No asker wired: safe tools must still run.
    store = InMemorySessionStore()

    core = LlmCore(client=client, store=store, toolkit=Toolkit())
    answer = await _collect(core, "sess_toolsafe")

    assert answer == "CONTENT"
    # The loop re-requested the model with the tool's real output.
    assert _model_saw_result(client, "file payload")
    # Both turns persisted as usual.
    assert [(m.role, m.text) for m in store.load("sess_toolsafe")] == [
        ("user", "hi"),
        ("agent", "CONTENT"),
    ]


@pytest.mark.asyncio
async def test_tool_loop_runs_ask_user_without_asking() -> None:
    # ask_user is the user's own input — it must never sit behind a permission prompt.
    client = ToolLoopClient(
        replay="CONTENT",
        streaming_chunks=["CONTENT"],
        function_call=_call_content(
            "selenita_ask_user",
            "call_ask",
            {"question": "Pick one", "options": ["a", "b"]},
        ),
    )
    # No asker wired: with a question-tool call on the line, this would hang forever
    # if the middleware asked for permission — it must run automatically instead.
    store = InMemorySessionStore()

    core = LlmCore(
        client=client, store=store, toolkit=Toolkit(include_ask_user=True)
    )
    answer = await _collect(core, "sess_askuser")

    assert answer == "CONTENT"
    assert _model_saw_result(client, "No interactive user is available")


@pytest.mark.asyncio
async def test_tool_loop_denies_mutating_tool_without_asker(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    client = ToolLoopClient(
        replay="done",
        streaming_chunks=["done"],
        function_call=_call_content(
            "selenita_write_file", "call_1", {"path": str(target), "content": "should not appear"}
        ),
    )
    store = InMemorySessionStore()

    core = LlmCore(client=client, store=store, toolkit=Toolkit())
    answer = await _collect(core, "sess_tooldeny")

    # Fail-closed: the file was never written and the model got a denial result.
    assert not target.exists()
    assert answer == "done"
    assert _model_saw_result(client, "Permission denied")


@pytest.mark.asyncio
async def test_tool_loop_asks_then_runs_mutating_tool(tmp_path: Path) -> None:
    target = tmp_path / "written.txt"
    asked: list[tuple[str, dict]] = []

    async def asker(tool: str, args: dict) -> str:
        asked.append((tool, args))
        return DECISION_APPROVE

    client = ToolLoopClient(
        replay="ok",
        streaming_chunks=["ok"],
        function_call=_call_content(
            "selenita_write_file", "call_1", {"path": str(target), "content": "written by tool"}
        ),
    )
    core = LlmCore(
        client=client,
        store=InMemorySessionStore(),
        toolkit=Toolkit(asker=asker),
    )
    events: list[ToolEvent] = []
    core.on_tool_event = events.append

    answer = await _collect(core, "sess_toolask")

    assert answer == "ok"
    assert asked == [("selenita_write_file", {"path": str(target), "content": "written by tool"})]
    assert target.read_text(encoding="utf-8") == "written by tool"
    assert _model_saw_result(client, "Wrote")
    # The adapters can render live $ blocks from these events.
    assert [(e.kind, e.title, e.status) for e in events] == [
        ("call", "Write file", None),
        ("result", "Write file", "done"),
    ]


@pytest.mark.asyncio
async def test_tool_loop_always_skips_future_prompts(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    asked: list[str] = []

    async def asker(tool: str, args: dict) -> str:
        asked.append(tool)
        return DECISION_ALWAYS

    client = ToolLoopClient(
        replay="done",
        streaming_chunks=["done"],
        function_call=_call_content(
            "selenita_write_file", "call_1", {"path": str(target), "content": "a"}
        ),
    )
    core = LlmCore(client=client, store=InMemorySessionStore(), toolkit=Toolkit(asker=asker))

    await _collect(core, "sess_a")
    await _collect(core, "sess_b")

    # The second turn reused the same agent: the gate remembered "always" in-process.
    assert asked == ["selenita_write_file"]


@pytest.mark.asyncio
async def test_tool_loop_always_for_run_command_only_covers_same_command(tmp_path: Path) -> None:
    """The "always" decision on run_command is scoped to the exact command (safety)."""
    asked: list[str] = []

    async def asker(tool: str, args: dict) -> str:
        asked.append(args["command"])
        return DECISION_ALWAYS

    client = ToolLoopClient(
        replay="done",
        streaming_chunks=["done"],
        function_call=_call_content("selenita_run_command", "c1", {"command": "echo one"}),
    )
    core = LlmCore(client=client, store=InMemorySessionStore(), toolkit=Toolkit(asker=asker))

    # Turn 1: new command → asks, remembers the exact command, runs it.
    await _collect(core, "sess_r1")
    # Turn 2: same command on the same core → the gate remembers it and does not ask.
    await _collect(core, "sess_r2")
    # Turn 3: a different command → asks again.
    client.function_call = _call_content("selenita_run_command", "c2", {"command": "echo two"})
    await _collect(core, "sess_r3")

    assert asked == ["echo one", "echo two"]
