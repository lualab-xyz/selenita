"""Tests of the LLM core on top of MAF (all with FakeChatClient: no network, no real keys)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from selenita.config import Settings
from selenita.core import EchoCore, create_core
from selenita.core import llm as llm_module
from selenita.core.chunks import CoreChunk
from selenita.core.llm import _INSTRUCTIONS as SELENITA_INSTRUCTIONS
from selenita.core.llm import LlmConfigError, LlmCore
from selenita.core.mcp import McpServerSpec
from selenita.core.permissions import ApprovalMiddleware
from selenita.core.tools import Toolkit
from selenita.sessions.store import InMemorySessionStore, SessionMessage
from tests._fakes import FakeChatClient


def _settings(overrides: dict[str, str] | None = None) -> Settings:
    base = {"SELENITA_SESSIONS_DIR": "/tmp/selenita-test-sessions", "SELENITA_LOG_LEVEL": "error"}
    return Settings.from_env({**base, **(overrides or {})})


def _core(fake: FakeChatClient, store: InMemorySessionStore) -> LlmCore:
    return LlmCore(client=fake, store=store)


def _answer_text(chunks: list[CoreChunk]) -> str:
    """Joined answer text (reasoning chunks must never leak into it)."""
    return "".join(c.text for c in chunks if c.kind == "answer")


@pytest.mark.asyncio
async def test_respond_streams_reply_and_persists_both_turns() -> None:
    store = InMemorySessionStore()
    fake = FakeChatClient("hello human")
    core = _core(fake, store)

    chunks = [chunk async for chunk in core.respond("sess_llm01", "greetings")]

    assert all(c.kind == "answer" for c in chunks)
    assert _answer_text(chunks) == "hello human"
    assert [(m.role, m.text) for m in store.load("sess_llm01")] == [
        ("user", "greetings"),
        ("agent", "hello human"),
    ]
    # The LLM receives exactly the user turn (empty history at start).
    assert [(m.role, m.text) for m in fake.last_messages] == [("user", "greetings")]


@pytest.mark.asyncio
async def test_stateless_core_passes_store_false_options() -> None:
    """Responses provider runs store=False so no previous_response_id is sent (D014)."""
    store = InMemorySessionStore()
    fake = FakeChatClient("ok")
    core = LlmCore(client=fake, store=store, stateless=True)

    _ = [chunk async for chunk in core.respond("sess_llm01", "hi")]

    # MAF merges agent-level options (instructions/tool_choice) with the run options.
    assert fake.last_options == {
        "store": False,
        "instructions": SELENITA_INSTRUCTIONS,
        "tool_choice": "auto",
    }


@pytest.mark.asyncio
async def test_non_stateless_core_passes_no_options() -> None:
    store = InMemorySessionStore()
    fake = FakeChatClient("ok")
    core = _core(fake, store)

    _ = [chunk async for chunk in core.respond("sess_llm01", "hi")]

    assert "store" not in fake.last_options


@pytest.mark.asyncio
async def test_history_from_store_is_replayed_to_llm() -> None:
    store = InMemorySessionStore()
    store.append("sess_llm01", SessionMessage(role="user", text="first question"))
    store.append("sess_llm01", SessionMessage(role="agent", text="first answer"))
    fake = FakeChatClient("ok")
    core = _core(fake, store)

    async for _ in core.respond("sess_llm01", "second question"):
        pass

    # D009: our JSONL history is replayed complete on every turn.
    assert [(m.role, m.text) for m in fake.last_messages] == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
    ]
    assert store.load("sess_llm01")[-1].role == "agent"


@pytest.mark.asyncio
async def test_streaming_chunks_are_forwarded_as_emitted() -> None:
    fake = FakeChatClient("irrelevant", streaming_chunks=["how ", "are you?"])
    core = _core(fake, InMemorySessionStore())

    chunks = [chunk async for chunk in core.respond("sess_llm02", "hello")]

    assert [(c.kind, c.text) for c in chunks] == [("answer", "how "), ("answer", "are you?")]


@pytest.mark.asyncio
async def test_reasoning_chunks_stream_before_answers_and_never_persist() -> None:
    store = InMemorySessionStore()
    fake = FakeChatClient("the answer", reasoning_chunks=["thinking 1 ", "thinking 2 "])
    core = _core(fake, store)

    chunks = [chunk async for chunk in core.respond("sess_llm06", "hello")]

    kinds = [c.kind for c in chunks]
    assert "reasoning" in kinds
    assert kinds.index("reasoning") < kinds.index("answer")
    assert [c.text for c in chunks if c.kind == "reasoning"] == ["thinking 1 ", "thinking 2 "]
    assert _answer_text(chunks) == "the answer"
    # Only the answer text reaches the persisted history.
    assert [(m.role, m.text) for m in store.load("sess_llm06")] == [
        ("user", "hello"),
        ("agent", "the answer"),
    ]


@pytest.mark.asyncio
async def test_empty_user_text_short_circuits_without_llm_call() -> None:
    fake = FakeChatClient("SHOULD NOT HAPPEN")
    core = _core(fake, InMemorySessionStore())

    chunks = [chunk async for chunk in core.respond("sess_llm03", "   ")]

    assert chunks == []
    assert fake.last_messages == []


@pytest.mark.asyncio
async def test_replace_mode_full_text_update_is_not_reemitted() -> None:
    # Reproduces the real Responses client flow: delta updates (append) plus an
    # output_text.done event with the whole text (replace). Only deltas are re-sent.
    fake = FakeChatClient("x", streaming_chunks=["1, ", "2, 3"], replace_replay="1, 2, 3")
    store = InMemorySessionStore()
    core = _core(fake, store)

    chunks = [chunk async for chunk in core.respond("sess_llm05", "count")]

    assert [c.text for c in chunks if c.kind == "answer"] == ["1, ", "2, 3"]
    assert store.load("sess_llm05")[-1].text == "1, 2, 3"


class _BoomClient:
    # get_response must be synchronous: MAF uses ResponseStream without await in
    # streaming, so the failure must fire when called, not when awaited.
    @property
    def additional_properties(self) -> dict[str, object]:
        return {}

    def get_response(self, messages, *, stream: bool = False, **kwargs):
        raise RuntimeError("gateway down")


@pytest.mark.asyncio
async def test_client_failure_degrades_with_warning_and_persists_nothing(caplog) -> None:
    store = InMemorySessionStore()
    core = LlmCore(client=_BoomClient(), store=store)  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR):
        chunks = [chunk async for chunk in core.respond("sess_llm04", "hello")]

    assert len(chunks) == 1
    assert chunks[0].kind == "answer"
    assert chunks[0].text.startswith("⚠")
    # The user turn was already persisted; the agent reply does not exist.
    assert [m.role for m in store.load("sess_llm04")] == ["user"]
    assert any("error in LLM turn" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_build_requires_api_key_for_openai_provider(tmp_path) -> None:
    settings = _settings(
        {
            "SELENITA_AGENT_MODE": "llm",
            "SELENITA_LLM_PROVIDER": "openai",
            "SELENITA_SESSIONS_DIR": str(tmp_path),
        }
    )
    with pytest.raises(LlmConfigError):
        LlmCore.build(settings)


@pytest.mark.asyncio
async def test_build_unknown_provider_raises() -> None:
    settings = _settings({"SELENITA_AGENT_MODE": "llm", "SELENITA_LLM_PROVIDER": "no-existe"})
    with pytest.raises(LlmConfigError):
        LlmCore.build(settings)


def test_build_openai_chat_provider_returns_core(tmp_path) -> None:
    from agent_framework.openai import OpenAIChatCompletionClient

    settings = _settings(
        {
            "SELENITA_AGENT_MODE": "llm",
            "SELENITA_LLM_PROVIDER": "openai-chat",
            "SELENITA_OPENAI_API_KEY": "sk-test",
            "SELENITA_SESSIONS_DIR": str(tmp_path),
        }
    )
    core = LlmCore.build(settings)
    assert isinstance(core, LlmCore)
    assert isinstance(core._client, OpenAIChatCompletionClient)


def test_build_openai_provider_uses_stateless_reasoning_client(tmp_path) -> None:
    """Responses provider: PlaintextReasoningClient + store=False (D014)."""
    from agent_framework.openai import OpenAIChatClient

    from selenita.core.clients import PlaintextReasoningClient

    settings = _settings(
        {
            "SELENITA_AGENT_MODE": "llm",
            "SELENITA_LLM_PROVIDER": "openai",
            "SELENITA_OPENAI_API_KEY": "sk-test",
            "SELENITA_SESSIONS_DIR": str(tmp_path),
        }
    )
    core = LlmCore.build(settings)
    assert isinstance(core._client, PlaintextReasoningClient)
    assert isinstance(core._client, OpenAIChatClient)  # API-compatible with the base client
    assert core._stateless is True


def test_build_ollama_provider_returns_core(tmp_path) -> None:
    from agent_framework_ollama import OllamaChatClient

    settings = _settings(
        {
            "SELENITA_AGENT_MODE": "llm",
            "SELENITA_LLM_PROVIDER": "ollama",
            "SELENITA_SESSIONS_DIR": str(tmp_path),
        }
    )
    core = LlmCore.build(settings)
    assert isinstance(core._client, OllamaChatClient)


def test_create_core_echo_mode_returns_echo() -> None:
    core = create_core(_settings({"SELENITA_AGENT_MODE": "echo"}))
    assert isinstance(core, EchoCore)


def test_create_core_llm_without_key_falls_back_to_echo_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        core = create_core(
            _settings({"SELENITA_AGENT_MODE": "llm", "SELENITA_LLM_PROVIDER": "openai"})
        )
    assert isinstance(core, EchoCore)
    assert any("echo mode" in record.message.lower() for record in caplog.records)


def test_create_core_llm_with_key_builds_llm_core(tmp_path) -> None:
    core = create_core(
        _settings(
            {
                "SELENITA_AGENT_MODE": "llm",
                "SELENITA_LLM_PROVIDER": "openai-chat",
                "SELENITA_OPENAI_API_KEY": "sk-test",
                "SELENITA_SESSIONS_DIR": str(tmp_path),
            }
        )
    )
    assert isinstance(core, LlmCore)


# --- Built-in tools wiring (tools epic) ------------------------------------------


def _tool_names(core: LlmCore) -> set[str]:
    tools = core._agent.default_options.get("tools") or ()
    return {tool.name for tool in tools}


def test_toolkit_wires_builtin_tools_and_approval_gate() -> None:
    store = InMemorySessionStore()
    core = LlmCore(client=FakeChatClient("ok"), store=store, toolkit=Toolkit())

    assert _tool_names(core) == {
        "selenita_read_file",
        "selenita_write_file",
        "selenita_edit_file",
        "selenita_run_command",
        "selenita_web_fetch",
    }
    assert core.approval_policy is not None
    assert any(isinstance(m, ApprovalMiddleware) for m in core._agent.middleware)


def test_no_toolkit_means_no_builtin_tools_and_no_gate() -> None:
    store = InMemorySessionStore()
    core = _core(FakeChatClient("ok"), store)

    assert _tool_names(core) == set()
    assert core.approval_policy is None


def test_toolkit_respects_flags_and_ask_user_tool() -> None:
    store = InMemorySessionStore()
    core = LlmCore(
        client=FakeChatClient("ok"),
        store=store,
        toolkit=Toolkit(include_mutating=False, include_ask_user=True),
    )

    assert _tool_names(core) == {
        "selenita_read_file",
        "selenita_web_fetch",
        "selenita_ask_user",
    }


def test_set_permission_asker_updates_the_policy() -> None:
    core = LlmCore(client=FakeChatClient("ok"), store=InMemorySessionStore(), toolkit=Toolkit())
    first = core.approval_policy

    async def asker(tool_name: str, arguments: Mapping[str, Any]) -> str:
        return "reject"

    core.set_permission_asker(asker)

    assert core.approval_policy is first
    assert first.asker is asker


# --- MCP tools attached to the core (tools epic, D013) -----------------------------


class _StubMcpTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_mcp_tools_attach_lazily_never_reopen_and_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[list[McpServerSpec]] = []
    closed: list[str] = []

    async def fake_open(specs: list[McpServerSpec]) -> list[_StubMcpTool]:
        opened.append(list(specs))
        return [_StubMcpTool(s.name) for s in specs]

    async def fake_close(tools: list[_StubMcpTool]) -> None:
        closed.extend(tool.name for tool in tools)

    monkeypatch.setattr(llm_module, "open_mcp_tools", fake_open)
    monkeypatch.setattr(llm_module, "close_mcp_tools", fake_close)
    core = LlmCore(
        client=FakeChatClient("hi"),
        store=InMemorySessionStore(),
        mcp_specs=[McpServerSpec("demo", "cmd")],
    )

    first = [chunk async for chunk in core.respond("sess_mcp1", "hello")]
    assert _answer_text(first) == "hi"
    assert [s.name for s in opened[0]] == ["demo"]

    # A later turn reuses the already-connected tools (single open per core).
    [chunk async for chunk in core.respond("sess_mcp1", "again")]
    assert len(opened) == 1

    await core.aclose()
    assert closed == ["demo"]
    await core.aclose()
    assert closed == ["demo"]  # idempotent


@pytest.mark.asyncio
async def test_all_mcp_servers_failing_keeps_the_turn_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_open(_specs: list[McpServerSpec]) -> list[_StubMcpTool]:
        return []

    monkeypatch.setattr(llm_module, "open_mcp_tools", fake_open)
    core = LlmCore(
        client=FakeChatClient("still here"),
        store=InMemorySessionStore(),
        mcp_specs=[McpServerSpec("bad", "b")],
    )
    chunks = [chunk async for chunk in core.respond("sess_mcp2", "hello")]
    assert _answer_text(chunks) == "still here"
