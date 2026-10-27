"""Integration tests of the ACP permission flow (D2) over the real stdio transport.

The server runs as a subprocess with the real LLM core, but the model is a scripted
fake OpenAI ``/responses`` server (``_openai_stub``): one tool call, then a message.
The harness here acts as an ACP client that ANSWERS ``session/request_permission``
— that is the round trip this suite pins down end to end:

server → ``tool_call`` (session/update) → ``session/request_permission`` → client
decision → tool runs or is denied → ``tool_call_update`` → model continues.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration._harness import AcpTestClient
from tests.integration._openai_stub import FakeResponsesServer, FunctionCallTurn, MessageTurn

_TIMEOUT = float(60)


def _updates(exchange) -> list[dict]:
    return [
        n["params"]["update"]
        for n in exchange.notifications
        if n["method"] == "session/update"
    ]


async def _launch_llm_client(tmpsessions: Path, server: FakeResponsesServer,
                             decisions: list[str] | None) -> AcpTestClient:
    client = await AcpTestClient.launch_env(
        {
            "SELENITA_AGENT_MODE": "llm",
            "SELENITA_LLM_PROVIDER": "openai",
            "SELENITA_OPENAI_BASE_URL": server.base_url,
            "SELENITA_OPENAI_MODEL": "stub",
            "SELENITA_OPENAI_API_KEY": "stub-key",
            "SELENITA_SESSIONS_DIR": str(tmpsessions),
            "SELENITA_LOG_LEVEL": "error",
        },
        permission_decisions=decisions,
    )
    return client


async def _setup_session(server: FakeResponsesServer, tmpsessions: Path,
                         decisions: list[str] | None, workdir: Path) -> tuple[AcpTestClient, str]:
    import asyncio

    client = await _launch_llm_client(tmpsessions, server, decisions)
    await client.initialize()
    new = await client.exchange("session/new", {"cwd": str(workdir), "mcpServers": []})
    # give the subprocess event loop scheduling room
    await asyncio.sleep(0)
    return client, new.result["sessionId"]


def _tool_call_requests(exchange) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for req in exchange.permission_requests:
        if req["method"] == "session/request_permission":
            tool_call = req["params"].get("toolCall", {})
            out.append((req["params"].get("sessionId", ""), tool_call))
    return out


@pytest.fixture
async def server() -> FakeResponsesServer:
    srv = FakeResponsesServer()
    await srv.start()
    yield srv
    await srv.close()


@pytest.fixture(autouse=True)
def _timeout(monkeypatch: pytest.MonkeyPatch):
    import tests.integration._harness as harness

    monkeypatch.setattr(harness, "TEST_TIMEOUT", _TIMEOUT)


async def test_allow_once_runs_the_tool(tmp_path: Path, server: FakeResponsesServer) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    target = workdir / "e2e.txt"
    server.queue(
        FunctionCallTurn(
            name="selenita_write_file",
            arguments=json.dumps({"path": str(target), "content": "HOLA"}),
        ),
        MessageTurn("file written"),
    )
    client, sid = await _setup_session(server, tmp_path / "sessions", None, workdir)
    try:
        ex = await client.prompt(sid, ["write the greeting file"])
        updates = _updates(ex)
        kinds = [u["sessionUpdate"] for u in updates]
        assert "tool_call" in kinds
        assert "tool_call_update" in kinds
        progress = next(u for u in updates if u["sessionUpdate"] == "tool_call_update")
        assert progress["status"] == "completed"
        assert progress["rawOutput"].strip().startswith("Wrote")
        assert [u["content"]["text"] for u in updates
                if u["sessionUpdate"] == "agent_message_chunk"] == ["file written"]

        # The permission round trip: one request with the three standard options.
        requests = _tool_call_requests(ex)
        assert len(requests) == 1
        _, tool_call = requests[0]
        assert tool_call["title"] == "Write file"
        assert tool_call["rawInput"]["path"] == str(target)
        option_kinds = [o["kind"] for o in ex.permission_requests[0]["params"]["options"]]
        assert option_kinds == ["allow_once", "allow_always", "reject_once"]

        assert target.read_text() == "HOLA"
    finally:
        await client.aclose()


async def test_reject_once_denies_and_reports_to_the_model(
    tmp_path: Path, server: FakeResponsesServer,
) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    target = workdir / "e2e.txt"
    server.queue(
        FunctionCallTurn(
            name="selenita_write_file",
            arguments=json.dumps({"path": str(target), "content": "HOLA"}),
        ),
        MessageTurn("understood, not writing"),
    )
    client, sid = await _setup_session(server, tmp_path / "sessions",
                                       ["reject_once"], workdir)
    try:
        ex = await client.prompt(sid, ["write the file"])
        updates = _updates(ex)
        progress = next(u for u in updates if u["sessionUpdate"] == "tool_call_update")
        assert progress["status"] == "failed"
        # The denial message must reach the model on the next request (stateless replay).
        next_input = json.dumps(server.requests[-1])
        assert "Permission denied by the user" in next_input
        assert not target.exists()
        assert "understood, not writing" in [u["content"]["text"] for u in updates
                                             if u["sessionUpdate"] == "agent_message_chunk"]
    finally:
        await client.aclose()


async def test_allow_always_caches_for_identical_calls(
    tmp_path: Path, server: FakeResponsesServer,
) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    target = workdir / "e2e.txt"
    server.queue(
        FunctionCallTurn(
            name="selenita_write_file",
            arguments=json.dumps({"path": str(target), "content": "uno"}),
        ),
        FunctionCallTurn(
            name="selenita_write_file",
            arguments=json.dumps({"path": str(target), "content": "dos"}),
        ),
        MessageTurn("both writes applied"),
    )
    client, sid = await _setup_session(server, tmp_path / "sessions",
                                       ["allow_always"], workdir)
    try:
        ex = await client.prompt(sid, ["write the file twice"])
        updates = _updates(ex)
        completed = [u for u in updates if u["sessionUpdate"] == "tool_call_update"
                     and u["status"] == "completed"]
        assert len(completed) == 2
        # allow_always must cache: only ONE permission request for both calls.
        assert len(_tool_call_requests(ex)) == 1
        assert target.read_text() == "dos"
    finally:
        await client.aclose()


async def test_cancelled_denies_fail_closed(tmp_path: Path,
                                            server: FakeResponsesServer) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    target = workdir / "e2e.txt"
    server.queue(
        FunctionCallTurn(
            name="selenita_write_file",
            arguments=json.dumps({"path": str(target), "content": "x"}),
        ),
        MessageTurn("cancelled ok"),
    )
    client, sid = await _setup_session(server, tmp_path / "sessions",
                                       ["cancelled"], workdir)
    try:
        ex = await client.prompt(sid, ["write the file"])
        updates = _updates(ex)
        progress = next(u for u in updates if u["sessionUpdate"] == "tool_call_update")
        assert progress["status"] == "failed"
        assert not target.exists()
    finally:
        await client.aclose()
