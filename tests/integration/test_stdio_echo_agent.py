"""Integration tests of the full ACP flow over the real stdio transport.

They run the server as a subprocess (`python -m selenita --acp`) speaking JSON-RPC NDJSON,
exactly like an ACP client (Zed) would. They need the package installed in the environment
(``uv sync`` / ``pip install -e .``).
"""

from __future__ import annotations

import pytest

from tests.integration._harness import AcpTestClient


async def test_full_acp_conversation() -> None:
    """initialize → session/new → session/prompt (echo chunks) → stopReason end_turn."""
    client = await AcpTestClient.launch()
    try:
        init = await client.initialize()
        assert init["protocolVersion"] == 1

        session = await client.new_session(cwd="/tmp")
        assert session["sessionId"].startswith("sess_")

        exchange = await client.prompt(session["sessionId"], ["hello", "selenita"])
        assert exchange.result["stopReason"] == "end_turn"
        assert exchange.chunk_texts() == ["hello", "selenita"]
        assert exchange.methods() == ["session/update", "session/update"]
    finally:
        await client.aclose()


async def test_second_session_is_independent() -> None:
    client = await AcpTestClient.launch()
    try:
        await client.initialize()
        first = (await client.new_session())["sessionId"]
        second = (await client.new_session())["sessionId"]
        assert first != second

        ex_first = await client.prompt(first, ["first"])
        assert ex_first.chunk_texts() == ["first"]
        ex_second = await client.prompt(second, ["second"])
        assert ex_second.chunk_texts() == ["second"]
    finally:
        await client.aclose()


async def test_prompt_with_empty_text_blocks() -> None:
    client = await AcpTestClient.launch()
    try:
        await client.initialize()
        session = (await client.new_session())["sessionId"]
        exchange = await client.prompt(session, [])
        assert exchange.result["stopReason"] == "end_turn"
        assert exchange.notifications == []
    finally:
        await client.aclose()


@pytest.mark.parametrize("arg", ["--port"])
async def test_unimplemented_flags_exit_cleanly(arg: str) -> None:
    """Announced but unimplemented flags exit with a controlled error, not a crash."""
    import asyncio
    import sys

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "selenita", "--acp", arg, "9999",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), 15)
        assert proc.returncode == 2
        assert b"TCP NDJSON transport (--port)" in stderr  # clear, controlled parser error
    finally:
        if proc.returncode is None:
            proc.kill()
