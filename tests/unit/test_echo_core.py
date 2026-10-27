"""Unit tests of the echo core (the logic shared by the three entry-point modes)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from selenita.config import Settings
from selenita.core.echo import EchoCore
from selenita.sessions.store import InMemorySessionStore, JsonlSessionStore


def make_core() -> tuple[EchoCore, InMemorySessionStore]:
    store = InMemorySessionStore()
    return EchoCore(store=store), store


async def collect(core: EchoCore, session_id: str, prompt: str) -> list:
    return [chunk async for chunk in core.respond(session_id, prompt)]


def test_core_creates_valid_session_ids() -> None:
    core, _ = make_core()
    sid = core.new_session_id()
    assert sid.startswith("sess_")
    assert sid != core.new_session_id()


async def test_respond_echoes_text_and_persists_exchange() -> None:
    core, store = make_core()
    chunks = await collect(core, "sess_x1", "hello world")
    assert [(c.kind, c.text) for c in chunks] == [("answer", "hello world")]

    history = store.load("sess_x1")
    assert [(m.role, m.text) for m in history] == [
        ("user", "hello world"),
        ("agent", "hello world"),
    ]


async def test_respond_keeps_multiline_in_a_single_chunk() -> None:
    core, _ = make_core()
    chunks = await collect(core, "sess_x2", "line one\nline two")
    assert [(c.kind, c.text) for c in chunks] == [("answer", "line one\nline two")]


async def test_empty_prompt_yields_nothing_and_persists_nothing() -> None:
    core, store = make_core()
    chunks = await collect(core, "sess_x3", "")
    assert chunks == []
    assert store.load("sess_x3") == []


def test_core_with_settings_persists_to_env_directory(tmp_path: Path) -> None:
    core = EchoCore(settings=Settings(sessions_dir=tmp_path))
    sid = core.new_session_id()
    import asyncio

    async def one_turn() -> None:
        async for _ in core.respond(sid, "persisted"):
            pass

    asyncio.run(one_turn())
    assert JsonlSessionStore(tmp_path).load(sid) != []


def test_store_failure_degrades_gracefully_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If the disk fails, the turn goes on and only the log notices (nothing on stdout)."""

    class BrokenStore:
        def append(self, session_id: str, message: object) -> None:
            raise OSError("disk full")

        def load(self, session_id: str) -> list:
            return []

        def exists(self, session_id: str) -> bool:
            return False

    core = EchoCore(store=BrokenStore())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        import asyncio

        chunks = asyncio.run(_collect(core, "sess_x9"))

    assert [c.text for c in chunks] == ["hello"]
    assert any("could not persist" in r.message for r in caplog.records)


async def _collect(core: EchoCore, session_id: str) -> list:
    return [chunk async for chunk in core.respond(session_id, "hello")]
